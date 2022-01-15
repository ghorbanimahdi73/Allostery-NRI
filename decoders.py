# Author Mahdi Ghorbani 
# Email: (ghorbani.mahdi73@gmail.com)
# Initial code was taken from the Original NRI implementation by Thomas Kipf https://github.com/ethanfetaya/NRI

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax
from layer_utils import CNN, MLP, myGRU, LinAct
_EPS = 1e-10

class MLPDecoder(nn.Module):
	'''
	MLPDecoder Module in NRI
	params:
		n_in_node: (int)
			number of features for each node
		edge_types: (int)
			number of edge types 
		msg_hid: (int)
			number of hidden units for msg passing
		msg_out: (int)
			number of hidden units after msg passing 
		n_hid: (int)
			number of hidden units
		do_prob: (float)
			dropout for decoder
		skip_first: (binary)
			whether to skip first interaction type
		dynamic_adj: (binary)
			dynamic adjacency is False for MLPDecoder
	'''
	def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid, do_prob=0., skip_first=True, dynamic_adj=False):
		super(MLPDecoder, self).__init__()
		self.msg_fc1 = nn.ModuleList([nn.Linear(2*n_in_node, msg_hid) for _ in range(edge_types)]) # one MLP for each edge type
		self.msg_fc2 = nn.ModuleList([nn.Linear(msg_hid, msg_out) for _ in range(edge_types)]) # one MLP for each edge type

		self.msg_out_shape = msg_out
		self.skip_first_edge_type = skip_first

		self.out_fc1 = nn.Linear(n_in_node+msg_out, n_hid)
		self.out_fc2 = nn.Linear(n_hid, n_hid)
		self.out_fc3 = nn.Linear(n_hid, n_in_node)

		print('Using learned interaction net decoder')
		self.dropout_prob = do_prob

	def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send, single_timesteps_rel_type):
		# single_timestep_inputs [batch_size, num_timesteps, num_atoms, feature_dim]
		# single_timestep_rel_type [batch_size, num_timesteps, num_edges, num_edge_types]
		# rel_rec/rel_send [num_edges, num_nodes]

		# ------------------- node2edge ------------------
		receivers = torch.matmul(rel_rec, single_timestep_inputs) 
		senders = torch.matmul(rel_send, single_timestep_inputs)\
		# [batch_size, num_timesteps, num_edges, feature_dim]

		pre_msg = torch.cat([senders, receivers], dim=-1)
		all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), pre_msg.size(2), self.msg_out_shape))

		if single_timestep_inputs.is_cuda:
			all_msgs = all_msgs.cuda()

		if self.skip_first_edge_type:
			# hard-coding first rel_type to be no interaction
			start_idx = 1
		else:
			start_idx = 0

		# run separate MLP for each edge-type
		for i in range(start_idx, len(self.msg_fc2)):
			msg = F.relu(self.msg_fc1[i](pre_msg))
			msg = F.dropout(msg, self.dropout_prob)
			msg = F.relu(self.msg_fc2[i](msg))
			msg = msg * single_timesteps_rel_type[:,:,:,i:i+1]
			all_msgs += msg

		# -------------------- edge2node -------------------
		# aggregate all msgs to receiver
		# all_msgs [batch_size, num_timesteps, num_edges, msg_out]
		# rel_rec [num_edges, num_nodes]
		agg_msgs = all_msgs.transpose(-2,-1).matmul(rel_rec).transpose(-2,-1)
		# [batch_size, num_timesteps, num_atoms, msg_out]
		agg_msgs = agg_msgs.contiguous()

		# skip connection
		aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)
		# [batch_size, num_timesteps, num_atoms, msg_out + feature_dim]

		# output MLP
		pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
		# [batch_size, num_timesteps, num_atoms, n_hid]
		pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
		# [batch_size, num_timesteps, num_atoms, n_hid]
		pred = self.out_fc3(pred)
		# [batch_size, num_timesteps, num_atoms, n_in_node]

		# predict position/velocity difference
		return single_timestep_inputs + pred

	def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
		# MSM just uses current step to predict future
		# rel_type [batch_size, num_edges, num_edge_types]
		# inputs [batch-size, num_timesteps, num_atoms, num_dims]

		inputs = inputs.transpose(1,2).contiguous()
		# [batch-size, num_timesteps, num-atoms, feature_dim]
		sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1), rel_type.size(2)]
		# [batch-size, num_timesteps, num_edges, num_edge_types]

		# rel_type [batch_size, num-edges, num_edge_types]
		rel_type = rel_type.unsqueeze(1).expand(sizes)
		# repeat rel-type 'timestep' times
		# shape [batch-size, num_timesteps, num_edges, num_edge_types]

		timesteps = inputs.size(1)
		#assert (pred_steps <= timesteps)
		preds = []

		# only take nth timestep as starting point (n: pred_steps) # teacher forcing
		last_pred = inputs[:, 0::pred_steps, :, :]
		curr_rel_type = rel_type[:, 0::pred_steps, :, :]
		# note assume rel_type is constant

		# run n prediction steps
		for step in range(0, pred_steps):
			last_pred = self.single_step_forward(last_pred, rel_rec, rel_send, curr_rel_type)
			preds.append(last_pred)

		sizes = [preds[0].size(0), preds[0].size(1)*pred_steps, preds[0].size(2), preds[0].size(3)]
		output = Variable(torch.zeros(sizes))

		if inputs.is_cuda:
			output = output.cuda()
		# reassemble correct timeline
		for i in range(len(preds)):
			output[:, i::pred_steps, :, :] = preds[i]

		pred_all = output[:, :(inputs.size(1)-1), :, :]
		return pred_all.transpose(1,2).contiguous()


class RNNDecoder(nn.Module):
	'''
	Recurrent decoder of NRI

	params:
	----------------
		n_in_node: (int)
			input dimension
		edge_types: (int)
			number of edge types
		n_hid: (int)
			number of hidden units in RNN
		do_prob: (float)
			dropout probability
		skip_first: (bool)
			whether to skip first edge type 
		dynamic_adj: (bool)
			Whether to use dynamic adjacency for decoder
	'''
	def __init__(self, n_in_node, edge_types, n_hid, do_prob=0., skip_first=False, dynamic_adj=False):
		# n_in_node = input dimension
		super(RNNDecoder, self).__init__()
		self.msg_fc1 = nn.ModuleList([nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)]) # one for each relation type
		self.msg_fc2 = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(edge_types)]) # one for each relation type

		self.msg_out_shape = n_hid
		self.skip_first_edge_type = skip_first
		self.dynamic_adj = dynamic_adj 

		self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
		self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
		self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

		self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
		self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
		self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

		self.out_fc1 = nn.Linear(n_hid, n_hid)
		self.out_fc2 = nn.Linear(n_hid, n_hid)
		self.out_fc3 = nn.Linear(n_hid, n_in_node)
		print('Using learned recurrent interaction net decoder')
		self.dropout_prob = do_prob

	def single_step_forward(self, inputs, rel_rec, rel_send, rel_type, hidden):
		'''
		inputs:
		-------------------------------------------------------
			inputs: [batch_size, num_nodes, feature_dim]
				input trajectory from data
			rel_rec: [batch_size, num_edges, num_nodes]
				indices of receiving nodes
			rel_send: [batch-size, num_edges, num_nodes]
				indices of sending nodes
			hidden: [batch_size, num_nodes, msg_out_shape]
				hidden state of previous timestep in RNN
		-------------------------------------------------------
		returns:
			pred: [batch_size, num_nodes, msg_out_shape]
				prediction of position/velocity of next step
			hidden: [batch_size, num_nodes, msg_out_shape]
				hidden state of the RNN

		'''
		#------------------- node2edge -------------------
		receivers = torch.matmul(rel_rec, hidden)
		senders = torch.matmul(rel_send, hidden)
		# [batch_size, num-edges, n_hid]

		pre_msg = torch.cat([senders, receivers], dim=-1)
		# pre_msg [batch-size, num-edges, n_hid*2]
		all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape))

		# all_msgs [batch_size, num_edges, msg_out_shape]
		if inputs.is_cuda:
			all_msgs = all_msgs.cuda()

		if self.skip_first_edge_type:
			start_idx = 1
			norm = float(len(self.msg_fc2)) - 1
		else:
			start_idx = 0
			norm = float(len(self.msg_fc2))

		# run separate MLP for each edge type
		# note: to exclude one edge-type, simple offset range by 1
		for i in range(start_idx, len(self.msg_fc2)):
			msg = F.tanh(self.msg_fc1[i](pre_msg))
			msg = F.dropout(msg, p=self.dropout_prob)
			msg = F.tanh(self.msg_fc2[i](msg))
			msg = msg * rel_type[:, :, i:i+1]
			all_msgs += msg/norm # normalize msgs by number of nodes

		# ---------------------- edge2node ----------------
		# all_msgs [batch_size, num_edges, msg_out_shape]
		agg_msgs = all_msgs.transpose(-2,-1).matmul(rel_rec).transpose(-2,-1)
		agg_msgs = agg_msgs.contiguous() / inputs.size(2) # average

		# agg_msgs [batch_size, num_nodes, msg_out_shape]
		# gru-style gated aggregation
		r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
		i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
		n = F.tanh(self.input_n(inputs) + r*self.hidden_h(agg_msgs))
		hidden = (1-i) * n + i * hidden
		# hidden [batch_size, num_nodes, msg_out_shape]
		# output MLP
		pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
		pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
		pred = self.out_fc3(pred)

		# pred [batch-size, num_nodes, num_features (6)]
		# prediction of position/velocity different
		pred = inputs + pred
		return pred, hidden

	def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1, burn_in=False,
		        burn_in_steps=1, dynamic_graph=False, encoder=None, temp=None):
		'''
		inputs:
			data: [batch-size, timesteps, num_nodes, feature_dim]
				the batch of trajectory data

			rel_type: [batch-size, num-edges edge_types]
				the sampled relations from encoder

			rel_rec/rel_send if dynamic_adj then [batch_size, timesteps, num_edges, nm_nodes]
							  other wise this will be [num_edges, num_nodes]

			pred_steps: (int) 
				number of prediction steps

			burn_in: (bool)
				Whether to use a portion of data (first few timesteps) for training and then predict the next (20) timesteps

			burn_in_steps: (int)
				number of steps to burn
				
		'''

		inputs = data.transpose(1,2).contiguous()
		time_steps = inputs.size(1)
		# data [batch_size, timesteps, num_nodes, feature_dim]
		# rel_type [batch-size, num-edges, edge_types]
		# if dynamic_adj then rel_rec/rel_send [batch-size, timesteps, num_edges, num_nodes]


		hidden = Variable(torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
		# [batch_size, num_nodes, msg_out_shape]
		if inputs.is_cuda:
			hidden = hidden.cuda()

		if not self.dynamic_adj:
			rel_rec_t = rel_rec
			rel_send_t = rel_send

		pred_all = []
		for step in range(0, inputs.size(1)-1):
			if burn_in:
				if step <= burn_in_steps:
					ins = inputs[:, step, :, :]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
				else:
					ins = pred_all[step-1]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]	
			else:
				assert (pred_steps <= time_steps)
				# Use ground truth trajectory inputs vs last prediction
				if not step % pred_steps:
					ins = inputs[:, step, :, :]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
				else:
					ins = pred_all[step-1]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
			if dynamic_graph and step >= burn_in_steps:
				# note assume burn_in-steps = args.timesteps
				logits = encoder(data[:, :, step - burn_in_steps:step, :].contiguous(), rel_rec_t, rel_send_t)
				rel_type = gumbel_softmax(logits, tau=temp, hard=True)

			# rel_rec, rel_send [batch_size, num_edges, num_nodes]
			pred, hidden = self.single_step_forward(ins, rel_rec_t, rel_send_t, rel_type, hidden)
			pred_all.append(pred)

		preds = torch.stack(pred_all, dim=1)
		# preds [batch-size, num_timesteps-1, num_nodes, feature_dim]
		return preds.transpose(1,2).contiguous()


class RNNSpatioTemporalDecoder(nn.Module):
	'''
	RNN Spatio-Temporal decoder implementation based on paper:
	'neural relational inference with efficient message passing' 
	a spatio-temporal msg passing for node2edge
	a spatio-temporal msg passing for edge2node
	
	params:
	----------------
		n_in_node: (int)
			input dimension
		edge_types: (int)
			number of edge types
		n_hid: (int)
			number of hidden units in RNN
		do_prob: (float)
			dropout probability
		skip_first: (bool)
			whether to skip first edge type 
		dynamic_adj: (bool)
			Whether to use dynamic adjacency for decoder
	'''
	def __init__(self, n_in_node, edge_types, n_hid, do_prob=0, skip_first=False, dynamic_adj=False):
		# n_in_node = input dimension
		super(RNNSpatioTemporalDecoder, self).__init__()

		self.dynamic_adj = dynamic_adj
		self.msg_fc1 = nn.ModuleList([nn.Linear(2*n_in_node, n_hid) for _ in range(edge_types)])
		self.msg_fc2 = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(edge_types)]) # one MLP for each edge type

		self.msg_out_shape = n_hid
		self.skip_first_edge_type = skip_first
		self.n_in_node = n_in_node
		self.dropout_prob = do_prob

		self.gru_edge = myGRU(n_hid, n_hid)
		self.gru_node = myGRU(n_hid+n_in_node, n_hid+n_in_node)

		self.out_fc1 = nn.Linear(n_in_node+n_hid, n_hid)
		self.out_fc2 = nn.Linear(n_hid, n_hid)
		self.out_fc3 = nn.Linear(n_hid, n_in_node)

		print('Using learned interaction net decoder')

	def single_step_forward(self, inputs, rel_rec, rel_send, rel_type, hidden_node, hidden_edge):

		'''
		inputs:
		-------------------------------------------------------
			inputs: [batch_size, num_nodes, feature_dim]
				input trajectory from data
			rel_rec: [batch_size, num_edges, num_nodes]
				indices of receiving nodes
			rel_send: [batch-size, num_edges, num_nodes]
				indices of sending nodes
			rel_types: [batch_size, num_edges, num_edge_types]
				the relations types
			hidden_node: [batch_size, num_nodes, hidden_dim]
				hidden state of previous timestep in node2edge GRU
			hidden_edge : [batch_size, num_edges, hidden_dim]
				hidden state of previous timestep in edge2node GRU

		returns:
			pred: [batch_size, num_nodes, msg_out_shape]
				prediction of position/velocity of next step
			cat : [batch_size, num_nodes, msg_out_shpae+n_in_node]

			msgs: [batch_size, num_nodes, msg_out_shape]
				hidden state of the RNN

		'''

		# ----------------- node2edge ----------------------
		receivers = torch.matmul(rel_rec, inputs)
		senders = torch.matmul(rel_send, inputs)
		pre_msg = torch.cat([senders, receivers], dim=-1)
		# pre_msgs [batch-size, num_edges, n_hid*2]
		all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape))

		# all_msgs [batch_size, num_edges, msg_out_shape]
		if inputs.is_cuda:
			all_msgs = all_msgs.cuda()

		if self.skip_first_edge_type:
			start_idx = 1
			norm = float(len(self.msg_fc2)) - 1
		else:
			start_idx = 0
			norm = float(len(self.msg_fc2))

		# run separate MLP for each edge type
		# note: to exclude one edge type, simply offset range by 1
		for i in range(start_idx, len(self.msg_fc2)):
			msg = F.tanh(self.msg_fc1[i](pre_msg))
			msg = F.dropout(msg, p=self.dropout_prob)
			msg = F.tanh(self.msg_fc2[i](msg))
			msg = msg*rel_type[:,:,i:i+1]
			all_msgs += msg/norm

		msgs = all_msgs
		# all_msgs [batch_size, num_edges, msg_out_shape]
		if hidden_edge is not None:
			msgs = self.gru_edge(all_msgs, hidden_edge)

		# ------------------ edge2node ---------------------
		agg_msgs = msgs.transpose(-2,-1).matmul(rel_rec).transpose(-2,-1)
		agg_msgs = agg_msgs.contiguous() / inputs.size(2)
		# agg_msgs [batch_size, num_nodes, msg_out_shpae]

		cat = torch.cat([inputs, agg_msgs], dim=-1) # n_in_node + msg_out
		if hidden_node is not None:
			cat = self.gru_node(cat, hidden_node)

		# gru-style gated aggregation
		# hidden [batch_size, num_nodes, msg_out_shape]
		# output MLP
		pred = F.dropout(F.relu(self.out_fc1(cat)), p=self.dropout_prob)
		pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
		pred = self.out_fc3(pred)
		# pred [batch_size, num_nodes, num-features]
		# predict pos/vel difference
		pred = inputs + pred
		return pred, cat, msgs

	def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1, burn_in=False,
		        burn_in_steps=1, dynamic_graph=False, encoder=None, temp=None):
		'''
		inputs:
			data: [batch-size, timesteps, num_nodes, feature_dim]
				the batch of trajectory data
			rel_type: [batch-size, num-edges edge_types]
				the sampled relations from encoder
			rel_rec/rel_send if dynamic_adj then [batch_size, timesteps, num_edges, nm_nodes]
							  other wise this will be [num_edges, num_nodes]
			pred_steps: (int) 
				number of prediction steps
			burn_in: (bool)
				Whether to use a portion of data (first few timesteps) for training and then predict the next (20) timesteps
			burn_in_steps: (int)
				number of steps to burn
				
		'''
		inputs = data.transpose(1,2).contiguous()
		time_steps = inputs.size(1)

		# data shape [batch_size, num_timtimsteps, num_nodes, feature_dim]
		# rel_type [batch_size, num_timesteps, num_edges, edge_types]

		hidden_node, hidden_edge = None, None
		pred_all = []
		for step in range(0, inputs.size(1)-1):
			if burn_in:
				if (step <= burn_in_steps):
					ins = inputs[:, step, :, :]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
				else:
					ins = pred_all[step-1]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
			else:
				assert (pred_steps <= time_steps)
				# use ground truth trajectory inputs vs last prediction
				if not step % pred_steps:
					ins = inputs[:, step, :, :]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
				else:
					ins = pred_all[step-1]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
			if dynamic_graph and step >= burn_in_steps:
				# note assume burn_in-steps = args.timesteps
				logits = encoder(data[:, :, step-burn_in_steps:step, :].contiguous(), rel_rec_t, rel_send_t)
				rel_type = gumbel_softmax(logits, tau=temp, hard=True)

			pred, hidden_node, hidden_edge = self.single_step_forward(ins,rel_rec_t, rel_send_t, rel_type, hidden_node, hidden_edge)
			pred_all.append(pred)

		preds = torch.stack(pred_all, dim=1)
		# preds [batch_size, num_timesteps-1, num_nodes, feature_dim]
		return preds.transpose(1,2).contiguous()
		

class AttnSpatioTemporalDecoder(nn.Module):
	'''
	Attn-based SpatioTemporal Decoder module
	Adding attention mechanism into Spatio-temporal msg passing (node2edge and edge2node)

	params:
	----------------
		n_in_node: (int)
			input dimension
		edge_types: (int)
			number of edge types
		n_hid: (int)
			number of hidden units in RNN
		attn_hid: (int)
			number of units for attention
		do_prob: (float)
			dropout probability
		skip_first: (bool)
			whether to skip first edge type 
		dynamic_adj: (bool)
			Whether to use dynamic adjecency matrix
	'''
	def __init__(self, n_in_node, edge_types, n_hid, attn_hid, do_prob=0, skip_first=False, dynamic_adj=False):
		super(AttnSpatioTemporalDecoder, self).__init__()
		self.input_emb = nn.Linear(n_in_node, n_hid)
		self.msg_fc1 = nn.ModuleList([nn.Linear(2*(n_in_node+n_hid), n_hid) for _ in range(edge_types)])
		self.msg_fc2 = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(edge_types)])

		self.msg_out_shape = n_hid
		self.skip_first_edge_type = skip_first
		self.n_in_node = n_in_node

		self.gru_edge = myGRU(n_hid, n_hid)
		self.gru_node = myGRU(n_hid+n_in_node, n_hid+n_in_node)

		self.out_fc1 = nn.Linear(n_in_node+n_hid, n_hid)
		self.out_fc2 = nn.Linear(n_hid, n_hid)
		self.out_fc3 = nn.Linear(n_hid, n_in_node)

		self.dynamic_adj = dynamic_adj
		self.attn_hid = attn_hid

		print('Using learned recurrent interaction net decdoer')

		self.dropout_prob = do_prob

		# attention mechanism
		self.attn = nn.Linear(n_hid+n_in_node, self.attn_hid)
		self.query = LinAct(n_in_node+n_hid, self.attn_hid)
		self.key = LinAct(n_in_node+n_hid, self.attn_hid)
		self.value = LinAct(n_in_node+n_hid, self.attn_hid)
		self.att_out = LinAct(self.attn_hid, n_in_node+n_hid)

	def temporal_attention(self, x, h):
		'''
		Update hidden states of nodes by temporal attention mechanism
		args:
			x: [step_attn, batch-size, num_nodes, n_in_node + n_hid] historical hidden states of temporal attention
			h: [batch_size, num_nodes, n_in_node + n_hid] hidden states of nodes from RNN

		returns:
			output [batch-size, num_nodes, n_in_node + n_hid] hidden state of nodes updated by attention
			out_attn [batch_size, num_nodes, step_attn] attention of historical steps w.r.t current step

		'''
		# concatenate current hidden states of nodes with historical hidden states
		h_current = h.unsqueeze(0).contiguous()
		# [1, batch-size, num_nodes,  n_hid + n_in_node]
		x = torch.cat([x, h_current], dim=0)

		# x [batch-size, num_nodes, step_attn+1, n_in_node + n_hid]
		x = x.permute(1, 2, 0, 3).contiguous()
		# query [batch_size, num-nodes, 1, attn_hid]
		query = self.query(h.unsqueeze(2))
		# key/value [batch-size, num_nodes, step_attn, attn_hid] # step-attn increases as we predict more steps
		key = self.key(x)
		value = self.value(x)

		# reshape to [batch_size, num_nodes, attn_hid, step_attn] # step atten goes from 0 to num_timesteps
		key = key.transpose(-2,-1).contiguous()

		# [batch_size, num_nodes, 1, step_attn]
		attention = torch.matmul(query, key) / (self.attn_hid**0.5)
		attention = attention.softmax(-1)

		# [batch_size, num_nodes, attn_hid]
		attn_value = torch.matmul(attention, value).squeeze(2)

		# [batch_size, num_nodes, n_in_node + n_hid]
		output = self.att_out(attn_value)

		# [batch_size, num_nodes, step-attn]
		out_attn = attention.squeeze(2).contiguous()
		return output, out_attn

	def single_step_forward(self, inputs, rel_rec, rel_send, rel_type, hidden_node, hidden_edge, hidden_attention):
		'''
		args:
			inputs: [batch-size, num_atoms, features]
				the features at current timestep
			rel_rec/rel_send [batch_size, num_nodes, num_edges]
				rel_rec/reL_send at current time
			rel_type = [batch_size, num_edges, num-edges_types]
				the type of relations from encoder
			hidden_node [batch_size, num_nodes, hidden_dim]
				hidden state of nodes at current time
			hidden_edge [batch_size, num_edges, hidden_dim]
				hidden state of edges at current time
			hidden_attention [step_attn, batch_size, num_nodes, dim]
				the attention hidden state
		
		returns:
			pred: [batch_size, num_nodes, n_in_node]

			hidden_attention: [step_attn, batch_size, num_nodes, n_in_node + n_hid]

			cat: [batch_size, num_nodes, n_in_node + n_hid]

			msgs: [batch_size, num_edges, n_hid]

		'''
		x_emb = self.input_emb(inputs)
		# [batch_size, num_atoms, n_hid]
		x_emb = torch.cat([x_emb, inputs], dim=-1)
		# [batch_size, num_atoms , n_in_node + n_hid]
		receivers = torch.matmul(rel_rec, x_emb)
		senders = torch.matmul(rel_send, x_emb)
		# [batch_size, num_edges, n_in_node + n_hid]
		pre_msg = torch.cat([senders, receivers], dim=-1)
		# [batch_size, num_edges, 2*(n_in_node+n_hid)]
		
		all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape))
		# [batch_size, num_edges, msg_out_shape]
		if inputs.is_cuda:
			all_msgs = all_msgs.cuda()

		if self.skip_first_edge_type:
			start_idx = 1
			norm = float(len(self.msg_fc2)) - 1
		else:
			start_idx = 0
			norm = float(len(self.msg_fc2))


		for i in range(start_idx, len(self.msg_fc2)):
			msg = F.tanh(self.msg_fc1[i](pre_msg))
			msg = F.dropout(msg, p=self.dropout_prob)
			msg = F.tanh(self.msg_fc2[i](msg))
			msg = msg * rel_type[:, :, i:i+1]
			all_msgs += msg/norm
		msgs = all_msgs
		# [batch-size, num_edges, msg_out_shape]

		if hidden_edge is not None:
			msgs = self.gru_edge(all_msgs, hidden_edge)
		# [batch_size, num-edges, n_hid]

		agg_msgs = msgs.transpose(-2,-1).matmul(rel_rec).transpose(-2,-1)
		agg_msgs = agg_msgs.contiguous() / inputs.size(2)
		# agg_msgs [batch_size, num_nodes, n_hid]

		cat = torch.cat([inputs, agg_msgs], dim=-1)
		# cat [batch_size, num_nodes, n_hid+n_in_node]

		if hidden_node is None:
			pred = F.dropout(F.relu(self.out_fc1(cat)), p=self.dropout_prob)
			pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
			pred = self.out_fc3(pred)
			hidden_attention = cat.unsqueeze(0)
			# [1, batch_size, num-nodes, n_in_node + n_hid]
		else:
			cat = self.gru_node(cat, hidden_node)
			# cat [batch_size, num_nodes, n_hid + n_in_node]
			cur_hidden, _ = self.temporal_attention(hidden_attention, cat)
			# cur_hidden [batch_size, num_nodes, n_in_node + n_hid]

			hidden_attention = torch.cat([hidden_attention, cur_hidden.unsqueeze(0)], dim=0)

			pred = F.dropout(F.relu(self.out_fc1(cur_hidden)), p=self.dropout_prob)
			pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
			pred = self.out_fc3(pred)

		pred = inputs + pred

		return pred, hidden_attention, cat, msgs

	def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1, burn_in=False, 
		        burn_in_steps=1, dynamic_graph=False,  encoder=None, temp=None):
		'''
		inputs:
			data: [batch-size, timesteps, num_nodes, feature_dim]
				the batch of trajectory data
			rel_type: [batch-size, num-edges edge_types]
				the sampled relations from encoder
			rel_rec/rel_send if dynamic_adj then [batch_size, timesteps, num_edges, nm_nodes]
							  other wise this will be [num_edges, num_nodes]
			pred_steps: (int) 
				number of prediction steps
			burn_in: (bool)
				Whether to use a portion of data (first few timesteps) for training and then predict the next (20) timesteps
			burn_in_steps: (int)
				number of steps to burn
			encoder: (bool) Whether to use dynamic_graph
			temp: (float) the temperature parameter for gumbel-softmax

		'''
		inputs = data.transpose(1,2).contiguous()
		time_steps = inputs.size(1)

		hidden_attention, hidden_node, hidden_edge = None, None, None
		pred_all = []
		for step in range(0, inputs.size(1)-1):
			if burn_in:
				if step <= burn_in_steps:
					ins = inputs[:, step, :, :]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
				else:
					ins = pred_all[step-1]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
			else:
				assert (pred_steps <= time_steps)
				if not step % pred_steps:
					# use ground truth trajectory inputs vs last prediction
					ins = inputs[:, step, :, :]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, step, :, :]
				else:
					ins = pred_all[step-1]
					if self.dynamic_adj:
						rel_rec_t = rel_rec[:, step, :, :]
						rel_send_t = rel_send[:, stpe, :, :]
			if dynamic_graph and step >= burn_in_steps:
				# Note assumes burn_in_steps = args.timesteps
				logits = encoder(data[:, :, step-burn_in_steps, :].contiguous(), rel_rec_t, rel_send_t)
				rel_type = gumbel_softmax(logits, tau=temp, hard=True)

			pred, hidden_attention, hidden_node, hidden_edge = self.single_step_forward(ins, rel_rec_t, rel_send_t, rel_type, hidden_node, hidden_edge, hidden_attention)
			pred_all.append(pred)

		preds = torch.stack(pred_all, dim=1)
		# [batch_size, num_timesteps-1, num_nodes, feature-dim]
		return preds.transpose(1,2).contiguous()