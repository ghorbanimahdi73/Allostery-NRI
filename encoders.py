import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax
from layer_utils import CNN, MLP

_EPS = 1e-10

class MLPEncoder(nn.Module):
	'''
	MLPEncoder module of NRI
	here we peform 2 rounds of graph msg passing node2edge and edge2node to eventually produce 
	an embedding for all edges. This info will later be used in the decoder.

	params:
		n_in: (int)
			number of inputs nodes 
		n_hid: (int)
			number of hidden units
		do_prob (float)
			dropout probability
		factor: (binary)
			whether to use factor graph for encoder
		dynamic_adj: binary
			This is False for MLPEncoder 
	'''
	def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, dynamic_adj=False):

		super(MLPEncoder, self).__init__()

		self.factor = factor
		self.dynamic_adj = dynamic_adj
		self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob, no_bn=True)
		self.mlp2 = MLP(n_hid*2, n_hid, n_hid, do_prob, no_bn=True)
		self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob, no_bn=True)

		if self.factor:
			self.mlp4 = MLP(n_hid*3, n_hid, n_hid, do_prob)
			print('Using factor graph MLP Encoder')
		else:
			self.mlp4 = MLP(n_hid*2, n_hid, n_hid, do_prob)
			print('Using MLP Encoder')
		self.fc_out = nn.Linear(n_hid, n_out)
		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal(m.weight.data)
				m.bias.data.fill_(0.1)

	def node2edge(self, x, rel_rec, rel_send):
		# assume the same graph across all samples
		# a dynamic_adj must be used along with a RNNEncoder
		# x [batch-size, num_nodes, hidden_dim]
		# rel_recv/rel_send [num_edges, num_nodes]
		receivers = torch.matmul(rel_rec, x)
		senders = torch.matmul(rel_send, x)
		edges = torch.cat([senders, receivers], dim=2)
		return edges

	def edge2node(self, x, rel_rec, rel_send):
		# assume the same graph across all samples
		# shape of x [batch-size, num_edges, feature]
		incoming = torch.matmul(rel_rec.t(), x)
		# shape [batch_size, num_atoms, n_hid]
		return incoming/incoming.size(1)

	def forward(self, inputs, rel_rec, rel_send):
		'''
		input_shape [batch_size, num_nodes, num_timesteps, feature_dim]
		rel_rec/rel_send [num-edges, num_nodes]
		'''
		x = inputs.view(inputs.size(0), inputs.size(1), -1)
		# new shape [batch_size, num_nodes, num_timesteps * feature_dim]
		x = self.mlp1(x)
		# [batch_size, num_nodes, n_hid]
		x = self.node2edge(x, rel_rec, rel_send)
		# [batch_size, num_edges, n_hid*2]
		x = self.mlp2(x)
		# [batch_size, num_edges, n_hid]
		x_skip = x
		# skip connection

		if self.factor:
			x = self.edge2node(x, rel_rec, rel_send)
			# shape [batch_size, num_nodes, n_hid]
			x = self.mlp3(x)
			# [batch_size, num_nodes, n_hid]
			x = self.node2edge(x, rel_rec, rel_send)
			# shape [batch_size, num_edges, n_hid*2]
			x = torch.cat((x, x_skip), dim=2)
			# shape [batch_size, num_eges, n_hid*3]
			x = self.mlp4(x)
			# [batch_size, num_edges, n_hid]
		else:
			x = self.mlp3(x)
			# [batch_size, edges, n_hid]
			x = torch.cat((x, x_skip), dim=2)
			# [batch_size, edges, n_hid*2]
			x = self.mlp4(x)
			# [batch_size, edges, n_hid]

		return self.fc_out(x) # [batch_size, num_edges, n_hid]


class CNNEncoder(nn.Module):
	def __init__(self, n_in, n_hid, n_out, do_prob, factor=True, dynamic_adj=False):
		super(CNNEncoder, self).__init__()
		'''
		n_in (int): dimension of features
		n_hid (int): number of channels in 1D convolution
		n_out (int): dim of output vectors
		do_prob = dropout
		factor: flag for factor graph CNN
		dynamic_adj: This is False for CNNEncoder
		'''
		self.dropout_prob = do_prob
		self.factor = factor
		self.cnn = CNN(n_in*2, n_hid, n_hid, do_prob)
		self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob, no_bn=True)
		self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob, no_bn=True)
		self.mlp3 = MLP(n_hid*3, n_hid, n_hid, do_prob, no_bn=True)
		self.fc_out = nn.Linear(n_hid, n_out)

		if self.factor:
			print('Using factor graph CNN Encoder')
		else:
			print('Using CNN Encoder')

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal(m.weight.data)
				m.bias.data.fill_(0.1)

	def node2edge_temporal(self, inputs, rel_rec, rel_send):
		# shape [batch_size, num_nodes, num_teimsteps, feature_dim]
		# assuming the same adjacency over all samples
		# rel_rec/rel_send shape [num_edges, num_nodes]
		x = inputs.view(inputs.size(0), inputs.size(1), -1)
		# [batch-size, num_nodes, num_timesteps*feature_dim]
		receivers = torch.matmul(rel_rec, x)
		# [batch-size, num_edges, num_timesteps*num_dim]
		receivers = receivers.view(inputs.size(0)*receivers.size(1), inputs.size(2), inputs.size(3))
		receivers = receivers.transpose(2,1)
		# shape [batch-size * num_edges, feature_dim, num_timesteps]

		senders = torch.matmul(rel_send, x)
		senders = senders.view(inputs.size(0)*senders.size(1), inputs.size(2), inputs.size(3))
		senders = senders.transpose(2,1)
		edges = torch.cat([senders, receivers], dim=1)
		# [batch_size * num_edges, feature_dim*2, num_timesteps]
		return edges

	def edge2node(self, x, rel_rec, rel_send):
		# assume the same graph across all samples
		# shape of x [batch-size, num_edges, feature]
		incoming = torch.matmul(rel_rec.t(), x)
		# shape [batch_size, num_atoms, n_hid]
		return incoming/incoming.size(1)

	def node2edge(self, x, rel_rec, rel_send):
		# shape of x [batch_size, num_nodes, n_hid]
		receivers = torch.matmul(rel_rec, x)
		senders = torch.matmul(rel_send, x)
		edges = torch.cat([senders, receivers], dim=2)
		return edges


	def forward(self, inputs, rel_rec, rel_send):

		# inputs [batch_size, num_atoms, num_timesteps, feature_dim]
		edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
		# [batch-size * num_edges, feature_dim*2, num_timesteps]
		x = self.cnn(edges)
		# [batch_size * num_edges, n_hid, num_timesteps]
		x = x.view(inputs.size(0), rel_rec.size(0), -1)
		# [batch_size, num_edges, n_hid * num_timesteps]
		x = self.mlp1(x)
		x_skip = x
		# shape [batch_size, num_edges, n_hid]

		if self.factor:
			x = self.edge2node(x, rel_rec, rel_send)
			# shape [batch_size, num_atoms, n_hid]
			x = self.mlp2(x)
			x = self.node2edge(x, rel_rec, rel_send)
			# [batch_size, num_edges, n_hid*2]
			x = torch.cat((x, x_skip), dim=2)
			# [batch-size, num_edges, n_hid*3]
			x = self.mlp3(x)
			# [batch_size, num_edges, n_hid]
		return self.fc_out(x)
			# [batch_size, num_edges, num_edge_types]


class RNNEncoder(nn.Module):
	def __init__(self, n_in, n_hid, n_out, rnn_hidden_size, rnn_type='gru', num_layers=1, do_prob=0., factor=True, dynamic_adj=False):
		super(RNNEncoder, self).__init__()
		'''
		n_out is the number of edge types
		'''
		self.factor = factor
		self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob, no_bn=False)
		self.mlp2 = MLP(n_hid*2, n_hid, n_hid, do_prob, no_bn=False)
		self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob, no_bn=False)
		self.mlp4 = MLP(n_hid*3, n_hid, n_hid, do_prob, no_bn=False)

		if rnn_hidden_size is None:
			rnn_hidden_size = n_hid

		if rnn_type == 'lstm':
			self.rnn_encoder = nn.LSTM(n_hid, rnn_hidden_size, batch_first=True)
		elif rnn_type == 'gru':
			self.rnn_encoder = nn.GRU(n_hid, rnn_hidden_size, batch_first=True)
		else:
			raise ValueError('rnn_type should be either lstm or gru') 

		if num_layers == 1:
			self.fc_out = nn.Linear(rnn_hidden_size, n_out)
		else:
			tmp_hidden_size = n_hid
			layers = [nn.Linear(rnn_hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
			for _ in range(num_layers-2):
				layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
				layers.append(nn.ELU(inplace=True))
			layers.append(nn.Linear(tmp_hidden_size, n_out))
			self.fc_out = nn.Sequential(*layers)

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal(m.weight.data)
				m.bias.data.fill_(0.1)

	def edge2node(self, x, rel_rec, rel_send):
		# x has shape [batch_size, num_edges, num_timesteps, n_hid]
		# rel_rec/rel_send [batch_size, num_tiemsteps, num_edges, num_nodes]
		incoming = torch.einsum('ieth, iten -> inth', x, rel_rec)
		return incoming / incoming.size(1)
		# [batch_size, num_nodes, num_timesteps, n_hid]


	def node2edge(self, x, rel_rec, rel_send):
		# x has shape [batch_size, num_nodes, num_tiemsteps, n_hid]
		# rel_rec/rel_send [batch_size, num_timesteps, num_edges, num_nodes]

		receivers = torch.einsum('inth, iten -> ieth', x, rel_rec)
		senders = torch.einsum('inth, iten -> ieth', x, rel_send)
		# [batch-size, num_edges, num_timesteps, n_hid]
		edges = torch.cat([senders, receivers], dim=3)
		#  [batch-size, num_edges, num_timesteps, n_hid * 2]
		return edges


	def forward(self, inputs, rel_rec, rel_send):
		# input_shape [batch_size, num_nodes, num_timesteps, feature_dim]
		# rel_rec, rel_send [batch_size, num_timesteps, num_edges, num_nodes]
		num_timesteps = inputs.size(2)
		x = self.mlp1(inputs)
		# [batch_size, num_nodes, num_timesteps, n_hid]
		x = self.node2edge(x, rel_rec, rel_send)
		# [batch_size, num_edges, num_timesteps, n_hid*2]
		x = self.mlp2(x)
		# [batch_size, num_edges, num_timesteps, n_hid]
		x_skip = x

		if self.factor:
			x = self.edge2node(x, rel_rec, rel_send)
			# [batch_size, num_nodes, num_timesteps, n_hid]
			x = self.mlp3(x)
			x = self.node2edge(x, rel_rec, rel_send)
			# shape [batch_size, num_edges, num_timesteps, n_hid*2]
			x = torch.cat((x, x_skip), dim=3)
			# shape [batch_size, num_edges, num_timesteps, n_hid*3]
			x = self.mlp4(x)
			# [batch_size, num_edges, num_timesteps, n_hid]

		old_shape = x.shape
		x = x.contiguous().view(-1, old_shape[2], old_shape[3])
		output, h = self.rnn_encoder(x)
		# outupt [batch_size*num_edges, num_timesteps, rnn_hidden_size]
		# h [batch_size*num_edges, rnn_hidden_size]
		# c [batch_size*num_edges, rnn_hidden_size]
		h  = h.view(inputs.size(0), rel_rec.size(2), -1)
		# [batch_size, num_edges, rnn_hidden_size]
		return self.fc_out(h)
		# [batch-size, num_edges, n_out]

		















