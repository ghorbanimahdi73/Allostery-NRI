import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax

_EPS = 1e-10

class MLP(nn.Module):
	def __init__(self, n_in, n_hid, n_out, do_prob=0., no_bn=False):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(n_in, n_hid)
		self.fc2 = nn.Linear(n_hid, n_out)
		self.no_bn = no_bn
		if self.no_bn:
			self.bn = nn.BatchNorm1d(n_out)
		self.dropout_prob = do_prob

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal(m.weight.data)
				m.bias.data.fill_(0.1)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def batch_norm(self, inputs):
		x = inputs.view(inputs.size(0)*inputs.size(1), -1)
		x = self.bn(x)
		return x.view(inputs.size(0), inputs.size(1), -1)

	def forward(self, inputs):
		# inputs [batch_size, num_atoms, num_featuers]
		x = F.elu(self.fc1(inputs))
		x = F.dropout(x, self.dropout_prob, training=self.training)
		x = F.elu(self.fc2(x))
		if self.no_bn:
			return self.batch_norm(x)
		else:
			return x


class CNN(nn.Module):
	def __init__(self, n_in, n_hid, n_out, do_prob=0.):
		super(CNN, self).__init__()
		self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
			                     dilation=1, return_indices=False, ceil_mode=False)
		self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
		self.bn1 = nn.BatchNorm1d(n_hid)
		self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
		self.bn2 = nn.BatchNorm1d(n_hid)
		self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
		self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
		self.dropout_prob = do_prob

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d):
				n = m.kernel_size[0] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2./n))
				m.bias.data.fill_(0.1)
			elif isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, inputs):
		# [batch_size*num_edges, n_hid, num_timesteps]
		x = F.relu(self.conv1(inputs))
		x = self.bn1(x)
		x = F.dropout(x, self.dropout_prob, training=self.training)
		x = self.pool(x)
		x = F.relu(self.conv2(x))
		x = self.bn2(x)
		pred = self.conv_predict(x)
		attention = my_softmax(self.conv_attention(x), axis=2) # attention over timesteps
		edge_prob = (pred*attention).mean(dim=2)
		return edge_prob 
		# [batch_size*num_edges, n_hid, num_timesteps]


class myGRU(nn.Module):
	def __init__(self, n_in_node, n_hid):
		super(myGRU, self).__init__()
		self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
		self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
		self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

		self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
		self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
		self.input_n = nn.Linear(n_in_node, n_hid, bias=True)


	def forward(self, inputs, hidden, state=None):

		r = F.sigmoid(self.input_r(inputs)+self.hidden_r(hidden))
		i = F.sigmoid(self.input_i(inputs)+self.hidden_i(hidden))
		n = F.tanh(self.input_n(inputs)+r*self.hidden_i(hidden))
		if state is None:
			state = hidden
		output = (1-i)*n + i*state

		return output

class LinAct(nn.Module):
	"""
	a linear layer with a non-linear activation function
	"""
	def __init__(self, n_in, n_out, do_prob=0., act=None):
		"""
		args: 
			n_in: input dimension
			n_out: output dimension
			do_prob: dropout rate
		"""
		super(LinAct, self).__init__()
		if act == None:
			act = nn.ReLU()
		self.model = nn.Sequential(
					nn.Linear(n_in, n_out),
					act,
					nn.Dropout(do_prob))
	def forward(self, x):
		return self.model(x)




