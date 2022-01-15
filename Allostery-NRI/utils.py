# Author Mahdi Ghorbani 
# Email: (ghorbani.mahdi73@gmail.com)
# Initial code was taken from the Original NRI implementation by Thomas Kipf https://github.com/ethanfetaya/NRI
# licence MIT

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from os import path
from tqdm import tqdm
from itertools import combinations


######################### Amino Acid physico-chemical properties ################
def encode_physico(filename):
	with open(filename, 'r') as f:
		sequence = open(filename)
		seq = sequence.readlines()

	protein_seq = list(seq[0])

	aminos = 'ACDEFGHIKLMNPQRSTVWY'
	aminos = list(aminos)

	H_Phob = [0.281, 0.458, 0, 0.027, 1, 0.198, 0.207, 0.792, 0.198, 0.783, 0.721, 0.12, 0.253, 0.123, 0.222, 0.235, 0.318, 0.687, 0.56, 0.922]
	H_Phyl = [0.453, 0.375, 1, 1, 0.14, 0.531, 0.453, 0.25, 1, 0.25, 0.328, 0.562, 0.531, 0.562, 1, 0.578, 0.468, 0.296, 0, 0.171]
	Polarity = [0.395, 0.074, 1, 0.913, 0.037, 0.506, 0.679, 0.037, 0.79, 0, 0.098, 0.827, 0.382, 0.691, 0.691, 0.53, 0.456, 0.123, 0.061, 0.16]
	Polarz = [0.112, 0.312, 0.256, 0.369, 0.709, 0, 0.562, 0.454, 0.535, 0.454, 0.54, 0.327, 0.32, 0.44, 0.711, 0.151, 0.264, 0.342, 1, 0.728]
	SFE = [0.589, 0.527, 0.191, 0.285, 0.936, 0.446, 0.582, 0.851, 0.325, 0.851,  0.957, 0.319, 0.702, 0.4, 0, 0.448, 0.557, 0.765, 1, 0.787]
	RAS = [0.222, 0.333, 0.416, 0.638, 0.75, 0, 0.666, 0.555, 0.694, 0.527, 0.611, 0.472, 0.388, 0.583, 0.833, 0.222, 0.361, 0.444, 1, 0.861]

	# https://doi.org/10.1016/j.jbi.2015.06.018

	all_ph = H_Phob + H_Phyl + Polarity + Polarz + SFE + RAS
	all_ph = np.array(all_ph)*2 - 1 # rescale to be between [-1, 1]
	all_ph = all_ph.reshape(6, 20)

	residue_dict = {}
	for i in range(len(aminos)):
	    residue_dict[aminos[i]]= all_ph[:,i]

	#print(residue_dict)
	print(seq)


	seq_phys = []
	for i in (protein_seq):
		seq_phys.append(residue_dict[i])
	seq_phys = np.array(seq_phys)
 	# [number of residues, number of physico-chemical features]
	return seq_phys

##################################################################################



def sym_hard(x, size):
    """
    Given the edge features x, set x(e_ji) = x(e_ij) to impose hard symmetric constraints.
    """
    i, j = np.array(list(combinations(range(size), 2))).T
    idx_s = j * (size - 1) + i * (i < j) + (i - 1) * (i > j)
    idx_t = i * (size - 1) + j * (j < i) + (j - 1) * (j > i)
    x[idx_t] = x[idx_s]
    return x

def my_softmax(input, axis=1):
	trans_input = input.transpose(axis, 0).contiguous()
	soft_max_1d = F.softmax(trans_input, dim=0)
	return soft_max_1d.transpose(axis, 0)

def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
	y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
	if hard:
		y_hard = (y_soft>0.5).float()
		y = Variable(y_hard.data - y_soft.data) + y_soft
	else:
		y = y_soft
	return y

def binary_concrete_sample(logits, tau=1, eps=1e-10):
	logistic_noise = sample_logistic(logits.size(), eps=eps)
	if logits.is_cuda:
		logistic_noise = logistic_noise.cuda()
	y = logits + Variable(logistic_noise)
	return F.sigmoid(y/tau)

def sample_logistic(shape, eps=1e-10):
	uniform = torch.rand(shape).float()
	return torch.log(uniform+eps) - torch.log(1-uniform+eps)

def sample_gumbel(shape, eps=1e-10):
	U = torch.rand(shape).float()
	return -torch.log(eps-torch.log(U + eps))

def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
	gumbel_noise = sample_gumbel(logits.size(), eps=eps)
	if logits.is_cuda:
		gumbel_noise = gumbel_noise.cuda()
	y = logits + Variable(gumbel_noise)
	return my_softmax(y/tau, axis=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
	"""
	args:
		logits: [batch-size, n_class]
		tau: non-negative scalar temp
		hard: if true take argmax, but differentiate w.r.t soft sample y
	returns:
		[batch-size, n_class] sample from gumbel-softmax dist
		if hard=True then the returned sample will be one-hot, otherwise it will
		be a probability distribution that sums to 1 across classes
	"""
	y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
	if hard:
		shape = logits.size()
		_, k = y_soft.data.max(-1)
		y_hard = torch.zeros(*shape)
		if y_soft.is_cuda:
			y_hard = y_hard.cuda()
		y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1]+(1,)), 1.0)

		y = Variable(y_hard - y_soft.data) + y_soft
	else:
		y = y_soft
	return y

def load_data(batch_size=1, shuffle=True, data_folder='', dynamic_adj=False):
    # the edges numpy arrays below are [ num_sims, N, N ]
    loc_train = np.load(path.join(data_folder,'loc_train.npy'))
    vel_train = np.load(path.join(data_folder,'vel_train.npy'))

    # loc_train/vel_train [num_sims, num_timesteps, num_atoms, feature_dim]

    loc_valid = np.load(path.join(data_folder,'loc_valid.npy'))
    vel_valid = np.load(path.join(data_folder,'vel_valid.npy'))

    loc_test = np.load(path.join(data_folder,'loc_test.npy'))
    vel_test = np.load(path.join(data_folder,'vel_test.npy'))


    num_atoms = loc_train.shape[1]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1	

  # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 2, 1, 3])
    vel_train = np.transpose(vel_train, [0, 2, 1, 3])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)

    loc_valid = np.transpose(loc_valid, [0, 2, 1, 3])
    vel_valid = np.transpose(vel_valid, [0, 2, 1, 3])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)

    loc_test = np.transpose(loc_test, [0, 2, 1, 3])
    vel_test = np.transpose(vel_test, [0, 2, 1, 3])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)

    #edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    #edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    #edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    #edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    #edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    #edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)    

    # [num_sims, num_atoms**2]
    print(loc_train.shape)
    # due to pytorch convention we need to parse something for the labels (edges)
    edges_train = np.zeros((loc_train.shape[0], num_atoms, num_atoms))
    edges_valid = np.zeros((loc_valid.shape[0], num_atoms, num_atoms))
    edges_test = np.zeros((loc_test.shape[0], num_atoms, num_atoms))
    

    feat_train = torch.FloatTensor(feat_train)
    feat_valid = torch.FloatTensor(feat_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_train = torch.LongTensor(edges_train)
    edges_valid = torch.LongTensor(edges_valid)
    edges_test = torch.LongTensor(edges_test)

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min

def load_data_physico(seq_file, batch_size=1, shuffle=True, data_folder='', dynamic_adj=False):
    # the edges numpy arrays below are [ num_sims, N, N ]
    loc_train = np.load(path.join(data_folder,'loc_train.npy'))
    vel_train = np.load(path.join(data_folder,'vel_train.npy'))

    # loc_train/vel_train [num_sims, num_timesteps, feature_dim, num_atoms]

    loc_valid = np.load(path.join(data_folder,'loc_valid.npy'))
    vel_valid = np.load(path.join(data_folder,'vel_valid.npy'))
    
    loc_test = np.load(path.join(data_folder,'loc_test.npy'))
    vel_test = np.load(path.join(data_folder,'vel_test.npy'))

    num_atoms = loc_train.shape[3]

    phys_prop = encode_physico(seq_file)
    print(phys_prop.shape)

    # reshape to [num_sims, num_atoms, num_timesteps, num_dims]
    phys_prop_train = np.tile(phys_prop, (loc_train.shape[0], loc_train.shape[1], 1, 1)).swapaxes(1,2)
    phys_prop_valid = np.tile(phys_prop, (loc_valid.shape[0], loc_valid.shape[1], 1, 1)).swapaxes(1,2)
    phys_prop_test = np.tile(phys_prop, (loc_test.shape[0], loc_test.shape[1], 1, 1)).swapaxes(1,2)

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1	

  # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])

    feat_train = np.concatenate([loc_train, vel_train, phys_prop_train], axis=3)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid, phys_prop_valid], axis=3)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test, phys_prop_test], axis=3) 

    # [num_sims, num_atoms**2]
    
    feat_train = torch.FloatTensor(feat_train)
    feat_valid = torch.FloatTensor(feat_valid)
    feat_test = torch.FloatTensor(feat_test)

    train_data = TensorDataset(feat_train)
    valid_data = TensorDataset(feat_valid)
    test_data = TensorDataset(feat_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i,:] for i, c in 
	                enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
	return labels_onehot


def get_offdiag_indices(num_nodes):
	""" linear off-diagonal indices"""
	ones = torch.ones(num_nodes, num_nodes)
	eye = torch.eye(num_nodes, num_nodes)
	offdiag_indices = (one-eye).nonzero().t()
	offdiag_indices = offdiag_indices[0]*num_nodes + offdiag_indices[1]
	return offdiag_indices

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
	kl_div = preds*(torch.log(preds+eps)-log_prior)
	return kl_div.sum() / (num_atoms*preds.size(0))

def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
	                       eps=1e-16):
	kl_div = preds*torch.log(preds+eps)
	if add_const:
		const = np.log(num_edge_types)
		kl_div += const
	return kl_div.sum() / (num_atoms*preds.size(0))

def nll_gaussian(preds, target, variance, add_const=False):
	neg_log_p = ((preds-target)**2/(2*variance))
	if add_const:
		const = 0.5*np.log(2*np.pi*variance)
		neg_log_p += const
	return neg_log_p.sum() / (target.size(0)*target.size(1))

def adj2sparse(edges):
    # convert to sparse adjacency matrix
    all_rel_rec = []
    all_rel_send = []
    for i in tqdm(range(edges.shape[0])):
        temp_rel_rec = []
        temp_rel_send = []
        for j in range(edges.shape[1]):
            rel_rec = np.array(encode_onehot(np.where(edges[i][j])[0]),dtype=np.float32)
            rel_send = np.array(encode_onehot(np.where(edges[i][j])[1]),dtype=np.float32)
            temp_rel_rec.append(rel_rec)
            temp_rel_send.append(rel_send)
        all_rel_rec.append(np.array(temp_rel_rec))
        all_rel_send.append(np.array(temp_rel_send))

    return np.array(all_rel_rec), np.array(all_rel_send)


