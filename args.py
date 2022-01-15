# Author Mahdi Ghorbani 
# Email: (ghorbani.mahdi73@gmail.com)
# Initial code was taken from the Original NRI implementation by Thomas Kipf https://github.com/ethanfetaya/NRI
# licence MIT

from __future__ import print_function, division

import argparse
import pickle 
import os

def buildParser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-cuda', action='store_true', default=False, )
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
	parser.add_argument('--batch-size', type=int, default=1, help='number of samples per batch')
	parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
	parser.add_argument('--encoder-hidden', type=int, default=64, help='number of hidden units in encoder')
	parser.add_argument('--decoder-hidden', type=int, default=64, help='number of hidden units in decoder')
	parser.add_argument('--temp', type=float, default=0.5, help='gumbel-softmax temperature')
	parser.add_argument('--num_atoms', type=int, default=305, help='number of nodes in the graph')
	parser.add_argument('--encoder', type=str, default='mlp', help='type of encoder model')
	parser.add_argument('--decoder', type=str, default='mlp', help='type of decoder model')
	parser.add_argument('--no-factor', action='store_true', default=False, help='Disble factor graph model')
	parser.add_argument('--encoder-dropout', type=float, default=0.0, help='dropout rate for encoder')
	parser.add_argument('--decoder-dropout', type=float, default=0.0, help='dropout rate for decoder')
	parser.add_argument('--save-folder', type=str, default='logs', help='Where to save the trained model')
	parser.add_argument('--data-folder', type=str, default='data', help='data folder to load data')
	parser.add_argument('--load-folder', type=str, default='', help='Where to load the trained model')
	parser.add_argument('--edge-types', type=int, default=2, help='number of edge types to infer')
	parser.add_argument('--dims', type=int, default=6, help='number of dimensions  (pos+vel)')
	parser.add_argument('--dims-phys', type=int, default=6, help='Number of dimensions of physicoChemical properties')
	parser.add_argument('--timesteps', type=int, default=125, help='number of timesteps per sample')
	parser.add_argument('--prediction-steps', type=int, default=10, metavar='N', help='number of steps to predict before reusing teacher forcing')
	parser.add_argument('--lr-decay', type=int, default=100, help='After how many epochs to decay lr by factor gamma')
	parser.add_argument('--gamma', type=float, default=0.5, help='lr decay factor')
	parser.add_argument('--skip-first', action='store_true', default=False, help='skip first edge type in the decoder i.e no interaction')
	parser.add_argument('--var', type=float, default=1e-3, help='output variance')
	parser.add_argument('--hard', action='store_true', default=False, help='uses discrete samples in training forward pass')
	parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior')
	parser.add_argument('--dynamic-graph', action='store_true', default=False, help='Whether test with dynamically recomputed graph')
	parser.add_argument('--sym-hard', action='store_true', default=False, help='Whether to use hard symmetry constraint on edges')
	parser.add_argument('--seq-file', type=str, default='seq.txt', help='Sequence file for reading the protein sequence')
	parser.add_argument('--use-PhysChem', action='store_true', default=False, help='Whether to use physicoChemical properties of amino acids in the encoder')
	parser.add_argument('--dynamic-adj', action='store_true', default=False, help='Whether to use dynamic adjacency matrix')
	return parser
