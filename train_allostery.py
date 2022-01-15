# Author Mahdi Ghorbani 
# Email: (ghorbani.mahdi73@gmail.com)
# Initial code was taken from the Original NRI implementation by Thomas Kipf https://github.com/ethanfetaya/NRI
# licence MIT

from __future__ import print_function, division
import time
import argparse
import pickle
import os
import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
from encoders import *
from decoders import *
from args import buildParser

total_steps = 125 # for plotting purposes during testing

#------------- import the args -----------
args = buildParser().parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
	print('Using cuda')
args.factor = not args.no_factor

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
	print('testing with dynamically recomputed graph')

# ------- save model and metadata always saves in a new subfolder -----
if args.save_folder:
	exp_counter = 0
	now = datetime.datetime.now()
	timestamp = now.isoformat()
	save_folder = args.save_folder  + '/'
	if not os.path.isdir(save_folder):
		os.mkdir(save_folder)
	meta_file = os.path.join(save_folder, 'metadata.pkl')
	encoder_file = os.path.join(save_folder, 'encoder.pt')
	decoder_file = os.path.join(save_folder, 'decoder.pt')

	log_file = os.path.join(save_folder, 'log.txt')
	log = open(log_file, 'w')

	pickle.dump({'args':args}, open(meta_file, 'wb'))

	with open(args.save_folder+'/args.txt','w') as f:
		f.write(str(args))
else:
	print('Warning: No save folder provided' + 'testing will throw an error')


# load data
if args.use_PhysChem:
	train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data_physico(args.seq_file,
											 args.batch_size, shuffle=True, data_folder='data')
else:
	train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(args.batch_size, 
											 shuffle=True, data_folder='data')

# ------------------- load static/dynamics adjacency -----------------------
if args.dynamic_adj:
	#edges_train = np.load(os.path.join(args.data_folder, 'edges_train.npy'))
	#print('loaded training edges ......')
	#edges_valid = np.load(os.path.join(args.data_folder, 'edges_valid.npy'))
	#print('loaded validation edges .....')
	#edges_test = np.load(os.path.join(args.data_folder, 'edges_test.npy'))
	#print('loaded testing edges .....')

	# [batch_size, num_timesteps, num_atoms, num_atoms]
	#print('Convering to sparse adjacency .....')
	#train_rel_recv, train_rel_send = adj2sparse(edges_train)
	#print('done with training edges .....')
	#valid_rel_recv, valid_rel_send = adj2sparse(edges_valid)
	#print('done with validation edges .....')
	#test_rel_recv, test_rel_send = adj2sparse(edges_test)
	#print('done with testing edges .....')

	#np.save('train_rel_recv.npy', train_rel_recv)
	#np.save('train_rel_send.npy', train_rel_send)
	#np.save('valid_rel_recv.npy', valid_rel_recv)
	#np.save('valid_rel_send.npy', valid_rel_send)
	#np.save('test_rel_recv.npy',  test_rel_recv)
	#np.save('test_rel_send.npy',  test_rel_send)

	train_rel_recv = np.load('train_rel_recv.npy')
	train_rel_send = np.load('train_rel_send.npy')
	valid_rel_recv = np.load('valid_rel_recv.npy')
	valid_rel_send = np.load('valid_rel_send.npy')
	test_rel_recv  = np.load('test_rel_recv.npy')
	test_rel_send  = np.load('test_rel_send.npy')


	train_rel_recv, train_rel_send = torch.FloatTensor(train_rel_recv), torch.FloatTensor(train_rel_send)
	valid_rel_recv, valid_rel_send = torch.FloatTensor(valid_rel_recv), torch.FloatTensor(valid_rel_send)
	test_rel_recv, test_rel_send =   torch.FloatTensor(test_rel_recv), torch.FloatTensor(test_rel_send)

	#if args.cuda:
	#	train_rel_recv, train_rel_send = train_rel_recv.cuda(), train_rel_send.cuda()
	#	valid_rel_recv, valid_rel_send = valid_rel_recv.cuda(), valid_rel_send.cuda()
	#	test_rel_recv, test_rel_send = test_rel_recv.cuda(), test_rel_send.cuda()

	train_rel_recv, train_rel_send = Variable(train_rel_recv), Variable(train_rel_send)
	valid_rel_recv, valid_rel_send = Variable(valid_rel_recv), Variable(valid_rel_send)
	test_rel_recv, test_rel_send = Variable(test_rel_recv), Variable(test_rel_send)
	print('Done with loading the dynamic adjacency matrices ....')
else:
	print('using static adjacency ....')
	all_edges = np.load(args.data_folder+'/adj_static.npy').reshape(args.num_atoms, args.num_atoms)
	rel_rec = np.array(encode_onehot(np.where(all_edges)[0]), dtype=np.float32)
	rel_send = np.array(encode_onehot(np.where(all_edges)[1]), dtype=np.float32)
	rel_rec_s = torch.FloatTensor(rel_rec)	
	rel_send_s = torch.FloatTensor(rel_send)

	if args.cuda:
		rel_rec_s.cuda()
		rel_send_s.cuda()
	rel_rec_s, rel_send_s = Variable(rel_rec_s), Variable(rel_send_s)

if args.dynamic_adj:
	args.dynamic_adj = True
else:
	args.dynamic_adj = False

# dynamic adjacency can only be used with a RNN Encoder
if args.encoder == 'mlp':
	# MLP Encoder 
	# we have same parameters for all timesteps
	print('Using MLP Encoder')
	encoder = MLPEncoder(n_in=args.timesteps*args.dims, n_hid=args.encoder_hidden, n_out=args.edge_types,
						do_prob=args.encoder_dropout, factor=args.factor, dynamic_adj=args.dynamic_adj)

elif args.encoder == 'cnn':
	# CNN Encoder
	print('Using CNN Encoder')
	encoder = CNNEncoder(n_in=args.dims, n_hid=args.encoder_hidden, n_out=args.edge_types, 
						do_prob=args.encoder_dropout, factor=args.factor, dynamic_adj=args.dynamic_adj)

elif args.encoder == 'RNNEncoder':
	# RNN Encoder with dynamics adjacency
	print('Using RNN Encoder')
	encoder = RNNEncoder(n_in=args.dims, n_hid=args.encoder_hidden, n_out=args.edge_types, 
		                rnn_hidden_size=args.encoder_hidden, rnn_type='gru', num_layers=1, do_prob=args.encoder_dropout,
		                factor=args.factor, dynamic_adj=args.dynamic_adj)

# dynamic adjacency only for a RNN Decoder
if args.decoder == 'mlp':
	# MLP Decoder
	print('Using MLP Decoder')
	decoder = MLPDecoder(n_in_node=args.dims,
		                 edge_types=args.edge_types,
		                 msg_hid=args.decoder_hidden,
		                 msg_out=args.decoder_hidden,
		                 n_hid=args.decoder_hidden,
		                 do_prob=args.decoder_dropout,
		                 skip_first=args.skip_first,
		                 dynamic_adj=False)

if args.decoder == 'RNNDecoder':
	# RNN Decoder
	print('Using RNNDecoder')
	decoder = RNNDecoder(n_in_node=args.dims,
		                 edge_types=args.edge_types,
		                 n_hid=args.decoder_hidden,
		                 do_prob=args.decoder_dropout,
		                 skip_first=args.skip_first,
		                 dynamic_adj=args.dynamic_adj)

if args.decoder == 'RNNSpatioTemporalDecoder':
	# Spatio Temporal decoder
	print('using RNNSpatioTemporalDecoder')
	decoder = RNNSpatioTemporalDecoder(n_in_node=args.dims,
		 							   edge_types=args.edge_types,
		 							   n_hid=args.decoder_hidden,
		 							   do_prob=args.decoder_dropout,
		 							   skip_first=args.skip_first,
		 							   dynamic_adj=args.dynamic_adj)

if args.decoder == 'AttnSpatioTemporalDecoder':
	print('Using Attention-SpatioTemporal decoder')
	decoder = AttnSpatioTemporalDecoder(n_in_node=args.dims,
										edge_types=args.edge_types,
										n_hid=args.decoder_hidden,
										attn_hid=args.decoder_hidden,
										do_prob=args.decoder_dropout,
										skip_first=args.skip_first,
										dynamic_adj=args.dynamic_adj)


if args.load_folder:
	encoder_file = os.path.join(args.load_folder, 'encoder.pt')
	encoder.load_state_dict(torch.load(encoder_file))
	decoder_file = os.path.join(args.load_folder, 'decoder.pt')
	decoder.load_state_dict(torch.load(decoder_file))

	args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr=args.lr)
encoder_params = sum(p.numel() for p in encoder.parameters())
decoder_params = sum(p.numel() for p in decoder.parameters())
print('number of parameters:', encoder_params+decoder_params)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

if args.prior: # Whether using sparsity prior for different edge types 
	# need to experiment on this later
	prior = np.array([0.7, 0.1, 0.1, 0.1]) # hard coded for now
	print('Using prior')
	print(prior)
	log_prior = torch.FloatTensor(np.log(prior))
	log_prior = torch.unsqueeze(log_prior, 0)
	log_prior = torch.unsqueeze(log_prior, 0)
	log_prior = Variable(log_prior)

	if args.cuda:
		log_prior = log_prior.cuda()

if args.cuda:
	encoder.cuda()
	decoder.cuda()

def train(epoch, best_val_loss):
	t = time.time()
	nll_train = []
	kl_train = []
	mse_train = []
	edges_train = []
	probs_train = []

	encoder.train()
	decoder.train()
	scheduler.step()
	# due to pytorch convention we load the relations from train_loader but don't use these
	for batch_idx, (data, relations) in enumerate(train_loader):
		if args.cuda:
			data = data.cuda()
			relations = relations.cuda()

		data = Variable(data)
		# data [batch, num_nodes, num_timesteps, num_features]
		optimizer.zero_grad()

		# if dynamic_adj then relations [batch, num_timesteps, num_nodes, num_nodes]
		# relations [batch, num_timesteps, num_nodes, num_nodes]
		if  args.dynamic_adj:
			rel_rec = train_rel_recv[batch_idx].unsqueeze(0)
			rel_send = train_rel_send[batch_idx].unsqueeze(0)
			logits = encoder(data, rel_rec.cuda(), rel_send.cuda())	
			# logits [batch_size, num_edges, num_edge_types]
		else:
			# rel_rec/rel_send [num-edges, num-nodes]
			rel_rec = rel_rec_s
			rel_send = rel_send_s
			logits = encoder(data, rel_rec.cuda(), rel_send.cuda())
			# logits [batch_size, num_edges, num_edge_types]

		if args.sym_hard:
			logits = sym_hard(logits, args.num_atoms)
			# logits [batch_size, num_edges, num_edge_types]
		edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
		# edges [batch_size, num_edges, num_edge_types]

		prob = my_softmax(logits, -1)
		# prob [batch_size, num_edges, num_edge_types]

		if (args.decoder == 'RNNDecoder') or (args.decoder == 'RNNSpatioTemporalDecoder') or (args.decoder =='AttnSpatioTemporalDecoder'):
			output = decoder(data, edges, rel_rec.cuda(), rel_send.cuda(), args.timesteps,
				             burn_in=True, burn_in_steps=args.timesteps-args.prediction_steps)
		else:
			output = data[:,:,:,:args.dims]
			output = decoder(data, edges, rel_rec.cuda(), rel_send.cuda(), args.prediction_steps)

		# output [batch_size, num_nodes, num_timesteps-1, num_features]
		output = output[:, :, :, :args.dims]
		target = data[:, :, 1:, :args.dims]
		loss_nll = nll_gaussian(output, target, args.var)

		if args.prior:
			loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
		else:
			loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

		loss = loss_nll + loss_kl

		loss.backward()
		optimizer.step()

		mse_train.append(F.mse_loss(output, target).item())
		nll_train.append(loss_nll.item())
		kl_train.append(loss_kl.item())
		_, edges_t = edges.max(-1)
		edges_train.append(edges_t.data.cpu().numpy())
		probs_train.append(prob.data.cpu().numpy())

	# ----------------------- validation ---------------------------
	scheduler.step()
	nll_val = []
	kl_val = []
	mse_val = []

	encoder.eval()
	decoder.eval()
	for batch_idx, (data, relations) in enumerate(valid_loader):
		if args.cuda:
			data = data.cuda()

		with torch.no_grad():
			if  args.dynamic_adj:
				rel_rec = valid_rel_recv[batch_idx].unsqueeze(0)
				rel_send = valid_rel_send[batch_idx].unsqueeze(0)
				logits = encoder(data, rel_rec.cuda(), rel_send.cuda())	
			else:
				rel_rec = rel_rec_s
				rel_send = rel_send_s
				logits = encoder(data, rel_rec.cuda(), rel_send.cuda())

			edges = gumbel_softmax(logits, tau=args.temp, hard=True)
			prob = my_softmax(logits, -1)

			# validation output uses teacher forcing
			data = data[:,:,:,:args.dims]
			output = decoder(data, edges, rel_rec.cuda(), rel_send.cuda(), 1)
			target = data[:,:,1:,:args.dims]

			loss_nll = nll_gaussian(output, target, args.var)
			if args.prior:
				loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
			else:
				loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

		mse_val.append(F.mse_loss(output, target).item())
		nll_val.append(loss_nll.item())
		kl_val.append(loss_kl.item())

	# ----------------- printing results ------------------------------

	print('Epoch: {0:4d}'.format(epoch),
		 'nll_train: {:.10f}'.format(np.mean(np.array(nll_train))),
		 'kl_train: {:.10f}'.format(np.mean(np.array(kl_train))),
		 'mse_train: {:.10f}'.format(np.mean(np.array(mse_train))),
		 'nll_val: {:.10f}'.format(np.mean(np.array(nll_val))),
		 'kl_val: {:.10f}'.format(np.mean(np.array(kl_val))),
		 'mse_val: {:.10f}'.format(np.mean(np.array(mse_val))),
		 'time {:.4f}s'.format(time.time()-t))
	edges_train = np.concatenate(edges_train)
	probs_train = np.concatenate(probs_train)

	#with open('edges_train.npy','wb') as f:
	#	np.save(f, edges_train)

	#with open('probs_train.npy','wb') as f:
	#	np.save(f, probs_train)

	if args.save_folder and np.mean(np.array(nll_val)) < best_val_loss:
		torch.save(encoder.state_dict(), encoder_file)
		torch.save(decoder.state_dict(), decoder_file)
		print('Best model so far, saving...')
		print('Epoch: {0:4d}'.format(epoch),
		 'nll_train: {:.10f}'.format(np.mean(np.array(nll_train))),
		 'kl_train: {:.10f}'.format(np.mean(np.array(kl_train))),
		 'mse_train: {:.10f}'.format(np.mean(np.array(mse_train))),
		 'nll_val: {:.10f}'.format(np.mean(np.array(nll_val))),
		 'kl_val: {:.10f}'.format(np.mean(np.array(kl_val))),
		 'mse_val: {:.10f}'.format(np.mean(np.array(mse_val))),
		 'time {:.4f}s'.format(time.time()-t), file=log)
	log.flush()

	return encoder, decoder, edges_train, probs_train, np.mean(np.array(nll_val))

# ----------------------- testing ---------------------------------
def test():
	nll_test = []
	kl_test = []
	mse_test = []
	edges_test = []
	probs_test = []
	tot_mse = 0
	counter = 0

	encoder.eval()
	decoder.eval()
	encoder.load_state_dict(torch.load(encoder_file))
	decoder.load_state_dict(torch.load(decoder_file))

	for batch_idx, (data,relations) in enumerate(test_loader):
		if args.cuda:
			data = data.cuda()

		with torch.no_grad():
			if args.dynamic_adj:
				rel_rec = test_rel_recv[batch_idx].unsqueeze(0)
				rel_send = test_rel_send[batch_idx].unsqueeze(0)
				logits = encoder(data, rel_rec.cuda(), rel_send.cuda())		
			else:
				rel_rec = rel_rec_s
				rel_send = rel_send_s
				logits = encoder(data, rel_rec.cuda(), rel_send.cuda())

			#assert (data.size(2) - args.timesteps) >= args.timesteps
			data_encoder = data[:,:,:args.timesteps, :].contiguous() # args.timestesps were for training and the rest for testing
			data_decoder = data[:,:,-args.timesteps:,:].contiguous()

			logits = encoder(data_encoder, rel_rec.cuda(), rel_send.cuda())
			edges = gumbel_softmax(logits, tau=args.temp, hard=True)

			prob = my_softmax(logits, -1)
			output = decoder(data_decoder, edges, rel_rec.cuda(), rel_send.cuda(), 1)

			target = data_decoder[:,:,1:,:]
			output = output[:,:,:, :args.dims]
			target = target[:,:,:, :args.dims]

			loss_nll = nll_gaussian(output, target, args.var)
			loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

			mse_test.append(F.mse_loss(output, target).item())
			nll_test.append(loss_nll.item())
			kl_test.append(loss_kl.item())
			_, edges_t = edges.max(-1)
			edges_test.append(edges_t.data.cpu().numpy())
			probs_test.append(prob.data.cpu().numpy())


			# for plotting purposes
			if (args.decoder == 'RNNDecoder') or (args.decoder == 'RNNSpatioTemporalDecoder') or (args.decoder=='AttnSpatioTemporalDecoder'):
				if args.dynamic_graph:
					output = decoder(data, edges, rel_rec.cuda(), rel_send.cuda(), total_steps,
						             burn_in=True, burn_in_steps=args.timesteps,
						             dynamic_graph=True, encoder=encoder,
						             temp=args.temp)

				else:
					data = data[:,:,:, :args.dims]
					output = decoder(data, edges, rel_rec.cuda(), rel_send.cuda(), total_steps,
						             burn_in=True, burn_in_steps=args.timesteps)

				output = output[:, :, args.timesteps:, :args.dims]
				target = data[:, :, -args.timesteps:, :args.dims]

			else:

				data_plot = data[:, :, args.timesteps:args.timesteps+21, :].contiguous()
				output = decoder(data_plot, edges, rel_rec.cuda(), rel_send.cuda(), 20) 
				output = output[:, :, :, :args.dims]
				target = data_plot[:, :, 1:, :args.dims]

			print('target ', target.shape)
			print('outupt ', output.shape)
			mse = ((target - output)**2).mean(dim=0).mean(dim=0).mean(dim=-1)
			tot_mse += mse.data.cpu().numpy()
			counter += 1
	mean_mse = tot_mse / counter
	#mse_str = '['
	#print(mean_mse.shape)
	#for mse_step in mean_mse[:-1]:
	#	mse_str += " {:.12f} ,".format(mse_step)
	#mse_str += " {:.12f} ".format(mean_mse)
	#mse_str += ']'

	print('--------------------------------')
	print('-------------Testing------------')
	print('--------------------------------')

	print('nll_test: {:.10f}'.format(np.mean(nll_test)),
		  'kl_test: {:.10f}'.format(np.mean(kl_test)),
		  'mse_test: {:.10f}'.format(np.mean(mse_test)))

	print('MSE: {}'.format(mean_mse), file=log)
	edges_test = np.concatenate(edges_test)
	probs_test = np.concatenate(probs_test)

	with open(args.save_folder+'edges_test.npy','wb') as f:
		np.save(f, edges_test)

	with open(args.saves_folder+'probs_test.npy','wb') as f:
		np.save(f, probs_test)

	if args.save_folder:
		print('--------------------------------')
		print('-------------Testing------------')
		print('--------------------------------')

		print('nll_test: {:.10f}'.format(np.mean(nll_test)),
			  'kl_test: {:.10f}'.format(np.mean(kl_test)),
			  'mse_test: {:.10f}'.format(np.mean(mse_test)),
			  file=log)
		print('MSE: {}'.format(mean_mse), file=log)
		log.flush()

	return edges_test, probs_test

# train the model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
	encoder, decoder, edges_train, probs_train, val_loss = train(
		 epoch, best_val_loss)
	if val_loss < best_val_loss:
		best_val_loss = val_loss
		best_epoch = epoch
np.save(str(args.save_folder)+'/out_edges_train.npy', edges_train)
np.save(str(args.save_folder)+'/out_probs_train.npy', probs_train)
print('optimization finished')
print('Best epoch: {:04d}'.format(best_epoch))
if args.save_folder:
	print('Best epoch: {:04d}'.format(best_epoch), file=log)
	log.flush()

# test
edges_test, probs_test = test()
if log is not None:
	print(save_folder)
	log.close()			










