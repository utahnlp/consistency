import sys
from pipeline import *
import argparse
import h5py
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from holder import *
from optimizer import *
from data import *
from util import *
from multiclass_loss import *
from transition_loss import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/bert_nli/")
parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="snli.train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="snli.val.hdf5")
parser.add_argument('--extra_train_data', help="Path to extra training data hdf5 file (optional).", default="")
parser.add_argument('--extra_val_data', help="Path to extra validation data hdf5 file (optional).", default="")
parser.add_argument('--unlabeled_data', help="Path to unlabeled training data hdf5 file. (optional)", default="")
parser.add_argument('--unlabeled_res', help="Path to unlabeled training res hdf5 file. (optional)", default="")
parser.add_argument('--unlabeled_triple_mode', help="Whether to use unlabeled data in triple mode (optional)", type=int, default=0)
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "glove.hdf5")
parser.add_argument('--char_idx', help="The path to word2char index file", default = "char.idx.hdf5")
parser.add_argument('--dict', help="The path to word dictionary", default = "snli.word.dict")
parser.add_argument('--char_dict', help="The path to char dictionary", default = "char.dict.txt")
parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
# resource specs
parser.add_argument('--train_res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--val_res', help="Path to validation resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--use_word_vec', help="Whether to use word vec", type=int, default=0)
parser.add_argument('--use_char_emb', help="Whether to use character embedding", type=int, default=0)
parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=300)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--num_char', help="The number of char", type=int, default=68)
parser.add_argument('--token_l', help="The maximal token length", type=int, default=16)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.2)
parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=30)
parser.add_argument('--optim', help="The name of optimizer to use for training", default='adam_fp16')
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.001)
parser.add_argument('--clip', help="The norm2 threshold to clip, set it to negative to disable", type=float, default=1.0)
parser.add_argument('--adam_betas', help="The betas used in adam", default='0.9,0.999')
parser.add_argument('--weight_decay', help="The factor of weight decay", type=float, default=0.01)
parser.add_argument('--num_label', help="The number of label", type=int, default=3)
# bert specs
parser.add_argument('--bert_gpuid', help="The GPU index for bert, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--fp16', help="Whether to use fp16 format", type=int, default=1)
parser.add_argument('--fix_bert', help="Whether to fix bert update", type=int, default=1)
parser.add_argument('--bert_size', help="The input bert dim", type=int, default=768)
parser.add_argument('--warmup_perc', help="The percentages of total expectec updates to warmup", type=float, default=0.1)
parser.add_argument('--warmup_epoch', help="The number of epochs for warmup", type=int, default=2)
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of loss, multiclass/transition", default='multiclass')
#
parser.add_argument('--rnn_type', help="What type of rnn to use, default lstm", default='lstm')
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_normal')
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=500)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--acc_batch_size', help="The accumulative batch size, -1 to disable", type=int, default=-1)
# transition specs
parser.add_argument('--fwd_mode', help="The mode of forward pass, pair/flip/triple/flip_and_unlabeled", default='pair')
parser.add_argument('--constrs', help="The list of transition constraints, separated by comma", default='6')
parser.add_argument('--lambd', help="The factor for transition loss", type=float, default=1.0)
parser.add_argument('--unlabeled_percent', help="The ratio of unlabeled data to use", type=float, default=1.0)
parser.add_argument('--lambd_p', help="The lambd for unlabeled P data", type=float, default=1)
parser.add_argument('--lambd_t', help="The lambd for unlabeled T data", type=float, default=1)
parser.add_argument('--dynamic_lambd', help="Whether to use dynamic lambd to warm up", type=int, default=0)
parser.add_argument('--eta', help="The scale applied on in-domain annotated loss", type=float, default=1)


def get_loss(opt, shared):
	if opt.loss == 'multiclass':
		return MulticlassLoss(opt, shared)
	elif opt.loss == 'transition':
		return TransitionLoss(opt, shared)
	else:
		raise Exception("unrecognized loss {0}".format(opt.loss))

# the default fwd pass for multiclass loss
def forward_pass(m, source, target, char_source, char_target, bert1, bert2, batch_ex_idx, batch_l, source_l, target_l, res_map):
	wv_idx1 = Variable(source, requires_grad=False)
	wv_idx2 = Variable(target, requires_grad=False)
	cv_idx1 = Variable(char_source, requires_grad=False)
	cv_idx2 = Variable(char_target, requires_grad=False)
	
	m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map)
	output = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2, bert1, bert2)
	return output


def forward_pass_triple(m, source, target, third, char_source, char_target, char_third, bert1, bert2, bert3, batch_ex_idx, batch_l, source_l, target_l, third_l, res_map):
	wv_idx1 = Variable(source, requires_grad=False)
	wv_idx2 = Variable(target, requires_grad=False)
	wv_idx3 = Variable(third, requires_grad=False)
	cv_idx1 = Variable(char_source, requires_grad=False)
	cv_idx2 = Variable(char_target, requires_grad=False)
	cv_idx3 = Variable(char_third, requires_grad=False)

	# record elmo and swap in res_map for each forward pass
	m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map)
	alpha = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2, bert1, bert2)

	m.update_context(batch_ex_idx, batch_l, target_l, third_l, res_map)
	beta = m.forward(wv_idx2, wv_idx3, cv_idx2, cv_idx3, bert2, bert3)

	m.update_context(batch_ex_idx, batch_l, source_l, third_l, res_map)
	gamma = m.forward(wv_idx1, wv_idx3, cv_idx1, cv_idx3, bert1, bert3)

	return [alpha, beta, gamma]	# a list of log probabilities in batch mode, i.e. each of (batch_l, num_label)

def forward_pass_flip(m, source, target, char_source, char_target, bert1, bert2, batch_ex_idx, batch_l, source_l, target_l, res_map):
	wv_idx1 = Variable(source, requires_grad=False)
	wv_idx2 = Variable(target, requires_grad=False)
	cv_idx1 = Variable(char_source, requires_grad=False)
	cv_idx2 = Variable(char_target, requires_grad=False)

	# record elmo and swap in res_map for each forward pass
	m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map)
	alpha = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2, bert1, bert2)

	m.update_context(batch_ex_idx, batch_l, target_l, source_l, res_map)
	beta = m.forward(wv_idx2, wv_idx1, cv_idx2, cv_idx1, bert2, bert1)

	return [alpha, beta, None]	# a list of log probabilities in batch mode, i.e. each of (batch_l, num_label)

#
def forward_pass_pair(m, source, target, char_source, char_target, bert1, bert2, batch_ex_idx, batch_l, source_l, target_l, res_map):
	wv_idx1 = Variable(source, requires_grad=False)
	wv_idx2 = Variable(target, requires_grad=False)
	cv_idx1 = Variable(char_source, requires_grad=False)
	cv_idx2 = Variable(char_target, requires_grad=False)
	
	m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map)
	output = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2, bert1, bert2)
	return [output, None, None]


# train batch by batch, accumulate batches until the size reaches acc_batch_size
def train_epoch(opt, shared, m, optim, data, epoch_id, sub_idx, extra=None, extra_idx=None, unlabeled=None, unlabeled_idx=None):
	train_loss = 0.0
	num_ex = 0
	start_time = time.time()
	num_correct = 0
	min_grad_norm2 = 1000000000000.0
	max_grad_norm2 = 0.0

	loss = get_loss(opt, shared)

	# subsamples of data
	# if subsample indices provided, permutate from subsamples
	#	else permutate from all the data
	data_size = sub_idx.size()[0]
	batch_order = torch.randperm(data_size)
	batch_order = sub_idx[batch_order]
	all_data = []
	for i in range(data_size):
		all_data.append((data, batch_order[i]))

	if extra is not None:
		for i in range(extra_idx.size()[0]):
			all_data.append((extra, extra_idx[i]))
		batch_order = torch.randperm(len(all_data))
		data_size = len(all_data)
		print('combined and shuffled extra train data, num batches: {0}'.format(data_size))

	# if training with unlabeled data
	if opt.fwd_mode == 'flip_and_unlabeled' or opt.fwd_mode == 'triple' or opt.fwd_mode == 'pair_and_unlabeled' or opt.fwd_mode == 'flip_and_triple':
		assert(unlabeled is not None)

		for i in range(unlabeled_idx.size()[0]):
			all_data.append((unlabeled, unlabeled_idx[i]))
		batch_order = torch.randperm(len(all_data))
		data_size = len(all_data)
		print('combined and shuffled labeled and unlabeled data, num batches: {0}'.format(data_size))

	# something special here
	#	we need to duplicate the unlabeled data and mark one copy as pair mode and one as triple mode
	if opt.fwd_mode == 'flip_and_triple':
		all_data = [(d, idx, False) for d, idx in all_data]
		for i in range(unlabeled_idx.size()[0]):
			all_data.append((unlabeled, unlabeled_idx[i], True))	# data, idx, triple_mode
		batch_order = torch.randperm(len(all_data))
		data_size = len(all_data)
		print('duplicated and shuffled data for mode {0}, num batches: {1}'.format(opt.fwd_mode, data_size))


	acc_batch_size = 0
	shared.is_train = True
	m.train(True)
	loss.begin_pass()
	m.begin_pass()
	for i in range(data_size):
		shared.epoch = epoch_id
		shared.has_gold = True
		shared.in_domain = False
		shared.data_size = data_size

		if opt.fwd_mode == 'pair':
			# pair mode
			# use the original data for indexing
			(data_name, source, target, char_source, char_target, bert1, bert2,
				batch_ex_idx, batch_l, source_l, target_l, label, res_map) = data[batch_order[i]]

			y_gold = Variable(label, requires_grad=False)
			output = forward_pass(m, source, target, char_source, char_target, bert1, bert2,
				batch_ex_idx, batch_l, source_l, target_l, res_map)

		elif opt.fwd_mode == 'flip':
			# flip mode
			# use the original data for indexing
			(data_name, source, target, char_source, char_target, bert1, bert2,
				batch_ex_idx, batch_l, source_l, target_l, label, res_map) = data[cur_idx]

			y_gold = Variable(label, requires_grad=False)
			output = forward_pass_flip(m, source, target, char_source, char_target, bert1, bert2,
				batch_ex_idx, batch_l, source_l, target_l, res_map)


		elif opt.fwd_mode == 'flip_and_unlabeled':
			assert(unlabeled is not None)
			cur_data, cur_idx = all_data[batch_order[i]]
			shared.has_gold = cur_data == data or cur_data == extra 	# tag whether current batch has ground truth label
			shared.in_domain = cur_data == extra 	# tag whether current batch is snli

			(data_name, source, target, char_source, char_target, bert1, bert2,
				batch_ex_idx, batch_l, source_l, target_l, label, res_map) = cur_data[cur_idx]

			y_gold = Variable(label, requires_grad=False)
			output = forward_pass_flip(m, source, target, char_source, char_target, bert1, bert2,
				batch_ex_idx, batch_l, source_l, target_l, res_map)


		elif opt.fwd_mode == 'pair_and_unlabeled':
			assert(unlabeled is not None)
			cur_data, cur_idx = all_data[batch_order[i]]
			shared.has_gold = cur_data == data or cur_data == extra	# tag whether current batch has ground truth label
			shared.in_domain = cur_data == extra 	# tag whether current batch is snli

			(data_name, source, target, char_source, char_target, bert1, bert2,
				batch_ex_idx, batch_l, source_l, target_l, label, res_map) = cur_data[cur_idx]
			y_gold = Variable(label, requires_grad=False)

			if shared.has_gold:
				output = forward_pass_pair(m, source, target, char_source, char_target, bert1, bert2,
					batch_ex_idx, batch_l, source_l, target_l, res_map)
			else:
				output = forward_pass_flip(m, source, target, char_source, char_target, bert1, bert2,
					batch_ex_idx, batch_l, source_l, target_l, res_map)


		elif opt.fwd_mode == 'triple':
			assert(unlabeled is not None)
			cur_data, cur_idx = all_data[batch_order[i]]
			shared.has_gold = cur_data == data or cur_data == extra # tag whether current batch has ground truth label)
			shared.in_domain = cur_data == extra 	# tag whether current batch is snli

			# if current data is in triple mode
			if cur_data.third is not None:
				(data_name, source, target, third, char_source, char_target, char_third, bert1, bert2, bert3,
					batch_ex_idx, batch_l, source_l, target_l, third_l, label, res_map) = cur_data[cur_idx]

				y_gold = Variable(label, requires_grad=False)
				output = forward_pass_triple(m, source, target, third, char_source, char_target, char_third,
					bert1, bert2, bert3, batch_ex_idx, batch_l, source_l, target_l, third_l, res_map)
			else:
				(data_name, source, target, char_source, char_target, bert1, bert2,
					batch_ex_idx, batch_l, source_l, target_l, label, res_map) = cur_data[cur_idx]
	
				y_gold = Variable(label, requires_grad=False)
				output = forward_pass_pair(m, source, target, char_source, char_target, bert1, bert2,
					batch_ex_idx, batch_l, source_l, target_l, res_map)


		elif opt.fwd_mode == 'flip_and_triple':
			assert(unlabeled is not None)
			cur_data, cur_idx, triple_mode = all_data[batch_order[i]]
			shared.has_gold = cur_data == data or cur_data == extra # tag whether current batch has ground truth label)
			shared.in_domain = cur_data == extra 	# tag whether current batch is snli

			# if just pair, do it ordinarily (pair and flip)
			if not triple_mode:
				if cur_data.third is not None:
					(data_name, source, target, third, char_source, char_target, char_third, bert1, bert2, bert3,
					batch_ex_idx, batch_l, source_l, target_l, third_l, label, res_map) = cur_data[cur_idx]
				else:
					(data_name, source, target, char_source, char_target, bert1, bert2,
						batch_ex_idx, batch_l, source_l, target_l, label, res_map) = cur_data[cur_idx]
	
				y_gold = Variable(label, requires_grad=False)
				output = forward_pass_flip(m, source, target, char_source, char_target, bert1, bert2,
					batch_ex_idx, batch_l, source_l, target_l, res_map)

			else:
				(data_name, source, target, third, char_source, char_target, char_third, bert1, bert2, bert3,
					batch_ex_idx, batch_l, source_l, target_l, third_l, label, res_map) = cur_data[cur_idx]

				y_gold = Variable(label, requires_grad=False)
				output = forward_pass_triple(m, source, target, third, char_source, char_target, char_third,
					bert1, bert2, bert3, batch_ex_idx, batch_l, source_l, target_l, third_l, res_map)


		else:
			raise Exception("unrecognized fwd_mode: {0}".format(opt.fwd_mode))

		# loss
		batch_loss = loss(output, y_gold)

		# stats
		train_loss += float(batch_loss.data)
		num_ex += batch_l
		time_taken = time.time() - start_time
		acc_batch_size += batch_l

		# accumulate grads
		grad_norm2 = optim.backward(m, batch_loss)

		# accumulate current batch until the rolled up batch size exceeds threshold or meet certain boundary
		if i == data_size-1 or acc_batch_size >= opt.acc_batch_size or (i+1) % opt.print_every == 0:
			optim.step(m)
			shared.num_update += 1

			# clear up grad
			m.zero_grad()
			acc_batch_size = 0

			# stats
			grad_norm2_avg = grad_norm2
			min_grad_norm2 = min(min_grad_norm2, grad_norm2_avg)
			max_grad_norm2 = max(max_grad_norm2, grad_norm2_avg)
			time_taken = time.time() - start_time
			loss_stats = loss.print_cur_stats()

			if (i+1) % opt.print_every == 0:
				stats = '{0}, Batch {1:.1f}k '.format(epoch_id+1, float(i+1)/1000)
				stats += 'Grad {0:.1f}/{1:.1f} '.format(min_grad_norm2, max_grad_norm2)
				stats += 'Loss {0:.4f} '.format(train_loss / num_ex)
				stats += loss.print_cur_stats()
				stats += 'Time {0:.1f}'.format(time_taken)
				print(stats)

	perf, extra_perf = loss.get_epoch_metric()

	m.end_pass()
	loss.end_pass()

	return perf, extra_perf, train_loss / num_ex, num_ex

def train(opt, shared, m, optim, train_data, val_data, extra, extra_val, unlabeled):
	best_val_perf = 0.0
	test_perf = 0.0
	train_perfs = []
	val_perfs = []
	extra_perfs = []

	train_idx, train_num_ex = train_data.subsample(opt.percent)
	print('{0} examples sampled for training'.format(train_num_ex))
	print('for the record, the first 10 training batches are: {0}'.format(train_idx[:10]))
	# sample the same proportion from the dev set as well
	#	but we don't want this to be too small
	minimal_dev_num = max(int(train_num_ex * 0.1), 1000)
	val_idx, val_num_ex = val_data.subsample(opt.percent, minimal_num=minimal_dev_num)
	print('{0} examples sampled for dev'.format(val_num_ex))
	print('for the record, the first 10 dev batches are: {0}'.format(val_idx[:10]))

	extra_idx = None
	if extra is not None:
		extra_idx, extra_num_ex = extra.subsample(opt.percent)
		print('{0} examples sampled for extra training data'.format(extra_num_ex))
		print('for the record, the first 10 extra training batches are: {0}'.format(extra_idx[:10]))

	extra_val_idx = None
	if extra_val is not None:
		minimal_dev_num = max(int(train_num_ex * 0.1), 1000)
		extra_val_idx, extra_val_num_ex = extra_val.subsample(opt.percent, minimal_num=minimal_dev_num)
		print('{0} examples sampled for extra dev'.format(extra_val_num_ex))
		print('for the record, the first 10 extra dev batches are: {0}'.format(extra_val_idx[:10]))

	unlabeled_idx = None
	if unlabeled is not None:
		unlabeled_idx, unlabeled_num_ex = unlabeled.subsample(opt.unlabeled_percent)
		print('{0} examples sampled for unlabeled'.format(unlabeled_num_ex))
		print('for the record, the first 10 unlabeled batches are: {0}'.format(unlabeled_idx[:10]))

	shared.num_train_ex = train_num_ex
	if unlabeled is not  None:
		shared.num_train_ex += unlabeled_num_ex
	if extra is not None:
		shared.num_train_ex += extra_num_ex
	shared.num_update = 0
	start = 0
	for i in range(start, opt.epochs):
		train_perf, extra_train_perf, loss, num_ex = train_epoch(opt, shared, m, optim, train_data, i, train_idx, extra, extra_idx, unlabeled, unlabeled_idx)
		train_perfs.append(train_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_train_perf])
		print('Train {0:.4f} All {1}'.format(train_perf, extra_perf_str))

		# evaluate
		#	and save if it's the best model
		val_perf, extra_val_perf, val_loss, num_ex = validate(opt, shared, m, val_data, val_idx, extra_val, extra_val_idx)
		val_perfs.append(val_perf)
		extra_perfs.append(extra_val_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_val_perf])
		print('Val {0:.4f} All {1}'.format(val_perf, extra_perf_str))

		perf_table_str = ''
		cnt = 0
		print('Epoch  | Train | Val ...')
		for train_perf, extra_perf in zip(train_perfs, extra_perfs):
			extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
			perf_table_str += '{0}\t{1:.4f}\t{2}\n'.format(cnt+1, train_perf, extra_perf_str)
			cnt += 1
		print(perf_table_str)

		if val_perf > best_val_perf:
			best_val_perf = val_perf
			print('saving model to {0}'.format(opt.save_file))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
			save_opt(opt, '{0}.opt'.format(opt.save_file))

		else:
			print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))



def validate(opt, shared, m, val_data, val_idx, extra_val, extra_val_idx):
	m.train(False)
	shared.is_train = False

	val_loss = 0.0
	num_ex = 0

	# in evaluation mode, always use multiclass loss
	loss = MulticlassLoss(opt, shared)

	data_size = val_idx.size()[0]
	all_val = []
	for i in range(data_size):
		all_val.append((val_data, val_idx[i]))


	if extra_val is not None:
		print('combining validation set...')
		for i in range(extra_val_idx.size()[0]):
			all_val.append((extra_val, extra_val_idx[i]))
		data_size += extra_val_idx.size()[0]


	#data_size = val_idx.size()[0]
	print('validating on the {0} batches...'.format(data_size))

	loss.begin_pass()
	m.begin_pass()
	for i in range(data_size):
		cur_data, cur_idx = all_val[i]
		(data_name, source, target, char_source, char_target, bert1, bert2,
			batch_ex_idx, batch_l, source_l, target_l, label, res_map) = cur_data[cur_idx]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		cv_idx1 = Variable(char_source, requires_grad=False)
		cv_idx2 = Variable(char_target, requires_grad=False)
		y_gold = Variable(label, requires_grad=False)

		# update network parameters
		m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map)

		# forward pass
		pred = m.forward(wv_idx1, wv_idx2, cv_idx1, cv_idx2, bert1, bert2)

		# loss
		batch_loss = loss(pred, y_gold)

		# stats
		val_loss += float(batch_loss.data)
		num_ex += batch_l

	perf, extra_perf = loss.get_epoch_metric()
	m.end_pass()
	loss.end_pass()
	return (perf, extra_perf, val_loss / num_ex, num_ex)




def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.train_data = opt.dir + opt.train_data
	opt.extra_train_data = opt.dir + opt.extra_train_data
	opt.val_data = opt.dir + opt.val_data
	opt.extra_val_data = opt.dir + opt.extra_val_data
	opt.unlabeled_data = opt.dir + opt.unlabeled_data
	opt.train_res = '' if opt.train_res == ''  else ','.join([opt.dir + path for path in opt.train_res.split(',')])
	opt.val_res = '' if opt.val_res == ''  else ','.join([opt.dir + path for path in opt.val_res.split(',')])
	opt.unlabeled_res = '' if opt.unlabeled_res == ''  else ','.join([opt.dir + path for path in opt.unlabeled_res.split(',')])
	opt.word_vecs = opt.dir + opt.word_vecs
	opt.char_idx = opt.dir + opt.char_idx
	opt.dict = opt.dir + opt.dict
	opt.char_dict = opt.dir + opt.char_dict

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	print(opt)

	# build model
	m = Pipeline(opt, shared)
	optim = get_optimizer(opt, shared)

	# initializing from pretrained
	if opt.load_file != '':
		m.init_weight()
		print('loading pretrained model from {0}...'.format(opt.load_file))
		param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
		m.set_param_dict(param_dict)
	else:
		m.init_weight()
		model_parameters = filter(lambda p: p.requires_grad, m.parameters())
		num_params = sum([np.prod(p.size()) for p in model_parameters])
		print('total number of trainable parameters: {0}'.format(num_params))
	
	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu
	m = optim.build_optimizer(m)	# build optimizer after distributing model to devices

	# loading data
	train_res_files = None if opt.train_res == '' else opt.train_res.split(',')
	train_data = Data(opt, opt.train_data, train_res_files)
	val_res_files = None if opt.val_res == '' else opt.val_res.split(',')
	val_data = Data(opt, opt.val_data, val_res_files)


	# loading extra
	extra_train = None
	if opt.extra_train_data != opt.dir:
		extra_train = Data(opt, opt.extra_train_data, None)

	extra_val = None
	if opt.extra_val_data != opt.dir:
		extra_val = Data(opt, opt.extra_val_data, None)

	# loading unlabeled
	unlabeled = None
	if opt.unlabeled_data != opt.dir:
		triple_mode = opt.unlabeled_triple_mode==1
		unlabeled_res_files = None if opt.unlabeled_res == '' else opt.unlabeled_res.split(',')
		unlabeled = Data(opt, opt.unlabeled_data, unlabeled_res_files, triple_mode=triple_mode)


	print('{0} batches in train set'.format(train_data.size()))

	train(opt, shared, m, optim, train_data, val_data, extra_train, extra_val, unlabeled)



if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))