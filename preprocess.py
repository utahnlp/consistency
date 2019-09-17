import sys
sys.path.insert(0, '../pytorch-transformers')
import os
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json
import torch
from torch import cuda
from pytorch_transformers import *

def get_tokenizer(key):
	model_map={"bert-base-uncased": (BertModel, BertTokenizer),
		"gpt2": (GPT2Model, GPT2Tokenizer),
		"roberta-base": (RobertaModel, RobertaTokenizer)}
	model_cls, tokenizer_cls = model_map[key]
	print('loading tokenizer: {0}'.format(key))
	tokenizer = tokenizer_cls.from_pretrained(key)
	return tokenizer

class Indexer:
	def __init__(self, symbols = ["<blank>"], num_oov=100):
		self.num_oov = num_oov

		self.d = {}
		self.cnt = {}
		for s in symbols:
			self.d[s] = len(self.d)
			self.cnt[s] = 0
			
		for i in range(self.num_oov): #hash oov words to one of 100 random embeddings
			oov_word = '<oov'+ str(i) + '>'
			self.d[oov_word] = len(self.d)
			self.cnt[oov_word] = 10000000	# have a large number for oov word to avoid being pruned
			
	def convert(self, w):		
		return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(self.num_oov)) + '>']

	def convert_sequence(self, ls):
		return [self.convert(l) for l in ls]

	def write(self, outfile, with_cnt=True):
		print(len(self.d), len(self.cnt))
		assert(len(self.d) == len(self.cnt))
		with open(outfile, 'w+') as f:
			items = [(v, k) for k, v in self.d.items()]
			items.sort()
			for v, k in items:
				if with_cnt:
					f.write('{0} {1} {2}\n'.format(k, v, self.cnt[k]))
				else:
					f.write('{0} {1}\n'.format(k, v))

	# register tokens only appear in wv
	#   NOTE, only do counting on training set
	def register_words(self, wv, seq, count):
		for w in seq:
			if w in wv and w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

	#   NOTE, only do counting on training set
	def register_all_words(self, seq, count):
		for w in seq:
			if w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

			
def pad(ls, length, symbol, pad_back = True):
	if len(ls) >= length:
		return ls[:length]
	if pad_back:
		return ls + [symbol] * (length -len(ls))
	else:
		return [symbol] * (length -len(ls)) + ls		

def get_glove_words(f):
	glove_words = set()
	for line in open(f, "r"):
		word = line.split()[0].strip()
		glove_words.add(word)
	return glove_words


def make_vocab(opt, glove_vocab, word_indexer, all_word_indexer, label_indexer, sent1, sent2, label, max_seq_l, count):
	num_ex = 0
	for _, (src_orig, targ_orig, l) in enumerate(zip(sent1, sent2, label)):
		targ = targ_orig.strip().split()
		src = src_orig.strip().split()
		l = l.rstrip()

		assert(len(targ) <= max_seq_l and len(src) <= max_seq_l)

		all_word_indexer.register_all_words(targ, count)
		word_indexer.register_words(glove_vocab, targ, count)

		all_word_indexer.register_all_words(src, count)
		word_indexer.register_words(glove_vocab, src, count)

		label_indexer.register_all_words([l], count)
		num_ex += 1

	return num_ex


def make_vocab_triple(opt, glove_vocab, word_indexer, all_word_indexer, label_indexer, sent1, sent2, sent3, label, max_seq_l, count):
	num_ex = 0
	for _, (src_orig, targ_orig, third_orig, l) in enumerate(zip(sent1, sent2, sent3, label)):

		targ = targ_orig.strip().split()
		src = src_orig.strip().split()
		third = third_orig.strip().split()
		l = l.rstrip()

		assert(len(targ) <= max_seq_l and len(src) <= max_seq_l)

		all_word_indexer.register_all_words(targ, count)
		word_indexer.register_words(glove_vocab, targ, count)

		all_word_indexer.register_all_words(src, count)
		word_indexer.register_words(glove_vocab, src, count)

		all_word_indexer.register_all_words(third, count)
		word_indexer.register_words(glove_vocab, third, count)

		label_indexer.register_all_words([l], count)
		num_ex += 1

	return num_ex


def convert(opt, tokenizer, word_indexer, all_word_indexer, label_indexer, sent1, sent2, label, output, num_ex):
	np.random.seed(opt.seed)

	max_seq_l = opt.max_seq_l + 1 #add 1 for BOS

	bert_tok_idx1 = np.zeros((num_ex, max_seq_l), dtype=int)
	bert_tok_idx2 = np.zeros((num_ex, max_seq_l), dtype=int)
	# better generate seg and mask on the fly
	targets = np.zeros((num_ex, max_seq_l), dtype=int)
	sources = np.zeros((num_ex, max_seq_l), dtype=int)
	all_sources = np.zeros((num_ex, max_seq_l), dtype=int)
	all_targets = np.zeros((num_ex, max_seq_l), dtype=int)
	labels = np.zeros((num_ex,), dtype =int)
	source_lengths = np.zeros((num_ex,), dtype=int)
	target_lengths = np.zeros((num_ex,), dtype=int)
	ex_idx = np.zeros(num_ex, dtype=int)
	batch_keys = np.array([None for _ in range(num_ex)])
	
	ex_id = 0
	for _, (src_orig, targ_orig, label_orig) in enumerate(zip(sent1, sent2, label)):
		targ_orig =  targ_orig.strip().split()
		src_orig =  src_orig.strip().split()
		label = label_orig.strip()

		src = pad(src_orig, max_seq_l, '<blank>')
		src = word_indexer.convert_sequence(src)
		   
		targ = pad(targ_orig, max_seq_l, '<blank>')
		targ = word_indexer.convert_sequence(targ)

		all_src = pad(src_orig, max_seq_l, '<blank>')
		all_src = all_word_indexer.convert_sequence(all_src)

		all_targ = pad(targ_orig, max_seq_l, '<blank>')
		all_targ = all_word_indexer.convert_sequence(all_targ)

		bert_tok_idx1[ex_id, :len(src_orig)] = np.asarray(tokenizer.convert_tokens_to_ids(src_orig))
		bert_tok_idx2[ex_id, :len(targ_orig)] = np.asarray(tokenizer.convert_tokens_to_ids(targ_orig))
		
		sources[ex_id] = np.array(src, dtype=int)
		targets[ex_id] = np.array(targ,dtype=int)
		all_sources[ex_id] = np.array(all_src, dtype=int)
		all_targets[ex_id] = np.array(all_targ, dtype=int)
		source_lengths[ex_id] = (sources[ex_id] != 0).sum() 
		target_lengths[ex_id] = (targets[ex_id] != 0).sum()
		labels[ex_id] = label_indexer.d[label]
		batch_keys[ex_id] = (source_lengths[ex_id], target_lengths[ex_id])
		ex_id += 1
		if ex_id % 100000 == 0:
			print("{}/{} sentences processed".format(ex_id, num_ex))
	
	print(ex_id, num_ex)
	if opt.shuffle == 1:
		rand_idx = np.random.permutation(ex_id)
		targets = targets[rand_idx]
		sources = sources[rand_idx]
		all_sources = all_sources[rand_idx]
		all_targets = all_targets[rand_idx]
		source_lengths = source_lengths[rand_idx]
		target_lengths = target_lengths[rand_idx]
		labels = labels[rand_idx]
		batch_keys = batch_keys[rand_idx]
		ex_idx = rand_idx
		bert_tok_idx1 = bert_tok_idx1[rand_idx]
		bert_tok_idx2 = bert_tok_idx2[rand_idx]
	
	# break up batches based on source/target lengths
	sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
	sorted_idx = [i for i, _ in sorted_keys]
	# rearrange examples	
	sources = sources[sorted_idx]
	targets = targets[sorted_idx]
	all_sources = all_sources[sorted_idx]
	all_targets = all_targets[sorted_idx]
	labels = labels[sorted_idx]
	target_l = target_lengths[sorted_idx]
	source_l = source_lengths[sorted_idx]
	ex_idx = rand_idx[sorted_idx]
	bert_tok_idx1 = bert_tok_idx1[sorted_idx]
	bert_tok_idx2 = bert_tok_idx2[sorted_idx]
	
	curr_l_src = 0
	curr_l_targ = 0
	batch_location = [] #idx where sent length changes
	for j,i in enumerate(sorted_idx):
		if batch_keys[i][0] != curr_l_src or batch_keys[i][1] != curr_l_targ:
			curr_l_src = source_lengths[i]
			curr_l_targ = target_lengths[i]
			batch_location.append(j)
	if batch_location[-1] != len(sources): 
		batch_location.append(len(sources)-1)
	
	#get batch sizes
	curr_idx = 0
	batch_idx = [0]
	for i in range(len(batch_location)-1):
		end_location = batch_location[i+1]
		while curr_idx < end_location:
			curr_idx = min(curr_idx + opt.batch_size, end_location)
			batch_idx.append(curr_idx)

	batch_l = []
	target_l_new = []
	source_l_new = []
	for i in range(len(batch_idx)):
		end = batch_idx[i+1] if i < len(batch_idx)-1 else len(sources)
		batch_l.append(end - batch_idx[i])
		source_l_new.append(source_l[batch_idx[i]])
		target_l_new.append(target_l[batch_idx[i]])
		
		# sanity check
		for k in range(batch_idx[i], end):
			assert(source_l[k] == source_l_new[-1])
			assert(sources[k, source_l[k]:].sum() == 0)

	
	# Write output
	f = h5py.File(output, "w")		
	f["source"] = sources
	f["target"] = targets
	f["label"] = labels
	f['all_source'] = all_sources
	f['all_target'] = all_targets
	f["target_l"] = np.array(target_l_new, dtype=int)
	f["source_l"] = np.array(source_l_new, dtype=int)
	f["batch_l"] = batch_l
	f["batch_idx"] = batch_idx
	f['ex_idx'] = ex_idx
	f['bert_tok_idx1'] = bert_tok_idx1
	f['bert_tok_idx2'] = bert_tok_idx2
	print("saved {} batches ".format(len(f["batch_l"])))
	f.close()  


def tokenize_and_write(tokenizer, path, output):
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	print('tokenizing sentences from {0}'.format(path))
	all_tokenized = []
	with open(path, 'r') as f:
		for l in f:
			if l.strip() == '':
				continue
			
			toks = tokenizer.tokenize(l)
			toks = [CLS] + toks + [SEP]
			all_tokenized.append(' '.join(toks))

	print('writing tokenized to {0}'.format(output))
	with open(output, 'w') as f:
		for seq in all_tokenized:
			f.write(seq + '\n')

	return all_tokenized

def load(path):
	all_lines = []
	with open(path, 'r') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			all_lines.append(l.strip())
	return all_lines


def process(opt):
	tokenizer = get_tokenizer(opt.bert_type)

	all_word_indexer = Indexer(symbols = ["<blank>", tokenizer.cls_token, tokenizer.sep_token])	# all tokens will be recorded
	word_indexer = Indexer(symbols = ["<blank>", tokenizer.cls_token, tokenizer.sep_token])		# only glove tokens will be recorded
	glove_vocab = get_glove_words(opt.glove)
	label_indexer = Indexer(symbols=["entailment", "neutral", "contradiction"], num_oov=0)

	# adding oov words (DEPRECATED)
	oov_words = []
	for i in range(0,100): #hash oov words to one of 100 random embeddings, per Parikh et al. 2016
		oov_words.append('<oov'+ str(i) + '>')
	word_indexer.register_all_words(oov_words, count=False)
	all_word_indexer.register_all_words(oov_words, count=False)

	#### tokenize
	tokenizer_output = opt.tokenizer_output+'.' if opt.tokenizer_output != opt.dir else opt.dir
	sent1 = tokenize_and_write(tokenizer, opt.sent1, tokenizer_output + 'train.sent1.txt')
	sent2 = tokenize_and_write(tokenizer, opt.sent2, tokenizer_output + 'train.sent2.txt')
	label = load(opt.label)
	sent1_val = tokenize_and_write(tokenizer, opt.sent1_val, tokenizer_output + 'dev.sent1.txt')
	sent2_val = tokenize_and_write(tokenizer, opt.sent2_val, tokenizer_output + 'dev.sent2.txt')
	label_val = load(opt.label_val)
	sent1_test = tokenize_and_write(tokenizer, opt.sent1_test, tokenizer_output + 'test.sent1.txt')
	sent2_test = tokenize_and_write(tokenizer, opt.sent2_test, tokenizer_output + 'test.sent2.txt')
	label_test = load(opt.label_test)

	print("First pass through data to get vocab...")

	num_train = make_vocab(opt, glove_vocab, word_indexer, all_word_indexer, label_indexer, sent1, sent2, label, opt.max_seq_l, count=True)
	print("Number of examples in training: {}".format(num_train))
	print("Number of sentences in training: {0}, number of tokens: {1}/{2}".format(num_train, len(word_indexer.d), len(all_word_indexer.d)))

	num_valid = make_vocab(opt, glove_vocab, word_indexer, all_word_indexer, label_indexer, sent1_val, sent2_val, label_val, opt.max_seq_l, count=True)
	print("Number of examples in valid: {}".format(num_valid))
	print("Number of sentences in valid: {0}, number of tokens: {1}/{2}".format(num_valid, len(word_indexer.d), len(all_word_indexer.d))) 

	num_test = make_vocab(opt, glove_vocab, word_indexer, all_word_indexer, label_indexer, sent1_test, sent2_test, label_test, opt.max_seq_l, count=False)	# no counting on test set
	print("Number of examples in test: {}".format(num_test))

	word_indexer.write(opt.output + ".word.dict")
	all_word_indexer.write(opt.output + ".allword.dict")
	label_indexer.write(opt.output + ".label.dict")
	print("vocab size: {}".format(len(word_indexer.d)))
	assert(len(label_indexer.d) == 3)

	convert(opt, tokenizer, word_indexer, all_word_indexer, label_indexer, sent1, sent2, label, opt.output + ".train.hdf5", num_train)
	convert(opt, tokenizer, word_indexer, all_word_indexer, label_indexer, sent1_val, sent2_val, label_val, opt.output + ".val.hdf5", num_valid)
	convert(opt, tokenizer, word_indexer, all_word_indexer, label_indexer, sent1_test, sent2_test, label_test, opt.output + ".test.hdf5", num_test)	
	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--sent1', help="Path to sent1 training data.", default = "train.raw.sent1.txt")
	parser.add_argument('--sent2', help="Path to sent2 training data.", default = "train.raw.sent2.txt")
	parser.add_argument('--label', help="Path to label data", default = "train.label.txt")	
	parser.add_argument('--sent1_val', help="Path to sent1 validation data.",default = "dev_excl_anno.raw.sent1.txt")
	parser.add_argument('--sent2_val', help="Path to sent2 validation data.", default = "dev_excl_anno.raw.sent2.txt")
	parser.add_argument('--label_val', help="Path to label validation data.",default = "dev_excl_anno.label.txt")
	parser.add_argument('--sent1_test', help="Path to sent1 test data.",default = "test.raw.sent1.txt")
	parser.add_argument('--sent2_test', help="Path to sent2 test data.",default = "test.raw.sent2.txt")
	parser.add_argument('--label_test', help="Path to label test data.",default = "test.label.txt")
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/bert_nli/")
	parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")

	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=48)
	parser.add_argument('--max_seq_l', help="Maximum sequence length. Sequences longer than this are dropped.", type=int, default=400)
	parser.add_argument('--tokenizer_output', help="Prefix of the tokenized output file names. ", type=str, default = "")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "rte")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type = int, default = 1)
	parser.add_argument('--seed', help="The random seed", type = int, default = 1)
	parser.add_argument('--glove', type = str, default = '')
	opt = parser.parse_args(arguments)

	opt.sent1 = opt.dir + opt.sent1
	opt.sent2 = opt.dir + opt.sent2
	opt.sent1_val = opt.dir + opt.sent1_val
	opt.sent2_val = opt.dir + opt.sent2_val
	opt.sent1_test = opt.dir + opt.sent1_test
	opt.sent2_test = opt.dir + opt.sent2_test
	opt.label = opt.dir + opt.label
	opt.label_val = opt.dir + opt.label_val
	opt.label_test = opt.dir + opt.label_test
	opt.output = opt.dir + opt.output
	opt.tokenizer_output = opt.dir + opt.tokenizer_output

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
