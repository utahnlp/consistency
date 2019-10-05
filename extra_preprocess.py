import sys
sys.path.insert(0, '../transformers')
import os
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json
import torch
from torch import cuda
from transformers import *

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

	def set_word(self, w, idx, count):
		self.d[w] = idx
		self.cnt[w] = count

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



def load_vocab_to_indexer(path, word_indexer):
	vocab = set()
	with open(path, 'r') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			p = l.split()
			tok, idx, cnt = p[0], int(p[1]), int(p[2])
			vocab.add(tok)
			#
			word_indexer.set_word(tok, idx, cnt)


def convert(opt, tokenizer, word_indexer, all_word_indexer, label_indexer, sent1, sent2, label, output, num_ex):
	np.random.seed(opt.seed)

	bert_tok_idx1 = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	bert_tok_idx2 = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	# better generate seg and mask on the fly
	#bert_seg_idx1 = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	#bert_mask = np.zeros((num_ex, opt.max_seq_l), dtype=int)
		
	max_seq_l = opt.max_seq_l + 1 #add 1 for BOS
	targets = np.zeros((num_ex, max_seq_l), dtype=int)
	sources = np.zeros((num_ex, max_seq_l), dtype=int)
	all_sources = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	all_targets = np.zeros((num_ex, opt.max_seq_l), dtype=int)
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

		all_src = pad(src_orig, opt.max_seq_l, '<blank>')
		all_src = all_word_indexer.convert_sequence(all_src)

		all_targ = pad(targ_orig, opt.max_seq_l, '<blank>')
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


def convert_triple(opt, tokenizer, word_indexer, all_word_indexer, label_indexer, sent1, sent2, sent3, label, output, num_ex):
	np.random.seed(opt.seed)

	bert_tok_idx1 = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	bert_tok_idx2 = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	bert_tok_idx3 = np.zeros((num_ex, opt.max_seq_l), dtype=int)
		
	max_seq_l = opt.max_seq_l + 1 #add 1 for BOS
	targets = np.zeros((num_ex, max_seq_l), dtype=int)
	sources = np.zeros((num_ex, max_seq_l), dtype=int)
	thirds = np.zeros((num_ex, max_seq_l), dtype=int)
	all_sources = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	all_targets = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	all_thirds = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	labels = np.zeros((num_ex,), dtype =int)
	source_lengths = np.zeros((num_ex,), dtype=int)
	target_lengths = np.zeros((num_ex,), dtype=int)
	third_lengths = np.zeros((num_ex,), dtype=int)
	ex_idx = np.zeros(num_ex, dtype=int)
	batch_keys = np.array([None for _ in range(num_ex)])
	
	ex_id = 0
	for _, (src_orig, targ_orig, third_orig, label_orig) in enumerate(zip(sent1, sent2, sent3, label)):
		targ_orig =  targ_orig.strip().split()
		src_orig =  src_orig.strip().split()
		third_orig =  third_orig.strip().split()
		label = label_orig.strip()

		src = pad(src_orig, max_seq_l, '<blank>')
		src = word_indexer.convert_sequence(src)
		   
		targ = pad(targ_orig, max_seq_l, '<blank>')
		targ = word_indexer.convert_sequence(targ)

		third = pad(third_orig, max_seq_l, '<blank>')
		third = word_indexer.convert_sequence(third)

		all_src = pad(src_orig, opt.max_seq_l, '<blank>')
		all_src = all_word_indexer.convert_sequence(all_src)

		all_targ = pad(targ_orig, opt.max_seq_l, '<blank>')
		all_targ = all_word_indexer.convert_sequence(all_targ)

		all_third = pad(third_orig, opt.max_seq_l, '<blank>')
		all_third = all_word_indexer.convert_sequence(all_third)

		bert_tok_idx1[ex_id, :len(src_orig)] = np.asarray(tokenizer.convert_tokens_to_ids(src_orig))
		bert_tok_idx2[ex_id, :len(targ_orig)] = np.asarray(tokenizer.convert_tokens_to_ids(targ_orig))
		bert_tok_idx3[ex_id, :len(third_orig)] = np.asarray(tokenizer.convert_tokens_to_ids(third_orig))
		
		sources[ex_id] = np.array(src, dtype=int)
		targets[ex_id] = np.array(targ,dtype=int)
		thirds[ex_id] = np.array(third, dtype=int)
		all_sources[ex_id] = np.array(all_src, dtype=int)
		all_targets[ex_id] = np.array(all_targ, dtype=int)
		all_thirds[ex_id] = np.array(all_third, dtype=int)
		source_lengths[ex_id] = (sources[ex_id] != 0).sum() 
		target_lengths[ex_id] = (targets[ex_id] != 0).sum()
		third_lengths[ex_id] = (thirds[ex_id] != 0).sum()
		labels[ex_id] = label_indexer.d[label]
		batch_keys[ex_id] = (source_lengths[ex_id], target_lengths[ex_id], third_lengths[ex_id])
		ex_id += 1
		if ex_id % 100000 == 0:
			print("{}/{} sentences processed".format(ex_id, num_ex))
	
	print(ex_id, num_ex)
	if opt.shuffle == 1:
		rand_idx = np.random.permutation(ex_id)
		targets = targets[rand_idx]
		sources = sources[rand_idx]
		thirds = thirds[rand_idx]
		all_sources = all_sources[rand_idx]
		all_targets = all_targets[rand_idx]
		all_thirds = all_thirds[rand_idx]
		source_lengths = source_lengths[rand_idx]
		target_lengths = target_lengths[rand_idx]
		third_lengths = third_lengths[rand_idx]
		labels = labels[rand_idx]
		batch_keys = batch_keys[rand_idx]
		ex_idx = rand_idx
		bert_tok_idx1 = bert_tok_idx1[rand_idx]
		bert_tok_idx2 = bert_tok_idx2[rand_idx]
		bert_tok_idx3 = bert_tok_idx3[rand_idx]
	
	# break up batches based on source/target lengths
	sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
	sorted_idx = [i for i, _ in sorted_keys]
	# rearrange examples	
	sources = sources[sorted_idx]
	targets = targets[sorted_idx]
	thirds = thirds[sorted_idx]
	all_sources = all_sources[sorted_idx]
	all_targets = all_targets[sorted_idx]
	all_thirds = all_thirds[sorted_idx]
	labels = labels[sorted_idx]
	target_l = target_lengths[sorted_idx]
	source_l = source_lengths[sorted_idx]
	third_l = third_lengths[sorted_idx]
	ex_idx = rand_idx[sorted_idx]
	bert_tok_idx1 = bert_tok_idx1[sorted_idx]
	bert_tok_idx2 = bert_tok_idx2[sorted_idx]
	bert_tok_idx3 = bert_tok_idx3[sorted_idx]
	
	curr_l_src = 0
	curr_l_targ = 0
	curr_l_third = 0
	batch_location = [] #idx where sent length changes
	for j,i in enumerate(sorted_idx):
		if batch_keys[i][0] != curr_l_src or batch_keys[i][1] != curr_l_targ or batch_keys[i][2] != curr_l_third:
			curr_l_src = source_lengths[i]
			curr_l_targ = target_lengths[i]
			curr_l_third = third_lengths[i]
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
	third_l_new = []
	for i in range(len(batch_idx)):
		end = batch_idx[i+1] if i < len(batch_idx)-1 else len(sources)
		batch_l.append(end - batch_idx[i])
		source_l_new.append(source_l[batch_idx[i]])
		target_l_new.append(target_l[batch_idx[i]])
		third_l_new.append(third_l[batch_idx[i]])
		
		# sanity check
		for k in range(batch_idx[i], end):
			assert(source_l[k] == source_l_new[-1])
			assert(sources[k, source_l[k]:].sum() == 0)

	
	# Write output
	f = h5py.File(output, "w")		
	f["source"] = sources
	f["target"] = targets
	f['third'] = thirds
	f["label"] = labels
	f['all_source'] = all_sources
	f['all_target'] = all_targets
	f['all_third'] = all_thirds
	f["target_l"] = np.array(target_l_new, dtype=int)
	f["source_l"] = np.array(source_l_new, dtype=int)
	f["third_l"] = np.array(third_l_new, dtype=int)
	f["batch_l"] = batch_l
	f["batch_idx"] = batch_idx
	f['ex_idx'] = ex_idx
	f['bert_tok_idx1'] = bert_tok_idx1
	f['bert_tok_idx2'] = bert_tok_idx2
	f['bert_tok_idx3'] = bert_tok_idx3
	print("saved {} batches ".format(len(f["batch_l"])))
	f.close() 


def tokenize_and_write(tokenizer, path, output):
	print('tokenizing sentences from {0}'.format(path))
	all_tokenized = []
	with open(path, 'r') as f:
		for l in f:
			if l.strip() == '':
				continue
			
			toks = tokenizer.tokenize(l)
			toks = ['[CLS]'] + toks + ['[SEP]']
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
	
	do_triple = opt.sent3 != opt.dir
	if do_triple:
		print('sent3 detected, will process in triple mode.')

	all_word_indexer = Indexer(symbols = ["<blank>", tokenizer.cls_token, tokenizer.sep_token])	# all tokens will be recorded
	word_indexer = Indexer(symbols = ["<blank>", tokenizer.cls_token, tokenizer.sep_token])		# only glove tokens will be recorded
	load_vocab_to_indexer(opt.vocab, word_indexer)
	load_vocab_to_indexer(opt.vocab_all, all_word_indexer)
	glove_vocab = get_glove_words(opt.glove)
	label_indexer = Indexer(symbols=["entailment", "neutral", "contradiction"], num_oov=0)

	oov_words = []
	for i in range(0,100): #hash oov words to one of 100 random embeddings, per Parikh et al. 2016
		oov_words.append('<oov'+ str(i) + '>')
	word_indexer.register_all_words(oov_words, count=False)
	all_word_indexer.register_all_words(oov_words, count=False)

	#### tokenize
	sent1 = tokenize_and_write(tokenizer, opt.sent1, opt.tokenizer_output + '.sent1.txt')
	sent2 = tokenize_and_write(tokenizer, opt.sent2, opt.tokenizer_output + '.sent2.txt')
	if opt.label != opt.dir:
		label = load(opt.label)
	else:
		label = ['neutral' for _ in range(len(sent1))]

	if do_triple:
		sent3 = tokenize_and_write(tokenizer, opt.sent3, opt.tokenizer_output + '.sent3.txt')

	print("First pass through data to get vocab...")

	if do_triple:
		num_triple = make_vocab_triple(opt, glove_vocab, word_indexer, all_word_indexer, label_indexer, sent1, sent2, sent3, label, opt.max_seq_l, count=True)
		print("Number of examples: {0}".format(num_triple))
		print("number of tokens: {0}/{1}".format(len(word_indexer.d), len(all_word_indexer.d)))
	else:
		num_ex = make_vocab(opt, glove_vocab, word_indexer, all_word_indexer, label_indexer, sent1, sent2, label, opt.max_seq_l, count=True)
		print("Number of examples: {0}".format(num_ex))
		print("number of tokens: {0}/{1}".format(len(word_indexer.d), len(all_word_indexer.d)))


	word_indexer.write(opt.output + ".word.dict")
	all_word_indexer.write(opt.output + ".allword.dict")
	label_indexer.write(opt.output + ".label.dict")
	print("vocab size: {}".format(len(word_indexer.d)))
	print(label_indexer.d)
	assert(len(label_indexer.d) == 3)

	if do_triple:
		convert_triple(opt, tokenizer, word_indexer, all_word_indexer, label_indexer, sent1, sent2, sent3, label, opt.output + ".hdf5", num_triple)
	else:
		convert(opt, tokenizer, word_indexer, all_word_indexer, label_indexer, sent1, sent2, label, opt.output + ".hdf5", num_ex)

	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--sent1', help="Path to sent1 extra data.", default = "anno.raw.sent1.txt")
	parser.add_argument('--sent2', help="Path to sent2 extra data.", default = "anno.raw.sent2.txt")
	parser.add_argument('--sent3', help="Path to sent3 extra data.", default = "")
	parser.add_argument('--label', help="Path to label data (optional)", default = "")
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/bert_nli/")	
	parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
	parser.add_argument('--vocab', help="Path to the glove vocabulary preprocessed", default = "./data/bert_nli/snli.word.dict")
	parser.add_argument('--vocab_all', help="Path to the all word vocabulary preprocessed", default = "./data/bert_nli/snli.allword.dict")
	
	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=32)
	parser.add_argument('--max_seq_l', help="Maximum sequence length. Sequences longer than this are dropped.", type=int, default=350)
	parser.add_argument('--tokenizer_output', help="Prefix of the tokenized output file names. ", type=str, default = "")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type = int, default = 1)
	parser.add_argument('--seed', help="The random seed", type = int, default = 1)
	parser.add_argument('--glove', type = str, default = '')
	opt = parser.parse_args(arguments)

	opt.sent1 = opt.dir + opt.sent1
	opt.sent2 = opt.dir + opt.sent2
	opt.sent3 = opt.dir + opt.sent3
	opt.label = opt.dir + opt.label
	opt.output = opt.dir + opt.output
	opt.tokenizer_output = opt.dir + opt.tokenizer_output

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
