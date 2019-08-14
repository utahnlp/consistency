import io
import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
import ujson
from util import *

class Data():
	def __init__(self, opt, data_file, res_files=None, triple_mode=False):
		self.opt = opt
		self.data_name = data_file

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.source = f['source'][:]	# indices to glove tokens
		self.target = f['target'][:]	# indices to glove tokens
		self.source_l = f['source_l'][:].astype(np.int32)
		self.target_l = f['target_l'][:].astype(np.int32)
		self.all_source = f['all_source'][:]
		self.all_target = f['all_target'][:]
		self.label = f['label'][:]
		self.batch_l = f['batch_l'][:].astype(np.int32)
		self.batch_idx = f['batch_idx'][:].astype(np.int32)
		self.bert_tok_idx1 = f['bert_tok_idx1'][:].astype(np.int32)
		self.bert_tok_idx2 = f['bert_tok_idx2'][:].astype(np.int32)
		self.ex_idx = f['ex_idx'][:].astype(np.int32)
		self.length = self.batch_l.shape[0]

		self.source = torch.from_numpy(self.source)
		self.target = torch.from_numpy(self.target)
		self.all_source = torch.from_numpy(self.all_source)
		self.all_target = torch.from_numpy(self.all_target)
		self.bert_tok_idx1 = torch.from_numpy(self.bert_tok_idx1)
		self.bert_tok_idx2 = torch.from_numpy(self.bert_tok_idx2)
		self.label = torch.from_numpy(self.label)

		# if triple presents
		self.third = None
		self.all_third = None
		self.third_l = None
		self.bert_tok_idx3 = None
		if 'third' in f and triple_mode is True:
			print('third sentence detected.')
			self.third = f['third'][:]
			self.all_third = f['all_third'][:]
			self.third_l = f['third_l'][:].astype(np.int32)
			self.bert_tok_idx3 = f['bert_tok_idx3'][:].astype(np.int32)

			self.third = torch.from_numpy(self.third)
			self.all_third = torch.from_numpy(self.all_third)
			self.bert_tok_idx3 = torch.from_numpy(self.bert_tok_idx3)

		# postpone the transfer to gpu to the batch running stage
		#if self.opt.gpuid != -1:
		#	self.source = self.source.cuda()
		#	self.target = self.target.cuda()
		#	self.span = self.span.cuda()

		# load char_idx file
		if opt.use_char_emb == 1:
			print('loading char idx from {0}'.format(opt.char_idx))
			f = h5py.File(opt.char_idx, 'r')
			self.char_idx = f['char_idx'][:]
			self.char_idx = torch.from_numpy(self.char_idx)
			assert(self.char_idx.shape[1] == opt.token_l)
			assert(self.char_idx.max()+1 == opt.num_char)
			print('{0} chars found'.format(self.char_idx.max()+1))

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]

			# get example token indices
			all_source_i = self.all_source[start:end, 0:self.source_l[i]]
			all_target_i = self.all_target[start:end, 0:self.target_l[i]]
			source_i = self.source[start:end, 0:self.source_l[i]]
			target_i = self.target[start:end, 0:self.target_l[i]]
			label_i = self.label[start:end]

			bert_tok1_i = self.bert_tok_idx1[start:end, 0:self.source_l[i]]
			bert_tok2_i = self.bert_tok_idx2[start:end, 0:self.target_l[i]]

			# sanity check
			assert(self.source[start:end, self.source_l[i]:].sum() == 0)
			assert(self.target[start:end, self.target_l[i]:].sum() == 0)

			if self.third is None:
				# src, tgt, all_src, all_tgt, batch_l, src_l, tgt_l, label, raw info
				self.batches.append((source_i, target_i, all_source_i, all_target_i, bert_tok1_i, bert_tok2_i,
					int(self.batch_l[i]), int(self.source_l[i]), int(self.target_l[i]), label_i))
			else:
				third_i = self.third[start:end, 0:self.third_l[i]]
				all_third_i = self.all_third[start:end, 0:self.third_l[i]]
				bert_tok3_i = self.bert_tok_idx3[start:end, 0:self.third_l[i]]
				assert(self.third[start:end, self.third_l[i]:].sum() == 0)

				# src, tgt, third, all_src, all_tgt, all_third, batch_l, src_l, tgt_l, third_l, label, raw info
				self.batches.append((source_i, target_i, third_i, all_source_i, all_target_i, all_third_i, bert_tok1_i, bert_tok2_i, bert_tok3_i,
					int(self.batch_l[i]), int(self.source_l[i]), int(self.target_l[i]), int(self.third_l[i]), label_i))


		# count examples
		self.num_ex = 0
		for i in range(self.length):
			self.num_ex += self.batch_l[i]


		# load resource files
		self.res_names = []
		if res_files is not None:
			for f in res_files:
				if f.endswith('txt'):
					res_names = self.__load_txt(f)

				elif f.endswith('json'):
					res_names = self.__load_json_res(f)

				else:
					assert(False)
				self.res_names.extend(res_names)


	def subsample(self, ratio, minimal_num=0):
		target_num_ex = int(float(self.num_ex) * ratio)
		target_num_ex = max(target_num_ex, minimal_num)
		sub_idx = torch.LongTensor(range(self.size()))
		sub_num_ex = 0

		if ratio != 1.0:
			rand_idx = torch.randperm(self.size())
			i = 0
			while sub_num_ex < target_num_ex and i < self.batch_l.shape[0]:
				sub_num_ex += self.batch_l[rand_idx[i]]
				i += 1
			sub_idx = rand_idx[:i]

		else:
			sub_num_ex = self.batch_l.sum()

		return sub_idx, sub_num_ex

	def split(self, sub_idx, ratio):
		num_ex = sum([self.batch_l[i] for i in sub_idx])
		target_num_ex = int(float(num_ex) * ratio)

		cur_num_ex = 0
		cur_pos = 0
		for i in range(len(sub_idx)):
			cur_pos = i
			cur_num_ex += self.batch_l[sub_idx[i]]
			if cur_num_ex >= target_num_ex:
				break

		return sub_idx[:cur_pos+1], sub_idx[cur_pos+1:], cur_num_ex, num_ex - cur_num_ex


	def __load_txt(self, path):
		lines = []
		print('loading resource from {0}'.format(path))
		# read file in unicode mode!!!
		with io.open(path, 'r+', encoding="utf-8") as f:
			for l in f:
				lines.append(l.rstrip())
		# the second last extension is the res name
		res_name = path.split('.')[-2]
		res_data = lines[:]

		# some customized parsing
		parsed = []
		parsed = res_data

		setattr(self, res_name, parsed)
		return [res_name]


	def __load_json_res(self, path):
		print('loading resource from {0}'.format(path))
		
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 2)
		res_type = next(iter(j_obj))

		res_name = None
		if j_obj[res_type] == 'map':
			res_name = self.__load_json_map(path)
		elif j_obj[res_type] == 'list':
			res_name = self.__load_json_list(path)
		else:
			assert(False)

		return [res_name]

	
	def __load_json_map(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)

		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			lut = {}
			for i, j in v.items():
				if i == res_name:
					lut[res_name] = [int(l) for l in j]
				else:
					lut[int(i)] = ([l for l in j[0]], [l for l in j[1]])

			res[int(k)] = lut
		
		setattr(self, res_name, res)
		return res_name


	def __load_json_list(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)
		
		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			p = v['p']
			h = v['h']

			# for token indices, shift by 1 to incorporate the nul-token at the beginning
			res[int(k)] = ([l for l in p], [l for l in h])
		
		setattr(self, res_name, res)
		return res_name


	def size(self):
		return self.length


	def __getitem__(self, idx):
		if self.third is None:
			(source, target, all_source, all_target, bert1, bert2,
				batch_l, source_l, target_l, label) = self.batches[idx]
		else:
			(source, target, third, all_source, all_target, all_third, bert1, bert2, bert3,
				batch_l, source_l, target_l, third_l, label) = self.batches[idx]
		token_l = self.opt.token_l

		# get char indices
		# 	the back forth data transfer should be eliminated
		char1, char2 = None, None
		if self.opt.use_char_emb == 1:
			char1 = self.char_idx[all_source.contiguous().view(-1)].view(batch_l, source_l, token_l)
			char2 = self.char_idx[all_target.contiguous().view(-1)].view(batch_l, target_l, token_l)

		# transfer to gpu if needed
		if self.opt.gpuid != -1:
			if self.opt.use_char_emb == 1:
				char1 = char1.cuda(self.opt.gpuid)
				char2 = char2.cuda(self.opt.gpuid)
			source = source.cuda(self.opt.gpuid)
			target = target.cuda(self.opt.gpuid)
			label = label.cuda(self.opt.gpuid)
			bert1 = bert1.long().cuda(self.opt.gpuid)
			bert2 = bert2.long().cuda(self.opt.gpuid)

		if self.third is not None:
			char3 = None
			if self.opt.use_char_emb == 1:
				char3 = char3.cuda(self.opt.gpuid)
				char3 = self.char_idx[all_third.contiguous().view(-1)].view(batch_l, third_l, token_l)
			
			if self.opt.gpuid != -1:
				if self.opt.use_char_emb == 1:
					char3 = char3.cuda(self.opt.gpuid)
				third = third.cuda(self.opt.gpuid)
				bert3 = bert3.long().cuda(self.opt.gpuid)

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		res_map = self.__get_res(idx)

		if self.third is None:
			return (self.data_name, source, target, char1, char2, bert1, bert2, 
				batch_ex_idx, batch_l, source_l, target_l, label, res_map)
		else:
			return (self.data_name, source, target, third, char1, char2, char3, bert1, bert2, bert3,
				batch_ex_idx, batch_l, source_l, target_l, third_l, label, res_map)


	def __get_res_elmo(self, res_name, idx, batch_ex_idx):
		if res_name == 'elmo_concated':
			embs = torch.from_numpy(self.elmo_file['{0}.concated_batch'.format(idx)][:])
			if self.opt.gpuid != -1:
				embs = embs.cuda(self.opt.gpuid)
			return embs
		else:
			raise Exception('unrecognized res {0}'.format(res_name))


	def __get_res_bert(self, res_name, idx, batch_ex_idx):
		if res_name == 'bert_concated':
			embs = torch.from_numpy(self.bert_file['{0}.concated_batch'.format(idx)][:])
			if self.opt.gpuid != -1:
				embs = embs.cuda(self.opt.gpuid)
			return embs
		else:
			raise Exception('unrecognized res {0}'.format(res_name))


	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None

		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		all_res = {}
		for res_n in self.res_names:
			# some customization for elmo is needed here for lazy loading
			if 'elmo' in res_n:
				batch_res = self.__get_res_elmo(res_n, idx, batch_ex_idx)
				all_res[res_n] = batch_res
			elif 'bert' in res_n:
				batch_res = self.__get_res_bert(res_n, idx, batch_ex_idx)
				all_res[res_n] = batch_res
			else:
				res = getattr(self, res_n)
				batch_res = [res[ex_id] for ex_id in batch_ex_idx]
				all_res[res_n] = batch_res

		return all_res


	# something at the beginning of each pass of training/eval
	#	e.g. setup preloading
	def begin_pass(self):
		pass


	def end_pass(self):
		pass
