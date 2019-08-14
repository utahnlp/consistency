import sys
import torch
from torch import nn
from torch import cuda
from holder import *
import numpy as np
from optimizer import *
import time
from embeddings import *
from bert_encoder import *
from linear_classifier import *


class Pipeline(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Pipeline, self).__init__()

		self.shared = shared
		self.opt = opt

		if opt.use_word_vec == 1:
			self.embeddings = WordVecLookup(opt, shared)

		# pipeline stages
		if opt.enc == 'bert':
			self.encoder = BertEncoder(opt, shared)
		else:
			assert(False)

		if opt.cls == 'linear':
			self.classifier = LinearClassifier(opt, shared)
		else:
			assert(False)


	def init_weight(self):
		missed_names = []
		if self.opt.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_uniform_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'xavier_normal':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_normal_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'no':
			for n, p in self.named_parameters():
				missed_names.append(n)
		else:
			assert(False)

		if len(missed_names) != 0:
			print('uninitialized fields: {0}'.format(missed_names))


	def forward(self, sent1, sent2, char1, char2, bert1, bert2):
		shared = self.shared

		if self.opt.use_word_vec == 1:
			sent1 = self.embeddings(sent1)	# (batch_l, sent_l1, word_vec_size)
			sent2 = self.embeddings(sent2)	# (batch_l, sent_l2, word_vec_size)
		else:
			sent1, sent2 = None, None

		# encoder
		enc = self.encoder(sent1, sent2, None, None, bert1, bert2)

		# classifier
		output = self.classifier(enc)

		# bookkeeping
		shared.enc = enc
		shared.output = output

		return output

	# call this explicitly
	def update_context(self, batch_ex_idx, batch_l, sent_l1, sent_l2, res_map=None):
		self.shared.batch_ex_idx = batch_ex_idx
		self.shared.batch_l = batch_l
		self.shared.sent_l1 = sent_l1
		self.shared.sent_l2 = sent_l2
		self.shared.res_map = res_map


	def begin_pass(self):
		if self.opt.use_word_vec == 1:
			self.embeddings.begin_pass()
		self.encoder.begin_pass()
		self.classifier.begin_pass()


	def end_pass(self):
		if self.opt.use_word_vec == 1:
			self.embeddings.end_pass()
		self.encoder.end_pass()
		self.classifier.end_pass()


	def distribute(self):
		modules = []
		if self.opt.use_word_vec == 1:
			modules.append(self.embeddings)
		modules.append(self.encoder)
		modules.append(self.classifier)

		for m in modules:
			if hasattr(m, 'fp16') and  m.fp16:
				m.half()

			if hasattr(m, 'customize_cuda_id'):
				print('pushing module to customized cuda id: {0}'.format(m.customize_cuda_id))
				m.cuda(m.customize_cuda_id)
			else:
				print('pushing module to default cuda id: {0}'.format(self.opt.gpuid))
				m.cuda(self.opt.gpuid)


	def get_param_dict(self):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		skipped_fields = []
		for n, p in self.named_parameters():
			# save all parameters that do not have skip_save flag
			# 	unlearnable parameters will also be saved
			if not hasattr(p, 'skip_save') or p.skip_save == 0:
				param_dict[n] =  torch2np(p.data, is_cuda)
			else:
				skipped_fields.append(n)
		#print('skipped fields:', skipped_fields)
		return param_dict

	def set_param_dict(self, param_dict):
		skipped_fields = []
		rec_fields = []
		for n, p in self.named_parameters():
			if n in param_dict:
				rec_fields.append(n)
				# load everything we have
				print('setting {0}'.format(n))
				p.data.copy_(torch.from_numpy(param_dict[n][:]))
			else:
				skipped_fields.append(n)
		print('skipped fileds: {0}'.format(skipped_fields))
