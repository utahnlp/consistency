import sys
import math
import torch
from torch import nn
from holder import *
from util import *
from bert_adam import *
#from apex.fp16_utils import FP16_Optimizer
from fp16_optimizer import FP16_Optimizer
from apex.optimizers import FusedAdam

class Adagrad:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.clip = opt.clip if opt.clip > 0.0  else 10000000000.0

	def step(self, m):
		params = [p for p in m.parameters() if p.requires_grad]
		if self.optim is None:
			self.optim = torch.optim.Adagrad(params, lr=self.opt.learning_rate)

		grad_norm2 = nn.utils.clip_grad_norm_(params, self.clip, norm_type=2)

		self.optim.step()

		return grad_norm2

			
class Adam:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.clip = opt.clip if opt.clip > 0.0  else 10000000000.0
		self.betas = [float(b) for b in opt.adam_betas.split(',')]

	def __step(self, named_params):
		params = [p[1] for p in named_params]
		if self.optim is None:
			self.optim = torch.optim.Adam(params, lr=self.opt.learning_rate, betas=self.betas)

		grad_norm2 = nn.utils.clip_grad_norm_(params, self.clip, norm_type=2)

		self.optim.step()

		return grad_norm2

	def step(self, m):
		params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		return self.__step(named_params)


# the huggingface's adam for bert
class AdamBert:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.betas = [float(b) for b in opt.adam_betas.split(',')]

	def step_by_params(self, named_params):
		params = [p for n, p in named_params if p.requires_grad]
		if self.optim is None:
			self.optim = BertAdam(params, lr=self.opt.learning_rate, max_grad_norm=self.opt.clip, b1=self.betas[0], b2=self.betas[1])

		self.optim.step()

		for n, p in m.named_parameters():
			if has_nan(p.data):
				print(n, p.data)
				assert(False)

		return 0.0	# just return 0 for grad_norm2

	def step(self, m):
		named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		return self.step_by_params(named_params)


# the apex's adam for fp16
class AdamFp16:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		
	def build_optimizer(self, named_params):
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [{'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.opt.weight_decay},
			{'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		fused_adam = FusedAdam(optimizer_grouped_parameters,
			lr=self.opt.learning_rate,
			bias_correction=False,
			max_grad_norm=self.opt.clip)
		#params = [p for n, p in named_params if p.requires_grad]
		#fused_adam = BertAdam(params, lr=self.opt.learning_rate, max_grad_norm=self.opt.clip, weight_decay=self.opt.weight_decay)
		return FP16_Optimizer(fused_adam, dynamic_loss_scale=True)

	def get_lr(self):
		#if self.opt.warmup_epoch <= 0:
		#	return self.opt.learning_rate
			
		avg_batch_size = 40	# this is just a roughly guess
		acc_l = avg_batch_size if self.opt.acc_batch_size < 0 else self.opt.acc_batch_size
		normalizer = self.shared.num_train_ex / acc_l * self.opt.epochs
		return self.opt.learning_rate * warmup_linear_flat(self.shared.num_update / normalizer, self.opt.warmup_perc)

	def step_by_params(self, named_params):
		if self.optim is None:
			self.optim = self.build_optimizer(named_params)

		cur_lr = self.get_lr()
		for param_group in self.optim.param_groups:
			param_group['lr'] = cur_lr

		self.optim.step()
		self.optim.zero_grad()

		return 0.0	# just return 0 for grad_norm2


	def step(self, m):
		named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		rs = self.step_by_params(named_params)
		return rs

	def backward_by_params(self, named_params, loss, retain_graph=False):
		if self.optim is None:
			self.optim = self.build_optimizer(named_params)
		self.optim.backward(loss, retain_graph=retain_graph)

	# this interface is only for apex's optimizer
	def backward(self, m, loss):
		named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		self.backward_by_params(named_params, loss)

# not really working
class AdamFp16Shared:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.bert_optim = AdamFp16(opt, shared)
		self.optim = AdamFp16(opt, shared)
		#self.union_optim = AdamFp16(opt, shared)

	def step(self, m):
		bert_m = [(n, p) for n, p in m.named_parameters() if p.requires_grad and hasattr(p, 'is_bert')]
		m = [(n, p) for n, p in m.named_parameters() if p.requires_grad and not hasattr(p, 'is_bert')]

		#torch.cuda.set_device(self.opt.bert_gpuid)
		#bert_skip = self.bert_optim.optim.should_skip()
#
		#torch.cuda.set_device(self.opt.gpuid)
		#normal_skip = self.optim.optim.should_skip()
#
		#if bert_skip or normal_skip:
		#	print('skip step for consistency across gpus.')
		#	return 0.0

		#print('before', bert_m[0][1].device, m[0][1].device)
		torch.cuda.set_device(self.opt.bert_gpuid)
		self.bert_optim.step_by_params(bert_m)

		torch.cuda.set_device(self.opt.gpuid)
		self.optim.step_by_params(m)

		# ideally, the two devices should be different
		#print('after', bert_m[0][1].device, m[0][1].device)

		return 0.0

	def backward(self, m, loss):
		bert_m = [(n, p) for n, p in m.named_parameters() if p.requires_grad and hasattr(p, 'is_bert')]
		m = [(n, p) for n, p in m.named_parameters() if p.requires_grad and not hasattr(p, 'is_bert')]

		#torch.cuda.set_device(self.opt.bert_gpuid)
		#bert_loss = loss.cuda(self.opt.bert_gpuid)
		#if self.bert_optim.optim is None:
		#	self.bert_optim.optim = self.bert_optim.build_optimizer(bert_m)
		#self.bert_optim.backward_by_params(bert_m, bert_loss, retain_graph=True)

		torch.cuda.set_device(self.opt.gpuid)
		normal_loss = loss.cuda(self.opt.gpuid)
		self.optim.backward_by_params(m, normal_loss, retain_graph=False)

		print(normal_loss.data)

		#self.union_optim.backward(m, loss)
		#print(bert_m[0][1].device, m[0][1].device)

		

class Adamax:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.clip = opt.clip if opt.clip > 0.0  else 10000000000.0
		self.betas = [float(b) for b in opt.adam_betas.split(',')]

	def step(self, m):
		params = [p for p in m.parameters() if p.requires_grad]
		if self.optim is None:
			self.optim = torch.optim.Adamax(params, lr=self.opt.learning_rate, betas=self.betas)

		grad_norm2 = nn.utils.clip_grad_norm_(params, self.clip, norm_type=2)

		self.optim.step()

		return grad_norm2


class Adadelta:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.clip = opt.clip if opt.clip > 0.0  else 10000000000.0

	def step(self, m):
		params = [p for p in m.parameters() if p.requires_grad]
		if self.optim is None:
			self.optim = torch.optim.Adadelta(params, lr=self.opt.learning_rate, rho=0.95)

		grad_norm2 = nn.utils.clip_grad_norm_(params, self.clip, norm_type=2)

		self.optim.step()

		return grad_norm2


def get_optimizer(opt, shared):
	optim = None
	if opt.optim == 'adagrad':
		optim = Adagrad(opt, shared)
	elif opt.optim == 'adam':
		optim = Adam(opt, shared)
	elif opt.optim == 'adam_bert':
		optim = AdamBert(opt, shared)
	elif opt.optim == 'adam_fp16':
		optim = AdamFp16(opt, shared)
	elif opt.optim == 'adam_fp16_shared':
		optim = AdamFp16Shared(opt, shared)
	elif opt.optim == 'adamax':
		optim = Adamax(opt, shared)
	elif opt.optim == 'adadelta':
		optim = Adadelta(opt, shared)
	else:
		print('unrecognized optim: {0}'.format(opt.optim))
		assert(False)
	return optim


def grad_sanity_check(optim, m, batch_size):
	optim.__SANITY_FLAG = False
	for n, p in m.named_parameters():
		if p.requires_grad:
			if p.grad is None:
				if optim.__SANITY_FLAG == False:
					print('{0} requires grad but has no grad, double check your graph'.format(n))
			else:
				if p.grad.is_sparse:
					print('sparse gradient found.')
					assert(False)
				p.grad.data.div_(batch_size)

	optim.__SANITY_FLAG = True


