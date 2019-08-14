import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# E_alpha and E_beta -> E_gamma
class Transition1(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Transition1, self).__init__()
		self.opt = opt
		self.shared = shared
		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.one = Variable(torch.ones(1), requires_grad=False)
		if opt.gpuid != -1:
			self.zero = self.zero.cuda(opt.gpuid)
			self.one = self.one.cuda(opt.gpuid)
		if opt.fp16 == 1:
			self.zero = self.zero.half()
			self.one = self.one.half()

	def forward(self, log_y_alpha, log_y_beta, log_y_gamma, gold):
		return torch.max(self.zero, log_y_alpha[:, 0] + log_y_beta[:, 0] - log_y_gamma[:, 0])

# E_alpha and C_beta -> C_gamma
class Transition2(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Transition2, self).__init__()
		self.opt = opt
		self.shared = shared
		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.one = Variable(torch.ones(1), requires_grad=False)
		if opt.gpuid != -1:
			self.zero = self.zero.cuda(opt.gpuid)
			self.one = self.one.cuda(opt.gpuid)
		if opt.fp16 == 1:
			self.zero = self.zero.half()
			self.one = self.one.half()

	def forward(self, log_y_alpha, log_y_beta, log_y_gamma, gold):
		return torch.max(self.zero, log_y_alpha[:, 0] + log_y_beta[:, 2] - log_y_gamma[:, 2])


# N_alpha and E_beta -> not C_gamma
class Transition3(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Transition3, self).__init__()
		self.opt = opt
		self.shared = shared
		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.one = Variable(torch.ones(1), requires_grad=False)
		if opt.gpuid != -1:
			self.zero = self.zero.cuda(opt.gpuid)
			self.one = self.one.cuda(opt.gpuid)
		if opt.fp16 == 1:
			self.zero = self.zero.half()
			self.one = self.one.half()

	def forward(self, log_y_alpha, log_y_beta, log_y_gamma, gold):
		very_small = 1e-4 if self.opt.fp16 == 1 else 1e-8	# for fp16, we need a larger small number:)
		log_not_y_gamma = (self.one - log_y_gamma.exp()).clamp(very_small).log()
		return torch.max(self.zero, log_y_alpha[:, 1] + log_y_beta[:, 0] - log_not_y_gamma[:, 2])


# N_alpha and C_beta -> not E_gamma
class Transition4(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Transition4, self).__init__()
		self.opt = opt
		self.shared = shared
		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.one = Variable(torch.ones(1), requires_grad=False)
		if opt.gpuid != -1:
			self.zero = self.zero.cuda(opt.gpuid)
			self.one = self.one.cuda(opt.gpuid)
		if opt.fp16 == 1:
			self.zero = self.zero.half()
			self.one = self.one.half()

	def forward(self, log_y_alpha, log_y_beta, log_y_gamma, gold):
		very_small = 1e-4 if self.opt.fp16 == 1 else 1e-8
		log_not_y_gamma = (self.one - log_y_gamma.exp()).clamp(very_small).log()
		return torch.max(self.zero, log_y_alpha[:, 1] + log_y_beta[:, 2] - log_not_y_gamma[:, 0])


# this loss is for fliping triple
# C_alpha <-> C_beta
class Transition5(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Transition5, self).__init__()
		self.opt = opt
		self.shared = shared
		self.zero = Variable(torch.zeros(1), requires_grad=False)
		if opt.gpuid != -1:
			self.zero = self.zero.cuda(opt.gpuid)
		if opt.fp16 == 1:
			self.zero = self.zero.half()

	def forward(self, log_y_alpha, log_y_beta, log_y_gamma, gold):
		# zero out grad on log_y_alpha here
		#	and leave it to the cross entropy loss
		log_c_alpha = Variable(log_y_alpha[:, 2].data, requires_grad=False)
		log_c_beta = log_y_beta[:, 2]
		return torch.abs(log_c_alpha - log_c_beta)


# this loss is for fliping triple
# C_alpha <-> C_beta
class Transition6(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Transition6, self).__init__()
		self.opt = opt
		self.shared = shared
		self.zero = Variable(torch.zeros(1), requires_grad=False)
		if opt.gpuid != -1:
			self.zero = self.zero.cuda(opt.gpuid)
		if opt.fp16 == 1:
			self.zero = self.zero.half()

	def forward(self, log_y_alpha, log_y_beta, log_y_gamma, gold):
		log_c_alpha = log_y_alpha[:, 2]
		log_c_beta = log_y_beta[:, 2]
		return torch.abs(log_c_alpha - log_c_beta)


# this loss is for fliping triple
# C_alpha <-> C_beta
class Transition7(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Transition7, self).__init__()
		self.opt = opt
		self.shared = shared
		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.one = Variable(torch.ones(1), requires_grad=False)
		self.half = Variable(torch.ones(1)*0.5, requires_grad=False)
		if opt.gpuid != -1:
			self.zero = self.zero.cuda(opt.gpuid)
			self.one = self.one.cuda(opt.gpuid)
			self.half = self.half.cuda(opt.gpuid)
		if opt.fp16 == 1:
			self.zero = self.zero.half()
			self.one = self.one.half()
			self.half = self.half.half()

	def forward(self, log_y_alpha, log_y_beta, log_y_gamma, gold):
		cross_ent = -log_y_alpha.gather(1, gold.view(-1, 1)).view(-1)	# (batch_l,)
		# maps {0, 1} -> {0, -1}
		mask = -(gold.view(-1) == 2).float()

		return -self.half * cross_ent + self.half * log_y_beta[:, 2] * mask


# Transition Loss with multiclass loss
class TransitionLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(TransitionLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.num_correct = 0
		self.num_ex = 0
		self.verbose = False
		self.labeled_coverage_cnt = 0
		self.unlabeled_pair_coverage_cnt = 0
		self.unlabeled_triple_coverage_cnt = 0
		# NOTE, do not creat loss node globally

		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.lambd = Variable(torch.ones(1) * opt.lambd, requires_grad=False)
		self.lambd_p = Variable(torch.ones(1) * opt.lambd_p, requires_grad=False)
		self.lambd_t = Variable(torch.ones(1) * opt.lambd_t, requires_grad=False)
		self.eta = Variable(torch.ones(1) * opt.eta, requires_grad=False)
		if opt.gpuid != -1:
			self.zero = self.zero.cuda(opt.gpuid)
			self.lambd = self.lambd.cuda(opt.gpuid)
			self.lambd_p = self.lambd_p.cuda(opt.gpuid)
			self.lambd_t = self.lambd_t.cuda(opt.gpuid)
			self.eta = self.eta.cuda(opt.gpuid)
		if opt.fp16 == 1:
			self.zero = self.zero.half()
			self.lambd = self.lambd.half()
			self.lambd_p = self.lambd_p.half()
			self.lambd_t = self.lambd_t.half()
			self.eta = self.eta.half()

		
	def get_mirror_constrs(self):
		constrs = []
		for n in self.opt.constrs.split(','):
			if n == '5':
				constrs.append(Transition5(self.opt, self.shared))
			elif n == '6':
				constrs.append(Transition6(self.opt, self.shared))
			elif n == '7':
				constrs.append(Transition7(self.opt, self.shared))
		return constrs

	def get_transition_constrs(self):
		constrs = []
		for n in self.opt.constrs.split(','):
			if n == '1':
				constrs.append(Transition1(self.opt, self.shared))
			elif n == '2':
				constrs.append(Transition2(self.opt, self.shared))
			elif n == '3':
				constrs.append(Transition3(self.opt, self.shared))
			elif n == '4':
				constrs.append(Transition4(self.opt, self.shared))
		return constrs


	def get_lambd(self):
		if self.opt.dynamic_lambd != 1:
			return self.lambd

		ratio = self.shared.num_update / self.shared.data_size
		lambd = 1.0 - np.exp(-ratio)
		lambd = Variable(torch.ones(1) * lambd, requires_grad=False)
		if self.opt.gpuid != -1:
			lambd = lambd.cuda(opt.gpuid)
		if self.opt.fp16 == 1:
			lambd = lambd.half()
		return lambd


	def count_coverage(self, loss, labeled, is_triple):
		for l in loss.data.cpu().float():
			if l > 0:
				if labeled:
					self.labeled_coverage_cnt += 1
				else:
					if is_triple:
						self.unlabeled_triple_coverage_cnt += 1
					else:
						self.unlabeled_pair_coverage_cnt += 1

	
	def forward(self, pack, gold):
		batch_l = self.shared.batch_l
		log_alpha, log_beta, log_gamma = pack

		if self.shared.has_gold:
			l_loss = -log_alpha.gather(1, gold.view(-1, 1)).view(-1)	# (batch_l,)

			if self.shared.in_domain:
				l_loss = l_loss * self.eta
			
			# in triple mode, we don't do transition regularization with gold label
			#	the same in pair_and_unlabeled mode where we don't want to use regularizer on example with gold label
			t_loss = self.zero
			if self.opt.fwd_mode != 'triple' and self.opt.fwd_mode != 'pair_and_unlabeled':
				lambd = self.get_lambd()
				constrs = self.get_mirror_constrs()
				for c in constrs:
					t_loss = t_loss + c(log_alpha, log_beta, log_gamma, gold) * lambd
				self.count_coverage(t_loss, labeled=True, is_triple=False)
	
			loss = l_loss + t_loss

		else:
			loss = self.zero

			if log_gamma is None:
				lambd = self.lambd_p
				constrs = self.get_mirror_constrs()
				is_triple = False
			else:
				lambd = self.lambd_t
				constrs = self.get_transition_constrs()
				is_triple = True
				
			for c in constrs:
				loss = loss + c(log_alpha, log_beta, log_gamma, gold) * lambd	
			self.count_coverage(loss, labeled=False, is_triple=is_triple)


		# stats
		self.num_correct += np.equal(pick_label(log_alpha.data.cpu()), gold.cpu()).sum()
		self.num_ex += batch_l

		loss = loss.sum()

		return loss


	# return a string of stats
	def print_cur_stats(self):
		stats = 'Acc {0:.3f} Cov {1}/{2}/{3} '.format(float(self.num_correct) / self.num_ex, self.labeled_coverage_cnt, self.unlabeled_pair_coverage_cnt, self.unlabeled_triple_coverage_cnt)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		acc = float(self.num_correct) / self.num_ex
		return acc, [acc] 	# and any other scalar metrics	


	def begin_pass(self):
		# clear stats
		self.num_correct = 0
		self.num_ex = 0
		self.labeled_coverage_cnt = 0
		self.unlabeled_pair_coverage_cnt = 0
		self.unlabeled_triple_coverage_cnt = 0

	def end_pass(self):
		print('trained on {0} examples.'.format(self.num_ex))
		print('labeled_coverage_cnt: {0}'.format(self.labeled_coverage_cnt))
		print('unlabeled_pair_coverage_cnt: {0}'.format(self.unlabeled_pair_coverage_cnt))
		print('unlabeled_triple_coverage_cnt: {0}'.format(self.unlabeled_triple_coverage_cnt))

