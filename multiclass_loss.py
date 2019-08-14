import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from sklearn.metrics import f1_score


# Multiclass Loss
class MulticlassLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(MulticlassLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.num_correct = 0
		self.num_ex = 0
		self.verbose = False
		# NOTE, do not creat loss node globally

		self.all_ex_idx = []
		self.all_pred = []
		self.all_gold = []

		self.conf_mat = [[0 for _ in range(self.opt.num_label)] for _ in range(self.opt.num_label)]
		self.conf_dict = {0: 0, 1: 2, 2: 1}	# we want to print in order E C N (default is E N C)
		

	def forward(self, pred, gold):
		log_p = pred
		batch_l = self.shared.batch_l
		assert(pred.shape == (batch_l, self.opt.num_label))

		# loss
		crit = torch.nn.NLLLoss(reduction='sum')	# for pytorch < 0.4.1, use size_average=False
		if self.opt.gpuid != -1:
			crit = crit.cuda(self.opt.gpuid)
		loss = crit(log_p, gold[:])

		# stats
		self.num_correct += np.equal(pick_label(log_p.data.cpu()), gold.cpu()).sum()
		self.num_ex += batch_l

		# other stats
		pred = pick_label(log_p.data.cpu())
		gold = gold.cpu()
		for ex_idx, p, g in zip(self.shared.batch_ex_idx, pred, gold):
			p = int(p)
			g = int(g)
			self.all_ex_idx.append(ex_idx)
			self.all_pred.append(p)
			self.all_gold.append(g)
			# update the confusion matrix
			self.conf_mat[self.conf_dict[g]][self.conf_dict[p]] += 1
				

		return loss


	# return a string of stats
	def print_cur_stats(self):
		stats = 'Acc {0:.3f} '.format(float(self.num_correct) / self.num_ex)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		macro_f1 = f1_score(np.asarray(self.all_gold), np.asarray(self.all_pred), average='macro')  
		weighted_f1 = f1_score(np.asarray(self.all_gold), np.asarray(self.all_pred), average='weighted')  

		acc = float(self.num_correct) / self.num_ex
		return acc, [acc, macro_f1, weighted_f1] 	# and any other scalar metrics	


	def begin_pass(self):
		# clear stats
		self.num_correct = 0
		self.num_ex = 0
		self.all_ex_idx = []
		self.all_pred = []
		self.all_gold = []
		self.conf_mat = [[0 for _ in range(self.opt.num_label)] for _ in range(self.opt.num_label)]

	def end_pass(self):
		if hasattr(self.opt, 'pred_output'):
			pred_path = self.opt.pred_output + '.pred.txt'
			print('writing predictions to {0}'.format(pred_path))
			with open(pred_path, 'w') as f:
				for idx, p in zip(self.all_ex_idx, self.all_pred):
					f.write('{0}\t{1}\n'.format(idx, p))

			print('confusion matrix of label {0}'.format([self.conf_dict[p] for p in [0,1,2]]))
			for row in self.conf_mat:
				print(row)

