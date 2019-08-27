import sys
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from locked_dropout import *

# linear classifier
class LinearClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(LinearClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		# weights will be initialized later
		self.linear = nn.Sequential(
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_label))

		self.fp16 = opt.fp16 == 1


	def forward(self, concated):
		batch_l, concated_l, enc_size = concated.shape

		head = concated[:, 0, :]

		scores = self.linear(head)	# (batch_l, num_label)

		log_p = nn.LogSoftmax(1)(scores)

		self.shared.y_scores = scores

		return log_p


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		
