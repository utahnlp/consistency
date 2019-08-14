import sys
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from locked_dropout import *
from bert_loader import *
from bert_encoder import *
from util import *

class RnnClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(RnnClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		bidir = True
		hidden_state = opt.hidden_size if not bidir else opt.hidden_size//2

		self.rnn = build_rnn(
			opt.rnn_type,
			input_size=opt.bert_size, 
			hidden_size=hidden_state, 
			num_layers=1,
			bias=True,
			batch_first=True,
			dropout=opt.dropout,
			bidirectional=bidir)

		self.drop = LockedDropout(opt.dropout)

		self.linear = nn.Sequential(
			#nn.Dropout(opt.dropout),	# no dropout here according to huggingface
			nn.Linear(opt.hidden_size+opt.bert_size, 2))	# 1 for start, 1 for end


	def rnn_over(self, rnn, enc):
		enc, _ = rnn(self.drop(enc))
		return enc

	def fp32(self, x):
		if x.dtype != torch.float32:
			return x.float()
		return x


	def forward(self, concated):
		batch_l, concated_l, bert_size = concated.shape

		concated = self.fp32(concated)

		rnn_enc = self.rnn_over(self.rnn, concated).contiguous()

		phi = torch.cat([rnn_enc, concated], 2)

		scores = self.linear(phi.view(-1, bert_size+self.opt.hidden_size)).view(batch_l, concated_l, 2)

		log_p = nn.LogSoftmax(1)(scores)

		log_p1 = log_p[:, :, 0]
		log_p2 = log_p[:, :, 1]

		self.shared.y_scores = scores

		return [log_p1, log_p2]

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass




	
