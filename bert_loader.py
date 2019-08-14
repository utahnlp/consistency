import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# the elmo loader
#	it takes no input but the current example idx
#	encodings are actually loaded from cached embeddings
class BertLoader(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BertLoader, self).__init__()
		self.opt = opt
		self.shared = shared

	def forward(self, concated, char_concated, bert_pack):
		bert_enc = self.shared.res_map['bert_concated']

		bert_enc = Variable(bert_enc, requires_grad=False)
		if self.opt.gpuid != -1:
			bert_enc = bert_enc.cuda(self.opt.gpuid)

		if self.opt.fp16 == 1:
			bert_enc = bert_enc.half()

		return bert_enc

	def begin_pass(self):
		pass

	def end_pass(self):
		pass

