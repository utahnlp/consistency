import sys
sys.path.insert(0, '../pytorch-pretrained-BERT')
import torch
from torch import cuda
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# encoder with Elmo
class BertEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BertEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.zero = to_device(self.zero, self.opt.bert_gpuid)

		if opt.fp16 == 1:
			self.zero = self.zero.half()
		
		print('loading BERT model...')
		self.bert = BertModel.from_pretrained('bert-base-uncased')

		print('verifying BERT model...')
		self.bert.eval()

		for n in self.bert.children():
			for p in n.parameters():
				p.skip_init = True
				p.is_bert = True	# tag as bert fields

		# if to lock bert
		if opt.fix_bert == 1:
			for n in self.bert.children():
				for p in n.parameters():
					p.requires_grad = False

		self.customize_cuda_id = self.opt.bert_gpuid
		self.fp16 = opt.fp16 == 1


	def get_seg_mask(self):
		mask = torch.ones(self.shared.batch_l, self.shared.sent_l1+self.shared.sent_l2-1).long()
		mask = to_device(mask, self.opt.bert_gpuid)

		seg1 = torch.zeros(self.shared.batch_l, self.shared.sent_l1).long()
		seg2 = torch.ones(self.shared.batch_l, self.shared.sent_l2-1).long()	# removing the heading [CLS]
		seg = torch.cat([seg1, seg2], 1)
		seg = to_device(seg, self.opt.bert_gpuid)

		return seg, mask



	def forward(self, sent1, sent2, char_sent1, char_sent2, bert1, bert2):
		bert1 = to_device(bert1, self.opt.bert_gpuid)
		bert2 = to_device(bert2, self.opt.bert_gpuid)
		bert_tok = torch.cat([bert1, bert2[:, 1:]], 1)	# removing the heading [CLS]

		seg, mask = self.get_seg_mask()

		assert(seg.shape[1] == bert_tok.shape[1])

		if self.opt.fix_bert == 1:
			with torch.no_grad():
				last, pooled = self.bert(bert_tok, seg, mask, output_all_encoded_layers=False)
		else:
			last, pooled = self.bert(bert_tok, seg, mask, output_all_encoded_layers=False)

		last = last + pooled.unsqueeze(1) * self.zero

		# move to the original device
		last = to_device(last, self.opt.gpuid)

		self.shared.bert_enc = last
		
		return last


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


