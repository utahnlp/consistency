import sys
sys.path.insert(0, '../transformers')
import torch
from torch import cuda
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *
from transformers import *

# encoder with Elmo
class BertEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BertEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		self.zero = Variable(torch.zeros(1), requires_grad=False)
		self.zero = to_device(self.zero, self.opt.bert_gpuid)
		
		print('loading BERT model...')
		self.bert = self._get_bert(self.opt.bert_type)

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


	def _get_bert(self, key):
		model_map={"bert-base-uncased": (BertModel, BertTokenizer),
			"gpt2": (GPT2Model, GPT2Tokenizer),
			"roberta-base": (RobertaModel, RobertaTokenizer)}
		model_cls, _ = model_map[key]
		return model_cls.from_pretrained(key)


	def forward(self, sent1, sent2, char_sent1, char_sent2, bert1, bert2):
		bert1 = to_device(bert1, self.opt.bert_gpuid)
		bert2 = to_device(bert2, self.opt.bert_gpuid)
		bert_tok = torch.cat([bert1, bert2[:, 1:]], 1)	# removing the heading [CLS]

		if self.opt.fix_bert == 1:
			with torch.no_grad():
				last, pooled = self.bert(bert_tok, None)
		else:
			last, pooled = self.bert(bert_tok, None)

		last = last + pooled.unsqueeze(1) * self.zero

		# move to the original device
		last = to_device(last, self.opt.gpuid)

		self.shared.bert_enc = last
		
		return last


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


