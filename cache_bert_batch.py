import sys
import argparse
import h5py
import torch
import numpy as np
from pipeline import *
from holder import *
from data import *

def load_pipeline(opt, shared):
	# build model
	m = Pipeline(opt, shared)

	# initialization
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	if opt.fp16 == 1:
		m.half()

	if opt.gpuid != -1:
		m = m.cuda(self.opt.gpuid)

	return m


def cache(opt, shared, m, data, output):
	print_every = 1000
	batch_cnt = 0

	f = h5py.File(output, 'w')

	m.train(False)
	data.begin_pass()
	m.begin_pass()
	for i in range(data.size()):
		(data_name, batch_ex_idx, concated, char_concated, batch_l, concated_l, context_l, query_l,
				bert_seg, bert_tok, bert_mask, span, concated_span, context_start, res_map) = data[i]

		wv_idx = Variable(concated, requires_grad=False)
		cv_idx = Variable(char_concated, requires_grad=False)
		y_gold = Variable(concated_span, requires_grad=False)

		bert_pack = (bert_seg, bert_tok, bert_mask)

		m.update_context(batch_ex_idx, batch_l, concated_l, context_l, query_l, context_start, res_map)

		# forward pass
		with torch.no_grad():
			output = m.forward(wv_idx, cv_idx, bert_pack)

		bert_enc = shared.bert_enc
		bert_enc = bert_enc.cpu().float().numpy()
		f['{0}.concated_batch'.format(i)] = bert_enc

		batch_cnt += 1
		if batch_cnt % print_every == 0:
			print('processed {0} batches'.format(batch_cnt))

	m.end_pass()
	data.end_pass()

	f.close()


def main(arguments):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--load_file', help="Path to where model to be loaded.", default="models/bert_fix1_adam_lr000003")
	parser.add_argument('--dir', help="Path to the data dir", default="data/squad_v1_bert/")
	parser.add_argument('--data', help="Path to training data hdf5 file.", default="squad-val.hdf5")
	parser.add_argument('--word_vecs', help="The path to word embeddings", default = "glove.hdf5")
	parser.add_argument('--char_idx', help="The path to word2char index file", default = "char.idx.hdf5")
	parser.add_argument('--dict', help="The path to word dictionary", default = "squad.word.dict")
	parser.add_argument('--char_dict', help="The path to char dictionary", default = "char.dict.txt")

	parser.add_argument('--output', help="Prefix of output files", default="dev")

	parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
	parser.add_argument('--use_word_vec', help="Whether to use word vec", type=int, default=0)
	parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=300)
	parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
	parser.add_argument('--num_char', help="The number of char", type=int, default=189)
	parser.add_argument('--token_l', help="The maximal token length", type=int, default=16)
	parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
	parser.add_argument('--span_l', help="The maximal span length allowed for prediction", type=int, default=10000)
	# bert specs
	parser.add_argument('--fp16', help="Whether to use fp16 format", type=int, default=1)
	parser.add_argument('--fix_bert', help="Whether to fix bert update", type=int, default=1)
	parser.add_argument('--bert_size', help="The input elmo dim", type=int, default=768)
	parser.add_argument('--warmup_perc', help="The percentages of total expectec updates to warmup", default=0.1)
	parser.add_argument('--warmup_epoch', help="The number of epochs for warmup", type=int, default=2)
	## pipeline stages
	parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
	parser.add_argument('--cls', help="The type of classifier, linear", default='linear')

	opt = parser.parse_args(arguments)
	shared = Holder()

	# path correction
	opt.data = opt.dir + opt.data
	opt.output = opt.dir + opt.output
	opt.word_vecs = opt.dir + opt.word_vecs
	opt.char_idx = opt.dir + opt.char_idx
	opt.dict = opt.dir + opt.dict
	opt.char_dict = opt.dir + opt.char_dict

	# load data
	data = Data(opt, opt.data, res_files=None)

	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
	
	m = load_pipeline(opt, shared)

	cache(opt, shared, m, data, opt.output+'.bert.hdf5')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

