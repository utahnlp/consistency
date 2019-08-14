import sys
import h5py
import torch
from torch import nn
from torch import cuda
import string
import re
from collections import Counter
import numpy as np

def build_boundary_selector(span_l):
	max_seq_l = 528	# something slightly larger than the actual max sequence l
	start_sel = []
	end_sel = []
	for l in range(1, span_l+1):
		m1 = torch.zeros(max_seq_l-l+1, max_seq_l)
		m2 = torch.zeros(max_seq_l-l+1, max_seq_l)
		for r in range(m1.shape[0]):
			m1[r, r] = 1.0
			m2[r, r+l-1] = 1.0
		start_sel.append(m1)
		end_sel.append(m2)
	return start_sel, end_sel


def build_span_selector(span_l):
	max_seq_l = 528	# something slightly larger than the actual max sequence l
	boundary_selectors = []
	for l in range(1, span_l+1):
		mask = torch.zeros(max_seq_l-l+1, max_seq_l)
		for r in range(mask.shape[0]):
			mask[r, r:r+l] = 1.0
		boundary_selectors.append(mask)
	return boundary_selectors


def spawn_spans(seq_l, max_span_l):
	boundaries = []
	for l in range(max_span_l):
		for k in range(seq_l-l):
			boundaries.append((k, k+l))
	return boundaries


def to_device(x, gpuid):
	if gpuid == -1:
		return x.cpu()
	if x.device != gpuid:
		return x.cuda(gpuid)
	return x

def has_nan(t):
	return torch.isnan(t).sum() == 1

def tensor_on_dev(t, is_cuda):
	if is_cuda:
		return t.cuda()
	else:
		return t

def pick_label(dist):
	return np.argmax(dist, axis=1)

def torch2np(t, is_cuda):
	return t.numpy() if not is_cuda else t.cpu().numpy()

def save_opt(opt, path):
	with open(path, 'w') as f:
		f.write('{0}'.format(opt))


def load_param_dict(path):
	# TODO, this is ugly
	f = h5py.File(path, 'r')
	return f


def save_param_dict(param_dict, path):
	file = h5py.File(path, 'w')
	for name, p in param_dict.items():
		file.create_dataset(name, data=p)

	file.close()


def load_dict(path):
	rs = {}
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			w, idx, cnt = l.strip().split()
			rs[int(idx)] = w
	return rs


def rand_tensor(shape, r1, r2):
	return (r1 - r2) * torch.rand(shape) + r2


def build_rnn(type, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional):
	if type == 'lstm':
		return nn.LSTM(input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bias=bias,
			batch_first=batch_first,
			dropout=dropout,
			bidirectional=bidirectional)
	elif type == 'gru':
		return nn.GRU(input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			bias=bias,
			batch_first=batch_first,
			dropout=dropout,
			bidirectional=bidirectional)
	else:
		assert(False)


###### official evaluation
# TODO, for unicode, there are versions of punctuations (esp. brackets)
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


###### official evaluation
def f1_bow(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

###### offcial evaluation
def em_bow(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

###### official evaluation
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Given a prediction and multiple valid answers, return the score of the best
    prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# tok_idx is scalar integer
# sent_ls is a list of integer for sentence lengths
def get_sent_idx(tok_idx, sent_ls):
	sent_idx = -1
	acc_l = 0
	for i, l in enumerate(sent_ls):
		acc_l += l
		if tok_idx < acc_l:
			sent_idx = i
			break
	assert(sent_idx != -1)
	return sent_idx

# the gold is a single span pf gold token_idx
#	could be the start or the end
def get_em_sent(pred_tok_idx, gold_tok_idx, context_sent_l):
	pred_sent_idx = torch.Tensor([get_sent_idx(int(idx), sent_l) for idx, sent_l in zip(pred_tok_idx, context_sent_l)])
	gold_sent_idx = torch.Tensor([get_sent_idx(int(idx), sent_l) for idx, sent_l in zip(gold_tok_idx, context_sent_l)])
	return (pred_sent_idx == gold_sent_idx).float()


def get_sent(tok_idx, context_sent_l, batch_token_span, batch_raw):
	sent_idx = [get_sent_idx(int(idx), sent_l) for idx, sent_l in zip(tok_idx, context_sent_l)]
	raw_sent = []
	for i, idx in enumerate(sent_idx):
		start = sum(context_sent_l[i][:idx])
		end = start + context_sent_l[i][idx]-1
		start = batch_token_span[i][start][0]
		end = batch_token_span[i][end][1]
		assert(start != -1)
		assert(end != -1)
		raw_sent.append(batch_raw[i][start:end+1])
	return raw_sent
	

# pick the best span given a maximal length
def pick_best_span_bounded(log_p1, log_p2, bound):
	log_p1, log_p2 = log_p1.cpu(), log_p2.cpu()
	assert(len(log_p1.shape) == 2)	# (batch_l, context_l)
	assert(len(log_p2.shape) == 2)
	batch_l, context_l = log_p1.shape
	cross = log_p1.unsqueeze(-1) + log_p2.unsqueeze(1)
	# build mask to search within bound steps
	mask = torch.ones(context_l, context_l).triu().tril(bound-1).unsqueeze(0)
	valid = cross * mask + (1.0 - mask) * -1e8

	spans = torch.zeros(batch_l, 2).long()
	for i in range(batch_l):
		max_idx = np.argmax(valid[i])
		max_idx = np.unravel_index(max_idx, valid[i].shape)
		spans[i] = torch.LongTensor(max_idx)
	return spans


def pick_idx(p):
	p = p.cpu().numpy()
	return np.argmax(p, axis=1)

def count_correct_idx(pred, gold):
	return np.equal(pred, gold).sum()


def get_answer_tokenized(token_idx1, token_idx2, tokenized, tok_to_orig_tok_map):
	assert(len(token_idx1.shape) == 1)
	assert(token_idx1.shape[0] == len(tokenized))

	batch_ans = []
	for i, (idx1, idx2) in enumerate(zip(token_idx1, token_idx2)):
		toks = tokenized[i][idx1:idx2+1]
		orig_tok_idx = tok_to_orig_tok_map[i][idx1:idx2+1]

		# compact answer
		ans_str = ''
		prev_orig_tok_idx = -1
		for t, orig_t_idx in zip(toks, orig_tok_idx):
			if orig_t_idx == prev_orig_tok_idx:
				# meaning they should merge
				ans_str += t
			else:
				ans_str += ' ' + t
			prev_orig_tok_idx = orig_t_idx

		ans_str = ans_str.strip()
		ans_str = ans_str.replace(' ##', '')
		ans_str = ans_str.replace('##', '')
		#
		batch_ans.append(ans_str)
	return batch_ans


def get_em_bow(pred_ans, gold_ans):
	assert(len(pred_ans) == len(gold_ans))
	ems = []
	for pred, gold in zip(pred_ans, gold_ans):
		ems.append(metric_max_over_ground_truths(em_bow, pred, gold))
	return ems

def get_f1_bow(pred_ans, gold_ans):
	assert(len(pred_ans) == len(gold_ans))
	f1s = []
	for pred, gold in zip(pred_ans, gold_ans):
		f1s.append(metric_max_over_ground_truths(f1_bow, pred, gold))
	return f1s


def get_norm2(t):
	return (t * t).sum()




if __name__ == '__main__':
	s1 = 'something in common (NAC)(PAG)'
	s2 = 'something weird'
	print(s1)
	print(s2)
	print(f1_bow(s1, s2))