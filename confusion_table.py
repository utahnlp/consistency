import argparse
import numpy as np

def load(path):
	with open(path, 'r') as f:
		preds = []
		for l in f:
			if l.strip() == '':
				continue

			toks = l.split()
			preds.append(int(toks[1]))
	return preds


def get_confusion_matrix(left, right, labels):
	mat = []
	for i in labels:
		mat.append([])
		for k in labels:
			mat[-1].append(sum([1 for p, q in zip(left, right) if p == i and q == k]))
	return mat

def get_confusion_ratio(mat):
	ratio = []
	for row in mat:
		ratio.append([])
		for item in row:
			ratio[-1].append(item/sum(row))
	return ratio

def get_error_rate(mat):
	error_cnt = mat[0][1] + mat[1][0] + mat[1][2] + mat[2][1]
	return error_cnt / np.asarray(mat).sum()

def get_stats(base, swapped):
	vio_cnt = 0
	precondition_cnt = 0
	for b, s in zip(base, swapped):
		if b == 2 or s == 2:
			precondition_cnt += 1
		if (b == 2 and s != 2) or (b != 2 and s == 2):
			vio_cnt += 1
	return vio_cnt, precondition_cnt


def process(base, swapped, labels):
	mat = get_confusion_matrix(base, swapped, [0, 2, 1])
	ratio = get_confusion_ratio(mat)
	er = get_error_rate(mat)
	vio_cnt, prec_cnt = get_stats(base, swapped)

	return mat, er, vio_cnt, prec_cnt


def main():
	parser = argparse.ArgumentParser(
		description =__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--log', help="The template of log to calculate confusion mat", default='models/lstm_clip5_adam_lr00001_perc1')

	opt = parser.parse_args()

	all_base =[]
	all_swapped = []
	for i in range(3):
		base = load(opt.log + '_seed{0}_swap0.pred.txt'.format(i+1))
		swapped = load(opt.log + '_seed{0}_swap1.pred.txt'.format(i+1))
		all_base.append(base)
		all_swapped.append(swapped)

	global_inc = []
	conditional_inc = []
	for b, s in zip(all_base, all_swapped):
		mat, er, vio_cnt, prec_cnt = process(b, s, [0, 2, 1])
		print(mat)
		print(er)
		print(vio_cnt, prec_cnt, vio_cnt/prec_cnt)
		global_inc.append(er)
		conditional_inc.append(vio_cnt/prec_cnt)

	print('average global_inc: {0}'.format(sum(global_inc)/3))
	print('average conditional_inc: {0}'.format(sum(conditional_inc)/3))


if __name__ == '__main__':
	main()