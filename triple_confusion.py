import argparse
import numpy as np

def load(path):
	with open(path, 'r') as f:
		preds = []
		ex_idx = []
		for l in f:
			if l.strip() == '':
				continue

			toks = l.split()
			ex_idx.append(int(toks[0]))
			preds.append(int(toks[1]))
	return ex_idx, preds


def get_stats(alpha, beta, gamma):
	constr_cnt1 = [0, 0]	# antecedent fires, statement violation
	constr_cnt2 = [0, 0]
	constr_cnt3 = [0, 0]
	constr_cnt4 = [0, 0]
	violation_idx = []
	for idx, (a, b, c) in enumerate(zip(alpha, beta, gamma)):
		# E and E -> E
		if a == 0 and b == 0:
			constr_cnt1[0] += 1
			if c != 0:
				constr_cnt1[1] += 1
				violation_idx.append([idx, 0])
		# E and C -> C
		if a == 0 and b == 2:
			constr_cnt2[0] += 1
			if c != 2:
				constr_cnt2[1] += 1
				violation_idx.append([idx, 1])
		# N and E -> not C
		if a == 1 and b == 0:
			constr_cnt3[0] += 1
			if c == 2:
				constr_cnt3[1] += 1
				violation_idx.append([idx, 2])
		# N and C -> not E
		if a == 1 and b == 2:
			constr_cnt4[0] += 1
			if c == 0:
				constr_cnt4[1] += 1
				violation_idx.append([idx, 3])

	return [constr_cnt1, constr_cnt2, constr_cnt3, constr_cnt4], violation_idx


def get_confusion(alpha, beta, gamma, labels):
	groups = [alpha, beta, gamma]

	all_cnts = []
	for g in groups:
		cnts = [0, 0, 0]
		for p in g:
			cnts[labels.index(p)] += 1

		all_cnts.append(cnts)
	return all_cnts


def process(log, labels):
	print(log)
	ex_idx, alpha = load(log + '_alpha.pred.txt')
	ex_idx, beta = load(log + '_beta.pred.txt')
	ex_idx, gamma = load(log + '_gamma.pred.txt')
	confs = get_confusion(alpha, beta, gamma, labels)
	for conf in confs:
		print(conf)

	num_ex = len(alpha)
	stats, violation_idx = get_stats(alpha, beta, gamma)
	print(stats)

	violation_idx = [(ex_idx[i], t) for i, t in violation_idx]
	print('first k violations:')
	print(violation_idx[:20])

	num_vio = 0
	num_precondition = 0
	for row in stats:
		num_precondition += row[0]
		num_vio += row[1]
		print('{0}, {1}, {2:.4f}'.format(row[0], row[1], row[1]/row[0] if row[0] !=0 else 0))

	return num_vio/num_ex, num_vio/num_precondition


def main():
	parser = argparse.ArgumentParser(
		description =__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--log', help="The template of log to calculate confusion mat", default='')
	parser.add_argument('--num_seed', help="The number of seed", type=int, default=3)

	opt = parser.parse_args()

	all_global = []
	all_cond = []
	for i in range(opt.num_seed):
		g, c = process(opt.log + '_seed{0}'.format(i+1), [0, 2, 1])
		all_global.append(g)
		all_cond.append(c)

	print('all_global: {0}'.format(all_global))
	print('all_cond: {0}'.format(all_cond))

	print('avg global: {0}'.format(sum(all_global)/opt.num_seed))
	print('avg cond: {0}'.format(sum(all_cond)/opt.num_seed))


if __name__ == '__main__':
	main()