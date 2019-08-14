import ujson
import sys
import argparse
import re


def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n'))

# NOTE, this output original sentences without any tokenization
def extract(opt, csv_file):
	all_sent1 = []
	all_sent2 = []
	all_label = []
	all_sent1_pos = []
	all_sent2_pos = []
	all_sent1_lemma = []
	all_sent2_lemma = []
	max_sent_l = 0

	skip_cnt = 0

	with open(csv_file, 'r') as f:
		line_idx = 0
		for l in f:
			line_idx += 1
			if line_idx == 1 or l.strip() == '':
				continue

			cells = l.rstrip().split('\t')
			label = cells[0].strip()
			sent1 = cells[5].strip()
			sent2 = cells[6].strip()

			if label == '-':
				print('skipping label {0}'.format(label))
				skip_cnt += 1
				continue
			else:
				print(label)

			assert(label in ['entailment', 'neutral', 'contradiction'])


			all_sent1.append(sent1)
			all_sent2.append(sent2)
			all_label.append(label)

	print('skipped {0} examples'.format(skip_cnt))

	return (all_sent1, all_sent2, all_label)


def main(args):

	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	
	parser.add_argument('--data', help="Path to SNLI txt file", default="data/bert_nli/snli_1.0_dev.txt")
	parser.add_argument('--output', help="Prefix to the path of output", default="data/bert_nli/dev")
	opt = parser.parse_args(args)
	all_sent1, all_sent2, all_label = extract(opt, opt.data)
	print('{0} examples processed.'.format(len(all_sent1)))

	write_to(all_sent1, opt.output + '.raw.sent1.txt')
	write_to(all_sent2, opt.output + '.raw.sent2.txt')
	write_to(all_label, opt.output + '.label.txt')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


