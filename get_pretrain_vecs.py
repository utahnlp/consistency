
import numpy as np
import h5py
import re
import sys
import operator
import argparse

def load_glove_vec(fname, vocab):
  dim = 0
  word_vecs = {}
  word_vec_size = None
  for line in open(fname, 'r'):
    d = line.split()

    # get info from the first word
    if word_vec_size is None:
      word_vec_size = len(d) - 1

    word = ' '.join(d[:len(d)-word_vec_size])
    vec = d[-word_vec_size:]
    vec = np.array(list(map(float, vec)))
    dim = vec.size

    if len(d) - word_vec_size != 1:
      print('multi word token found: {0}...'.format(line[:30]))

    if word in vocab:
      word_vecs[word] = vec
  return word_vecs, dim


def load_default_glove(path):
  f = h5py.File(path, 'r')
  return f["word_vecs"][:].astype(np.float32)

def main():
  parser = argparse.ArgumentParser(
    description =__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--dict', help="The extracted token file (only tokens that are in the vocab)", default='data/snli.word.dict')
  parser.add_argument('--glove', help='The pretrained word vectors', default='')
  parser.add_argument('--output', help="output hdf5 file", default='data/glove')
  parser.add_argument('--default_glove', help="When this present, the result glove will be initialized from this", default='')
  parser.add_argument('--dim', help="The dimensionality of vectors", type=int, default=300)
  
  args = parser.parse_args()

  np.random.seed(3435)

  vocab = open(args.dict, "r").read().split("\n")[:-1]
  vocab = list(map(lambda x: (x.split()[0], int(x.split()[1])), vocab))
  word2idx = {x[0]: x[1] for x in vocab}

  
  if args.default_glove != '':
    print('initializing glove from {0}'.format(args.default_glove))
    default_glove = load_default_glove(args.default_glove)
    default_l = default_glove.shape[0]

    rs = np.random.normal(scale = 0.05, size = (max(len(vocab), default_l), args.dim))
    
    rs[:default_l, :] = default_glove
  else:
    rs = np.random.normal(scale = 0.05, size = (len(vocab), args.dim))

  print("vocab size: " + str(len(vocab)))
  w2v, dim = load_glove_vec(args.glove, word2idx)
  print("matched word vector size: {0}, dim: {1}".format(len(w2v), dim))

  # TODO, normalize or not???? 
  #for i in range(len(vocab)):
  # rs[i] = rs[i] / np.linalg.norm(rs[i])
    
  print("num words in pretrained model is " + str(len(w2v)))
  for word, vec in w2v.items():
    rs[word2idx[word]] = vec
  
  with h5py.File(args.output + '.hdf5', "w") as f:
    f["word_vecs"] = np.array(rs)
  
if __name__ == '__main__':
  main()