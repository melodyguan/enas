from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import numpy as np


def main():
  with open("train") as finp:
    lines = finp.read().strip().replace("\n", "<eos>")
    words = lines.split(" ")

  vocab, index = {}, {}
  for word in sorted(words):
    if word not in vocab:
      index[len(vocab)] = word
      vocab[word] = len(vocab)
  print("vocab size: {}".format(len(vocab)))

  x_train = [vocab[word] for word in words] + [vocab["<eos>"]]
  x_train = np.array(x_train, dtype=np.int32)

  with open("valid") as finp:
    lines = finp.read().strip().replace("\n", "<eos>")
    words = lines.split(" ")

  x_valid = [vocab[word] for word in words] + [vocab["<eos>"]]
  x_valid = np.array(x_valid, dtype=np.int32)

  with open("test") as finp:
    lines = finp.read().strip().replace("\n", "<eos>")
    words = lines.split(" ")

  x_test = [vocab[word] for word in words] + [vocab["<eos>"]]
  x_test = np.array(x_test, dtype=np.int32)

  print("train size: {}".format(np.size(x_train)))
  print("valid size: {}".format(np.size(x_valid)))
  print("test size: {}".format(np.size(x_test)))

  with open("ptb.pkl", "w") as fout:
    pickle.dump((x_train, x_valid, x_test, vocab, index), fout, protocol=2)


if __name__ == "__main__":
  main()
