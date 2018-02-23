from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"
_SPECIAL_TOKENS = [BOS, EOS, UNK]

def _load_vocab(vocab_filename):
  """
  Returns:
    vocab: list of words. the last word is UNK.
  """

  print("Loading vocab from '{}'".format(vocab_filename))
  with open(vocab_filename) as finp:
    lines = finp.read().split("\n")

  vocab = []
  for line in lines:
    line = line.strip()
    if line:
      tokens = line.split("\t")
      word, index = tokens[0].strip(), int(tokens[-1])
      if word not in _SPECIAL_TOKENS:
        assert tokens[-1].startswith("-"), (
          "Invalid index for word '{}'".format(word))
        vocab.append(word)

  vocab.extend(_SPECIAL_TOKENS)

  return vocab

def _create_textline_dataset(filename, vocab):
  print("Loading vocab from '{}'".format(filename))
  dataset = tf.data.TextLineDataset(filename)
  dataset = dataset.map(lambda string: tf.string_split([string]).values)

  unk_id = len(vocab) - 1
  table = tf.contrib.lookup.index_table_from_tensor(vocab, default_value=unk_id)
  dataset = dataset.map(lambda words: table.lookup(words))

  return dataset

def _batch_pad_shuffle(x_dataset, y_dataset, padding_id, batch_size=32):
  xy_dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
  xy_dataset = xy_dataset.padded_batch(
    batch_size,
    padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None])),
    padding_values=(tf.to_int64(padding_id), tf.to_int64(padding_id)))
  xy_dataset = xy_dataset.shuffle(1000)
  xy_iterator = xy_dataset.make_initializable_iterator()
  init_op = xy_iterator.initializer
  x, y = xy_iterator.get_next()

  return x, y, init_op


def load_data(data_path,
              source_lang,
              target_lang,
              train_prefix,
              valid_prefix,
              vocab_prefix,
              batch_size=32):

  print("-" * 80)
  print("Loading data from {}".format(data_path))
  with tf.device("/cpu:0"):
    with tf.name_scope("train_data"):
      # vocab
      x_vocab_file = os.path.join(
        data_path, "{}.{}".format(vocab_prefix, source_lang))
      x_vocab = _load_vocab(x_vocab_file)

      y_vocab_file = os.path.join(
        data_path, "{}.{}".format(vocab_prefix, target_lang))
      y_vocab = _load_vocab(y_vocab_file)

      # train dataset
      x_train_file = os.path.join(
        data_path, "{}.{}".format(train_prefix, source_lang))
      x_dataset = _create_textline_dataset(x_train_file, x_vocab)

      y_train_file = os.path.join(
        data_path, "{}.{}".format(train_prefix, target_lang))
      y_dataset = _create_textline_dataset(y_train_file, y_vocab)

      # batch, pad, shuffle
      eos_id = len(x_vocab) - 2
      x_train, y_train, train_init = _batch_pad_shuffle(
        x_dataset, y_dataset, eos_id, batch_size=batch_size)

      # valid dataset
      x_valid_file = os.path.join(
        data_path, "{}.{}".format(valid_prefix, source_lang))
      x_dataset = _create_textline_dataset(x_valid_file, x_vocab)

      y_valid_file = os.path.join(
        data_path, "{}.{}".format(valid_prefix, target_lang))
      y_dataset = _create_textline_dataset(y_valid_file, y_vocab)

      # batch, pad, shuffle
      x_valid, y_valid, valid_init = _batch_pad_shuffle(
        x_dataset, y_dataset, eos_id, batch_size=batch_size)

  return (x_train, y_train, train_init,
          x_valid, y_valid, valid_init,
          x_vocab, y_vocab)

