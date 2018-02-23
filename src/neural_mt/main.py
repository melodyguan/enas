from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle as pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.neural_mt.data_utils import load_data

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("search_for", None, "[rhn|base|enas]")

DEFINE_string("source_lang", None, "Source language")
DEFINE_string("target_lang", None, "Target language")
DEFINE_string("train_prefix", None, "Train file prefix")
DEFINE_string("valid_prefix", None, "Valid file prefix")
DEFINE_string("vocab_prefix", None, "Vocab file prefix")

DEFINE_boolean("controller_training", False, "")
DEFINE_boolean("controller_sync_replicas", False, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

def get_ops():
  """Create relevant models."""

  (x_train, y_train, train_init,
   x_valid, y_valid, valid_init,
   x_vocab, y_vocab) = load_data(
    FLAGS.data_path,
    FLAGS.source_lang,
    FLAGS.target_lang,
    FLAGS.train_prefix,
    FLAGS.valid_prefix,
    FLAGS.vocab_prefix)

  global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
  ops = {}
  child_ops = {}
  controller_ops = {}

  ops = {
    "child": child_ops,
    "controller": controller_ops,
    "train_init": train_init,
    "valid_init": valid_init,
  }

  return ops


def train(mode="train"):
  assert mode in ["train", "eval"], "Unknown mode '{}'".format(mode)

  g = tf.Graph()
  with g.as_default():
    ops = get_ops()
    child_ops = ops["child"]
    controller_ops = ops["controller"]

    saver = tf.train.Saver(max_to_keep=10)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      FLAGS.output_dir, save_steps=1000, saver=saver)

    hooks = [checkpoint_saver_hook]
    if FLAGS.controller_training and FLAGS.controller_sync_replicas:
      sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
      hooks.append(sync_replicas_hook)

    print("-" * 80)
    print("Starting session")
    train_sess = tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.output_dir)

    start_time = time.time()
    train_sess.run(ops["train_init"])
    train_sess.run(ops["valid_init"])

    print("DONE")
    sys.exit(0)


def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  utils.print_user_flags()
  train(mode="train")
  train(mode="eval")


if __name__ == "__main__":
  tf.app.run()

