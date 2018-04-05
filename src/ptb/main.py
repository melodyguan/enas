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

from src.ptb.ptb_enas_child import PTBEnasChild
from src.ptb.ptb_enas_controller import PTBEnasController

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("search_for", None, "[rhn|base|enas]")

DEFINE_string("child_fixed_arc", None, "")
DEFINE_integer("batch_size", 25, "")
DEFINE_integer("child_base_number", 4, "")
DEFINE_integer("child_num_layers", 2, "")
DEFINE_integer("child_bptt_steps", 20, "")
DEFINE_integer("child_lstm_hidden_size", 200, "")
DEFINE_float("child_lstm_e_keep", 1.0, "")
DEFINE_float("child_lstm_x_keep", 1.0, "")
DEFINE_float("child_lstm_h_keep", 1.0, "")
DEFINE_float("child_lstm_o_keep", 1.0, "")
DEFINE_boolean("child_lstm_l_skip", False, "")
DEFINE_float("child_lr", 1.0, "")
DEFINE_float("child_lr_dec_rate", 0.5, "")
DEFINE_float("child_grad_bound", 5.0, "")
DEFINE_float("child_temperature", None, "")
DEFINE_float("child_l2_reg", None, "")
DEFINE_float("child_lr_dec_min", None, "")
DEFINE_float("child_optim_moving_average", None,
             "Use the moving average of Variables")
DEFINE_float("child_rnn_l2_reg", None, "")
DEFINE_float("child_rnn_slowness_reg", None, "")
DEFINE_float("child_lr_warmup_val", None, "")
DEFINE_float("child_reset_train_states", None, "")
DEFINE_integer("child_lr_dec_start", 4, "")
DEFINE_integer("child_lr_dec_every", 1, "")
DEFINE_integer("child_avg_pool_size", 1, "")
DEFINE_integer("child_block_size", 1, "")
DEFINE_integer("child_rhn_depth", 4, "")
DEFINE_integer("child_lr_warmup_steps", None, "")
DEFINE_string("child_optim_algo", "sgd", "")

DEFINE_boolean("child_sync_replicas", False, "")
DEFINE_integer("child_num_aggregate", 1, "")
DEFINE_integer("child_num_replicas", 1, "")

DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", None, "")
DEFINE_float("controller_skip_target", None, "")
DEFINE_float("controller_skip_rate", None, "")

DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after how many this number of epochs")
DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")

DEFINE_integer("num_epochs", 300, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")


def get_ops(x_train, x_valid, x_test):
  """Create relevant models."""

  ops = {}

  if FLAGS.search_for == "enas":
    assert FLAGS.child_lstm_hidden_size % FLAGS.child_block_size == 0, (
      "--child_block_size has to divide child_lstm_hidden_size")

    if FLAGS.child_fixed_arc is not None:
      assert not FLAGS.controller_training, (
        "with --child_fixed_arc, cannot train controller")

    child_model = PTBEnasChild(
      x_train,
      x_valid,
      x_test,
      rnn_l2_reg=FLAGS.child_rnn_l2_reg,
      rnn_slowness_reg=FLAGS.child_rnn_slowness_reg,
      rhn_depth=FLAGS.child_rhn_depth,
      fixed_arc=FLAGS.child_fixed_arc,
      batch_size=FLAGS.batch_size,
      bptt_steps=FLAGS.child_bptt_steps,
      lstm_num_layers=FLAGS.child_num_layers,
      lstm_hidden_size=FLAGS.child_lstm_hidden_size,
      lstm_e_keep=FLAGS.child_lstm_e_keep,
      lstm_x_keep=FLAGS.child_lstm_x_keep,
      lstm_h_keep=FLAGS.child_lstm_h_keep,
      lstm_o_keep=FLAGS.child_lstm_o_keep,
      lstm_l_skip=FLAGS.child_lstm_l_skip,
      vocab_size=10000,
      lr_init=FLAGS.child_lr,
      lr_dec_start=FLAGS.child_lr_dec_start,
      lr_dec_every=FLAGS.child_lr_dec_every,
      lr_dec_rate=FLAGS.child_lr_dec_rate,
      lr_dec_min=FLAGS.child_lr_dec_min,
      lr_warmup_val=FLAGS.child_lr_warmup_val,
      lr_warmup_steps=FLAGS.child_lr_warmup_steps,
      l2_reg=FLAGS.child_l2_reg,
      optim_moving_average=FLAGS.child_optim_moving_average,
      clip_mode="global",
      grad_bound=FLAGS.child_grad_bound,
      optim_algo="sgd",
      sync_replicas=FLAGS.child_sync_replicas,
      num_aggregate=FLAGS.child_num_aggregate,
      num_replicas=FLAGS.child_num_replicas,
      temperature=FLAGS.child_temperature,
      name="ptb_enas_model")

    if FLAGS.child_fixed_arc is None:
      controller_model = PTBEnasController(
        rhn_depth=FLAGS.child_rhn_depth,
        lstm_size=100,
        lstm_num_layers=1,
        lstm_keep_prob=1.0,
        tanh_constant=FLAGS.controller_tanh_constant,
        temperature=FLAGS.controller_temperature,
        lr_init=FLAGS.controller_lr,
        lr_dec_start=0,
        lr_dec_every=1000000,  # never decrease learning rate
        l2_reg=FLAGS.controller_l2_reg,
        entropy_weight=FLAGS.controller_entropy_weight,
        bl_dec=FLAGS.controller_bl_dec,
        optim_algo="adam",
        sync_replicas=FLAGS.controller_sync_replicas,
        num_aggregate=FLAGS.controller_num_aggregate,
        num_replicas=FLAGS.controller_num_replicas)

      child_model.connect_controller(controller_model)
      controller_model.build_trainer(child_model)

      controller_ops = {
        "train_step": controller_model.train_step,
        "loss": controller_model.loss,
        "train_op": controller_model.train_op,
        "lr": controller_model.lr,
        "grad_norm": controller_model.grad_norm,
        "valid_ppl": controller_model.valid_ppl,
        "optimizer": controller_model.optimizer,
        "baseline": controller_model.baseline,
        "ppl": controller_model.ppl,
        "reward": controller_model.reward,
        "entropy": controller_model.sample_entropy,
        "sample_arc": controller_model.sample_arc,
      }
    else:
      child_model.connect_controller(None)
      controller_ops = None
  else:
    raise ValueError("Unknown search_for {}".format(FLAGS.search_for))

  child_ops = {
    "global_step": child_model.global_step,
    "loss": child_model.loss,
    "train_op": child_model.train_op,
    "train_ppl": child_model.train_ppl,
    "train_reset": child_model.train_reset,
    "valid_reset": child_model.valid_reset,
    "test_reset": child_model.test_reset,
    "lr": child_model.lr,
    "grad_norm": child_model.grad_norm,
    "optimizer": child_model.optimizer,
  }

  ops = {
    "child": child_ops,
    "controller": controller_ops,
    "num_train_batches": child_model.num_train_batches,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
  }

  return ops


def train(mode="train"):
  assert mode in ["train", "eval"], "Unknown mode '{0}'".format(mode)

  with open(FLAGS.data_path) as finp:
    x_train, x_valid, x_test, _, _ = pickle.load(finp)
    print("-" * 80)
    print("train_size: {0}".format(np.size(x_train)))
    print("valid_size: {0}".format(np.size(x_valid)))
    print(" test_size: {0}".format(np.size(x_test)))

  g = tf.Graph()
  with g.as_default():
    ops = get_ops(x_train, x_valid, x_test)
    child_ops = ops["child"]
    controller_ops = ops["controller"]

    if FLAGS.child_optim_moving_average is None or mode == "eval":
      saver = tf.train.Saver(max_to_keep=10)
    else:
      saver = child_ops["optimizer"].swapping_saver(max_to_keep=10)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      FLAGS.output_dir, save_steps=ops["num_train_batches"], saver=saver)

    hooks = [checkpoint_saver_hook]
    if FLAGS.child_sync_replicas:
      sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
      hooks.append(sync_replicas_hook)
    if FLAGS.controller_training and FLAGS.controller_sync_replicas:
      hooks.append(controller_ops["optimizer"].make_session_run_hook(True))

    print("-" * 80)
    print("Starting session")
    with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
        start_time = time.time()

        if mode == "eval":
          sess.run(child_ops["valid_reset"])
          ops["eval_func"](sess, "valid", verbose=True)
          sess.run(child_ops["test_reset"])
          ops["eval_func"](sess, "test", verbose=True)
          sys.exit(0)

        num_batches = 0
        total_tr_ppl = 0
        best_valid_ppl = 67.00
        while True:
          run_ops = [
            child_ops["loss"],
            child_ops["lr"],
            child_ops["grad_norm"],
            child_ops["train_ppl"],
            child_ops["train_op"],
          ]
          loss, lr, gn, tr_ppl, _ = sess.run(run_ops)
          num_batches += 1
          total_tr_ppl += loss / FLAGS.child_bptt_steps
          global_step = sess.run(child_ops["global_step"])

          if FLAGS.child_sync_replicas:
            actual_step = global_step * FLAGS.num_aggregate
          else:
            actual_step = global_step
          epoch = actual_step // ops["num_train_batches"]
          curr_time = time.time()
          if global_step % FLAGS.log_every == 0:
            log_string = ""
            log_string += "epoch={:<6d}".format(epoch)
            log_string += " ch_step={:<6d}".format(global_step)
            log_string += " loss={:<8.4f}".format(loss)
            log_string += " lr={:<8.4f}".format(lr)
            log_string += " |g|={:<10.2f}".format(gn)
            log_string += " tr_ppl={:<8.2f}".format(
              np.exp(total_tr_ppl / num_batches))
            log_string += " mins={:<10.2f}".format(
                float(curr_time - start_time) / 60)
            print(log_string)

          if (FLAGS.child_reset_train_states is not None and
              np.random.uniform(0, 1) < FLAGS.child_reset_train_states):
            print("reset train states")
            sess.run([
              child_ops["train_reset"],
              child_ops["valid_reset"],
              child_ops["test_reset"],
            ])

          if actual_step % ops["eval_every"] == 0:
            sess.run([
              child_ops["train_reset"],
              child_ops["valid_reset"],
              child_ops["test_reset"],
            ])
            if (FLAGS.controller_training and
                epoch % FLAGS.controller_train_every == 0):
              sess.run([
                child_ops["train_reset"],
                child_ops["valid_reset"],
                child_ops["test_reset"],
              ])
              print("Epoch {}: Training controller".format(epoch))
              for ct_step in xrange(FLAGS.controller_train_steps *
                                    FLAGS.controller_num_aggregate):
                run_ops = [
                  controller_ops["loss"],
                  controller_ops["entropy"],
                  controller_ops["lr"],
                  controller_ops["grad_norm"],
                  controller_ops["reward"],
                  controller_ops["baseline"],
                  controller_ops["train_op"],
                ]
                loss, entropy, lr, gn, rw, bl, _ = sess.run(run_ops)
                controller_step = sess.run(controller_ops["train_step"])

                if ct_step % FLAGS.log_every == 0:
                  curr_time = time.time()
                  log_string = ""
                  log_string += "ctrl_step={:<6d}".format(controller_step)
                  log_string += " loss={:<7.3f}".format(loss)
                  log_string += " ent={:<5.2f}".format(entropy)
                  log_string += " lr={:<6.4f}".format(lr)
                  log_string += " |g|={:<10.7f}".format(gn)
                  log_string += " rw={:<7.3f}".format(rw)
                  log_string += " bl={:<7.3f}".format(bl)
                  log_string += " mins={:<.2f}".format(
                      float(curr_time - start_time) / 60)
                  print(log_string)

              print("Here are 10 architectures")
              for _ in xrange(10):
                arc, rw = sess.run([
                  controller_ops["sample_arc"],
                  controller_ops["reward"],
                ])
                print("{} rw={:<.3f}".format(np.reshape(arc, [-1]), rw))

            sess.run([
              child_ops["train_reset"],
              child_ops["valid_reset"],
              child_ops["test_reset"],
            ])
            print("Epoch {}: Eval".format(epoch))
            valid_ppl = ops["eval_func"](sess, "valid")
            if valid_ppl < best_valid_ppl:
              best_valid_ppl = valid_ppl
              sess.run(child_ops["test_reset"])
              ops["eval_func"](sess, "test", verbose=True)

            sess.run([
              child_ops["train_reset"],
              child_ops["valid_reset"],
              child_ops["test_reset"],
            ])
            total_tr_ppl = 0
            num_batches = 0

            print("-" * 80)

          if epoch >= FLAGS.num_epochs:
            ops["eval_func"](sess, "test", verbose=True)
            break


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


if __name__ == "__main__":
  tf.app.run()

