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

from src.cifar10.data_utils import read_data
from src.cifar10.general_controller import GeneralController
from src.cifar10.general_child import GeneralChild

from src.cifar10.micro_controller import MicroController
from src.cifar10.micro_child import MicroChild

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")
DEFINE_string("search_for", None, "Must be [macro|micro]")

DEFINE_integer("batch_size", 32, "")

DEFINE_integer("num_epochs", 300, "")
DEFINE_integer("child_lr_dec_every", 100, "")
DEFINE_integer("child_num_layers", 5, "")
DEFINE_integer("child_num_cells", 5, "")
DEFINE_integer("child_filter_size", 5, "")
DEFINE_integer("child_out_filters", 48, "")
DEFINE_integer("child_out_filters_scale", 1, "")
DEFINE_integer("child_num_branches", 4, "")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_block_size", 3, "")
DEFINE_integer("child_lr_T_0", None, "for lr schedule")
DEFINE_integer("child_lr_T_mul", None, "for lr schedule")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_lr", 0.1, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.5, "")
DEFINE_float("child_drop_path_keep_prob", 1.0, "minimum drop_path_keep_prob")
DEFINE_float("child_l2_reg", 1e-4, "")
DEFINE_float("child_lr_max", None, "for lr schedule")
DEFINE_float("child_lr_min", None, "for lr schedule")
DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")

DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_op_tanh_reduce", 1.0, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", None, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_search_whole_channels", False, "")
DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")
DEFINE_boolean("controller_use_critic", False, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

def get_ops(images, labels):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """

  assert FLAGS.search_for is not None, "Please specify --search_for"

  if FLAGS.search_for == "micro":
    ControllerClass = MicroController
    ChildClass = MicroChild
  else:
    ControllerClass = GeneralController
    ChildClass = GeneralChild

  child_model = ChildClass(
    images,
    labels,
    use_aux_heads=FLAGS.child_use_aux_heads,
    cutout_size=FLAGS.child_cutout_size,
    whole_channels=FLAGS.controller_search_whole_channels,
    num_layers=FLAGS.child_num_layers,
    num_cells=FLAGS.child_num_cells,
    num_branches=FLAGS.child_num_branches,
    fixed_arc=FLAGS.child_fixed_arc,
    out_filters_scale=FLAGS.child_out_filters_scale,
    out_filters=FLAGS.child_out_filters,
    keep_prob=FLAGS.child_keep_prob,
    drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
    num_epochs=FLAGS.num_epochs,
    l2_reg=FLAGS.child_l2_reg,
    data_format=FLAGS.data_format,
    batch_size=FLAGS.batch_size,
    clip_mode="norm",
    grad_bound=FLAGS.child_grad_bound,
    lr_init=FLAGS.child_lr,
    lr_dec_every=FLAGS.child_lr_dec_every,
    lr_dec_rate=FLAGS.child_lr_dec_rate,
    lr_cosine=FLAGS.child_lr_cosine,
    lr_max=FLAGS.child_lr_max,
    lr_min=FLAGS.child_lr_min,
    lr_T_0=FLAGS.child_lr_T_0,
    lr_T_mul=FLAGS.child_lr_T_mul,
    optim_algo="momentum",
    sync_replicas=FLAGS.child_sync_replicas,
    num_aggregate=FLAGS.child_num_aggregate,
    num_replicas=FLAGS.child_num_replicas,
  )

  if FLAGS.child_fixed_arc is None:
    controller_model = ControllerClass(
      search_for=FLAGS.search_for,
      search_whole_channels=FLAGS.controller_search_whole_channels,
      skip_target=FLAGS.controller_skip_target,
      skip_weight=FLAGS.controller_skip_weight,
      num_cells=FLAGS.child_num_cells,
      num_layers=FLAGS.child_num_layers,
      num_branches=FLAGS.child_num_branches,
      out_filters=FLAGS.child_out_filters,
      lstm_size=64,
      lstm_num_layers=1,
      lstm_keep_prob=1.0,
      tanh_constant=FLAGS.controller_tanh_constant,
      op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
      temperature=FLAGS.controller_temperature,
      lr_init=FLAGS.controller_lr,
      lr_dec_start=0,
      lr_dec_every=1000000,  # never decrease learning rate
      l2_reg=FLAGS.controller_l2_reg,
      entropy_weight=FLAGS.controller_entropy_weight,
      bl_dec=FLAGS.controller_bl_dec,
      use_critic=FLAGS.controller_use_critic,
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
      "valid_acc": controller_model.valid_acc,
      "optimizer": controller_model.optimizer,
      "baseline": controller_model.baseline,
      "entropy": controller_model.sample_entropy,
      "sample_arc": controller_model.sample_arc,
      "skip_rate": controller_model.skip_rate,
    }
  else:
    assert not FLAGS.controller_training, (
      "--child_fixed_arc is given, cannot train controller")
    child_model.connect_controller(None)
    controller_ops = None

  child_ops = {
    "global_step": child_model.global_step,
    "loss": child_model.loss,
    "train_op": child_model.train_op,
    "lr": child_model.lr,
    "grad_norm": child_model.grad_norm,
    "train_acc": child_model.train_acc,
    "optimizer": child_model.optimizer,
    "num_train_batches": child_model.num_train_batches,
  }

  ops = {
    "child": child_ops,
    "controller": controller_ops,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
    "num_train_batches": child_model.num_train_batches,
  }

  return ops


def train():
  if FLAGS.child_fixed_arc is None:
    images, labels = read_data(FLAGS.data_path)
  else:
    images, labels = read_data(FLAGS.data_path, num_valids=0)

  g = tf.Graph()
  with g.as_default():
    ops = get_ops(images, labels)
    child_ops = ops["child"]
    controller_ops = ops["controller"]

    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      FLAGS.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)

    hooks = [checkpoint_saver_hook]
    if FLAGS.child_sync_replicas:
      sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
      hooks.append(sync_replicas_hook)
    if FLAGS.controller_training and FLAGS.controller_sync_replicas:
      sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
      hooks.append(sync_replicas_hook)

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
        start_time = time.time()
        while True:
          run_ops = [
            child_ops["loss"],
            child_ops["lr"],
            child_ops["grad_norm"],
            child_ops["train_acc"],
            child_ops["train_op"],
          ]
          loss, lr, gn, tr_acc, _ = sess.run(run_ops)
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
            log_string += "ch_step={:<6d}".format(global_step)
            log_string += " loss={:<8.6f}".format(loss)
            log_string += " lr={:<8.4f}".format(lr)
            log_string += " |g|={:<8.4f}".format(gn)
            log_string += " tr_acc={:<3d}/{:>3d}".format(
                tr_acc, FLAGS.batch_size)
            log_string += " mins={:<10.2f}".format(
                float(curr_time - start_time) / 60)
            print(log_string)
            
          if actual_step % ops["eval_every"] == 0:
            if (FLAGS.controller_training and
                epoch % FLAGS.controller_train_every == 0):
              print("Epoch {}: Training controller".format(epoch))
              for ct_step in range(FLAGS.controller_train_steps *
                                    FLAGS.controller_num_aggregate):
                run_ops = [
                  controller_ops["loss"],
                  controller_ops["entropy"],
                  controller_ops["lr"],
                  controller_ops["grad_norm"],
                  controller_ops["valid_acc"],
                  controller_ops["baseline"],
                  controller_ops["skip_rate"],
                  controller_ops["train_op"],
                ]
                loss, entropy, lr, gn, val_acc, bl, skip, _ = sess.run(run_ops)
                controller_step = sess.run(controller_ops["train_step"])

                if ct_step % FLAGS.log_every == 0:
                  curr_time = time.time()
                  log_string = ""
                  log_string += "ctrl_step={:<6d}".format(controller_step)
                  log_string += " loss={:<7.3f}".format(loss)
                  log_string += " ent={:<5.2f}".format(entropy)
                  log_string += " lr={:<6.4f}".format(lr)
                  log_string += " |g|={:<8.4f}".format(gn)
                  log_string += " acc={:<6.4f}".format(val_acc)
                  log_string += " bl={:<5.2f}".format(bl)
                  log_string += " mins={:<.2f}".format(
                      float(curr_time - start_time) / 60)
                  print(log_string)

              print("Here are 10 architectures")
              for _ in range(10):
                arc, acc = sess.run([
                  controller_ops["sample_arc"],
                  controller_ops["valid_acc"],
                ])
                if FLAGS.search_for == "micro":
                  normal_arc, reduce_arc = arc
                  print(np.reshape(normal_arc, [-1]))
                  print(np.reshape(reduce_arc, [-1]))
                else:
                  start = 0
                  for layer_id in range(FLAGS.child_num_layers):
                    if FLAGS.controller_search_whole_channels:
                      end = start + 1 + layer_id
                    else:
                      end = start + 2 * FLAGS.child_num_branches + layer_id
                    print(np.reshape(arc[start: end], [-1]))
                    start = end
                print("val_acc={:<6.4f}".format(acc))
                print("-" * 80)

            print("Epoch {}: Eval".format(epoch))
            if FLAGS.child_fixed_arc is None:
              ops["eval_func"](sess, "valid")
            ops["eval_func"](sess, "test")

          if epoch >= FLAGS.num_epochs:
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
  train()


if __name__ == "__main__":
  tf.app.run()
