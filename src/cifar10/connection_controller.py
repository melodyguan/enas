import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.controller import Controller
from src.utils import get_train_ops
from src.common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class ConnectionController(Controller):
  def __init__(self,
               num_layers=4,
               num_forwards_limit=1,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               tanh_constant=None,
               temperature=None,
               lr_init=1e-3,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               l2_reg=0,
               entropy_weight=None,
               clip_mode=None,
               grad_bound=None,
               use_critic=False,
               bl_dec=0.999,
               optim_algo="adam",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               skip_target=0.8,
               skip_rate=0.5,
               name="controller"):

    print "-" * 80
    print "Building ConvController"

    self.num_layers = num_layers
    self.num_forwards_limit = num_forwards_limit
    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers 
    self.lstm_keep_prob = lstm_keep_prob
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.l2_reg = l2_reg
    self.entropy_weight = entropy_weight
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.use_critic = use_critic
    self.bl_dec = bl_dec

    self.skip_target = skip_target
    self.skip_rate = skip_rate

    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.name = name

    self._create_params()
    self._build_sampler()

  def _create_params(self):
    with tf.variable_scope(self.name):
      with tf.variable_scope("lstm"):
        self.w_lstm = []
        for layer_id in xrange(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
            self.w_lstm.append(w)

      with tf.variable_scope("embedding"):
        self.g_emb_0 = tf.get_variable("g_emb_0", [1, self.lstm_size])
        self.g_emb_1 = tf.get_variable("g_emb_1", [1, self.lstm_size])
        self.g_emb_2 = tf.get_variable("g_emb_2", [1, self.lstm_size])

      with tf.variable_scope("attention"):
        self.attn_w_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
        self.attn_w_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
        self.attn_v = tf.get_variable("v", [self.lstm_size, 1])

      with tf.variable_scope("critic"):
        self.w_critic = tf.get_variable("w", [self.lstm_size, 1])

  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""

    arc_seq = []
    sample_log_probs = []
    sample_entropy = []
    all_h = []
    all_h_w = []
    skip_penaltys = []

    # sampler ops
    inputs = self.g_emb_0
    prev_c = [tf.zeros([1, self.lstm_size], dtype=tf.float32)
              for _ in xrange(self.lstm_num_layers)]
    prev_h = [tf.zeros([1, self.lstm_size], dtype=tf.float32)
              for _ in xrange(self.lstm_num_layers)]
    skip_targets = tf.constant([self.skip_target, 1.0 - self.skip_target], dtype=tf.float32)
    for layer_id in xrange(self.num_layers):
      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      all_h.append(next_h[-1])
      all_h_w.append(tf.matmul(next_h[-1], self.attn_w_1))

      if layer_id > 1:
        query = tf.matmul(next_h[-1], self.attn_w_2)
        query = query + tf.concat(all_h_w[:-2], axis=0)
        query = tf.tanh(query)
        query = tf.matmul(query, self.attn_v)

        if self.temperature is not None:
          query /= self.temperature
        if self.tanh_constant is not None:
          query = self.tanh_constant * tf.tanh(query)

        logits = tf.concat([-query, query], axis=1)

        skip_probs = tf.sigmoid(logits)
        kl = skip_probs * tf.log(skip_probs / skip_targets)
        skip_penaltys.append(tf.reduce_sum(kl))

        config_id = tf.multinomial(logits, 1)
        config_id = tf.to_int32(config_id)
        config_id = tf.reshape(config_id, [layer_id - 1])
        arc_seq.append(config_id)

        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=config_id)
        sample_log_probs.append(log_prob)

        entropy = log_prob * tf.exp(-log_prob)
        sample_entropy.append(tf.stop_gradient(entropy))

        config_id = tf.reshape(config_id, [1, layer_id - 1])
        config_id = tf.to_float(config_id)
        inputs = tf.matmul(config_id, tf.concat(all_h[:-2], axis=0))
        inputs /= (1.0 + tf.reduce_sum(config_id))
      elif layer_id == 1:
        inputs = self.g_emb_2
      else:
        inputs = self.g_emb_1

    arc_seq = tf.concat(arc_seq, axis=0)
    self.sample_arc = arc_seq

    self.sample_log_probs = tf.concat(sample_log_probs, axis=0)
    self.ppl = tf.exp(tf.reduce_mean(self.sample_log_probs))

    sample_entropy = tf.concat(sample_entropy, axis=0)
    self.sample_entropy = tf.reduce_sum(sample_entropy)

    skip_penaltys = tf.stack(skip_penaltys)
    self.skip_penaltys = tf.reduce_sum(skip_penaltys)

    self.all_h = all_h

  def build_trainer(self, child_model):
    # actor
    child_model.build_valid_rl()
    self.valid_acc = (tf.to_float(child_model.valid_shuffle_acc) /
                      tf.to_float(child_model.batch_size))
    self.reward = self.valid_acc

    skip_rate = tf.reduce_sum(tf.stop_gradient(self.sample_arc))
    normalize = (self.num_layers - 2) * (self.num_layers - 1) / 2
    self.skip_rate = tf.to_float(skip_rate) / tf.to_float(normalize)

    if self.entropy_weight is not None:
      self.reward += self.entropy_weight * self.sample_entropy

    if self.use_critic:
      # critic
      all_h = tf.concat(self.all_h, axis=0)
      all_h = tf.stop_gradient(all_h)
      value_function = tf.matmul(all_h, self.w_critic)
      advantage = value_function - self.reward
      critic_loss = tf.reduce_sum(advantage ** 2)
      self.baseline = tf.reduce_mean(value_function)
      self.loss = -tf.reduce_sum(self.sample_log_probs * advantage)

      critic_train_step = tf.Variable(
          0, dtype=tf.int32, trainable=False, name="critic_train_step")
      critic_train_op, _, _, _ = get_train_ops(
        critic_loss,
        [self.w_critic],
        critic_train_step,
        clip_mode=self.clip_mode,
        lr_init=5e-3,
        lr_dec_start=0,
        lr_dec_every=int(1e9),
        optim_algo="adam",
        sync_replicas=False)
    else:
      # or baseline
      self.sample_log_probs = tf.reduce_sum(self.sample_log_probs)
      self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
      baseline_update = tf.assign_sub(
        self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))

      with tf.control_dependencies([baseline_update]):
        self.reward = tf.identity(self.reward)
      self.loss = self.sample_log_probs * (self.reward - self.baseline)

    self.train_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="train_step")
    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    print "-" * 80
    for var in tf_variables:
      print var

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss + self.skip_rate * self.skip_penaltys,
      tf_variables,
      self.train_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

    if self.use_critic:
      self.train_op = tf.group(self.train_op, critic_train_op)

