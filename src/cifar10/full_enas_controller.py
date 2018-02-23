import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.controller import Controller
from src.utils import get_train_ops
from src.common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class FullEnasController(Controller):
  def __init__(self,
               num_layers_min=4,
               num_forwards_limit=1,
               num_types=8,
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
               bl_dec=0.999,
               optim_algo="adam",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               name="controller"):

    print "-" * 80
    print "Building ConvController"

    self.num_layers_min = num_layers_min
    self.num_forwards_limit = num_forwards_limit
    self.num_types = num_types
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
    self.bl_dec = bl_dec
    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.name = name

    self._create_params()
    self._build_sampler()

    #with tf.Session() as sess:
    #  sess.run(tf.global_variables_initializer())
    #  for _ in xrange(5):
    #    sess.run(self.grow_layer)
    #    run_ops = [
    #      self.layer_types,
    #      self.layer_skips,
    #      self.sample_log_probs,
    #      self.sample_entropys,
    #    ]
    #    layer_types, layer_skips, lp, ent = sess.run(run_ops)

    #    print "layer_types"
    #    print layer_types
    #    print "layer_skips"
    #    print layer_skips
    #    print "lp: {}".format(lp)
    #    print "ent: {}".format(ent)
    #sys.exit(0)

  def _create_params(self):
    with tf.variable_scope(self.name):
      with tf.variable_scope("lstm"):
        self.w_lstm = []
        for layer_id in xrange(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
            self.w_lstm.append(w)

      with tf.variable_scope("embedding"):
        self.g_emb = tf.get_variable("go", [1, self.lstm_size])
        self.w_emb = tf.get_variable("w", [self.num_types, self.lstm_size])

      with tf.variable_scope("softmax"):
        self.w_soft = tf.get_variable("w", [self.lstm_size, self.num_types])

      with tf.variable_scope("attention"):
        self.attn_w_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
        self.attn_w_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
        self.attn_v = tf.get_variable("v", [self.lstm_size, 1])

      with tf.variable_scope("curriculum"):
        self.num_layers = tf.Variable(self.num_layers_min, dtype=tf.int32,
                                      trainable=False, name="num_layers")
        self.grow_layer = tf.assign_add(self.num_layers, 1)

  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""

    sample_log_probs = []
    sample_entropy = []
    all_h = []
    all_h_w = []

    # sampler ops
    inputs = self.g_emb
    prev_c = [tf.zeros([1, self.lstm_size], dtype=tf.float32)
              for _ in xrange(self.lstm_num_layers)]
    prev_h = [tf.zeros([1, self.lstm_size], dtype=tf.float32)
              for _ in xrange(self.lstm_num_layers)]
    sample_log_probs = tf.TensorArray(tf.float32, 2 * self.num_layers,
                                      clear_after_read=False, infer_shape=False,
                                      name="log_probs")
    sample_entropys = tf.TensorArray(tf.float32, 2 * self.num_layers,
                                     clear_after_read=False, infer_shape=False,
                                     name="entropys")
    all_h = tf.TensorArray(tf.float32, self.num_layers, clear_after_read=False,
                           infer_shape=False, name="all_h")
    all_hw = tf.TensorArray(tf.float32, self.num_layers, clear_after_read=False,
                            infer_shape=False, name="all_hw")

    layer_types = tf.TensorArray(tf.int32, self.num_layers,
                                 clear_after_read=False, infer_shape=False)
    layer_skips = tf.TensorArray(tf.int32, self.num_layers,
                                 clear_after_read=False, infer_shape=False)

    def condition(layer_id, *args):
      return tf.less(layer_id, self.num_layers)

    def body(layer_id, inputs, prev_c, prev_h, all_h, all_hw, layer_types,
             layer_skips, sample_log_probs, sample_entropys):

      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)

      # layer type:
      logits = tf.matmul(next_h[-1], self.w_soft)
      if self.temperature is not None:
        logits /= self.temperature
      if self.tanh_constant is not None:
        logits = self.tanh_constant * tf.tanh(logits)

      layer_type = tf.multinomial(logits, 1)
      layer_type = tf.to_int32(layer_type)
      layer_type = tf.reshape(layer_type, [1])
      layer_types = layer_types.write(layer_id, layer_type)

      inputs = tf.nn.embedding_lookup(self.w_emb, layer_type)

      log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=layer_type)
      log_prob = tf.reduce_sum(log_prob)
      sample_log_probs = sample_log_probs.write(2 * layer_id, log_prob)

      entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
      sample_entropys = sample_entropys.write(2 * layer_id, entropy)

      # skip connections:
      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      all_h = all_h.write(layer_id, next_h[-1])
      all_hw = all_hw.write(layer_id, tf.matmul(next_h[-1], self.attn_w_1))

      def _sample_skip():
        tf_range = tf.range(0, layer_id - 1, dtype=tf.int32)
        attn_mem = all_hw.gather(tf_range)
        attn_mem = tf.reshape(attn_mem, [layer_id - 1, self.lstm_size])
        
        query = tf.matmul(next_h[-1], self.attn_w_2) + attn_mem
        query = tf.tanh(query)
        logits = tf.matmul(query, self.attn_v)

        if self.temperature is not None:
          logits /= self.temperature
        if self.tanh_constant is not None:
          logits = self.tanh_constant * tf.tanh(logits)

        logits = tf.concat([-logits, logits], axis=1)

        layer_skip = tf.multinomial(logits, 1)
        layer_skip = tf.reshape(layer_skip, [1, layer_id - 1])
        layer_skip = tf.to_float(layer_skip)
        _inputs = all_hw.gather(tf_range)
        _inputs = tf.reshape(_inputs, [layer_id - 1, self.lstm_size])
        _inputs = tf.matmul(layer_skip, _inputs)
        _inputs /= (1.0 + tf.reduce_sum(layer_skip))

        layer_skip = tf.to_int32(layer_skip)
        layer_skip = tf.reshape(layer_skip, [layer_id - 1])
        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=layer_skip)
        log_prob = tf.reduce_sum(log_prob)

        layer_skip = tf.concat([layer_skip, [1]], axis=0)
        layer_skip = tf.pad(layer_skip, [[0, self.num_layers - layer_id]])

        return _inputs, layer_skip, log_prob

      def _not_sample_skip(_inputs, _layer_id):
        layer_skip = tf.cond(
          tf.equal(_layer_id, 0),
          lambda: tf.zeros([self.num_layers], dtype=tf.int32),
          lambda: tf.concat(
            [[1], tf.zeros([self.num_layers - 1], dtype=tf.int32)], axis=0))
        log_prob = tf.constant(0, dtype=tf.float32)

        return _inputs, layer_skip, log_prob

      inputs, layer_skip, log_prob = tf.cond(tf.greater(layer_id, 1),
        lambda: _sample_skip(),
        lambda: _not_sample_skip(next_h[-1] / 2.0, layer_id))
      layer_skips = layer_skips.write(layer_id, layer_skip)

      sample_log_probs = sample_log_probs.write(2 * layer_id + 1, log_prob)

      entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
      sample_entropys = sample_entropys.write(2 * layer_id + 1, entropy)

      inputs.set_shape([1, self.lstm_size])

      return (layer_id + 1, inputs, next_c, next_h, all_h, all_hw, layer_types,
             layer_skips, sample_log_probs, sample_entropys)

    loop_vars = [
      tf.constant(0, dtype=tf.int32),
      inputs,
      prev_c,
      prev_h,
      all_h,
      all_hw,
      layer_types,
      layer_skips,
      sample_log_probs,
      sample_entropys,
    ]

    outputs = tf.while_loop(condition, body, loop_vars)

    layer_types = outputs[-4].stack()
    layer_skips = outputs[-3].stack()
    sample_log_probs = outputs[-2].stack()
    sample_entropys = outputs[-1].stack()

    self.layer_types = tf.reshape(layer_types, [self.num_layers])
    self.layer_skips = layer_skips
    self.sample_log_probs = tf.reduce_sum(sample_log_probs)
    self.sample_entropys = tf.reduce_sum(sample_entropys)

  def build_trainer(self, child_model):
    print "-" * 80
    print "Build controller trainner"

    # actor
    child_model.build_valid_rl()
    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    print "-" * 80
    for var in tf_variables:
      print var

    # baseline
    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    self.valid_acc = []
    self.loss = []
    self.train_step = []
    self.train_op = []
    self.lr = []
    self.grad_norm = []
    self.optimizer = []

    for acc in child_model.valid_shuffle_acc:
      reward = tf.to_float(acc) / tf.to_float(child_model.batch_size)
      reward = tf.stop_gradient(reward)
      self.valid_acc.append(reward)

      if self.entropy_weight is not None:
        reward += self.entropy_weight * self.sample_entropys

      baseline_update = tf.assign_sub(
        self.baseline, (1 - self.bl_dec) * (self.baseline - reward))
      with tf.control_dependencies([baseline_update]):
        reward = tf.identity(reward)

      loss = self.sample_log_probs * (reward - self.baseline)
      self.loss.append(loss)

      train_step = tf.Variable(0, dtype=tf.int32, trainable=False)
      self.train_step.append(train_step)

      train_op, lr, grad_norm, optimizer = get_train_ops(
        loss,
        tf_variables,
        train_step,
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

      self.train_op.append(train_op)
      self.lr.append(lr)
      self.grad_norm.append(grad_norm)
      self.optimizer.append(optimizer)

