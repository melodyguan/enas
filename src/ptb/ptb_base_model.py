import os
import sys

import numpy as np
import tensorflow as tf

from src.common_ops import lstm

from src.utils import count_model_params
from src.utils import get_train_ops

from src.ptb.data_utils import ptb_input_producer

class PTBBaseModel(object):
  def __init__(self,
               x_train,
               x_valid,
               x_test,
               batch_size=32,
               bptt_steps=25,
               lstm_num_layers=2,
               lstm_hidden_size=32,
               lstm_e_keep=1.0,
               lstm_x_keep=1.0,
               lstm_h_keep=1.0,
               lstm_o_keep=1.0,
               lstm_l_skip=False,
               vocab_size=10000,
               lr_init=1.0,
               lr_dec_start=4,
               lr_dec_every=1,
               lr_dec_rate=0.5,
               l2_reg=None,
               clip_mode="global",
               grad_bound=5.0,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               name="ptb_lstm",
               seed=None,
              ):
    """
    Args:
      lr_dec_every: number of epochs to decay
    """
    print "-" * 80
    print "Build model {}".format(name)

    self.batch_size = batch_size
    self.bptt_steps = bptt_steps
    self.lstm_num_layers = lstm_num_layers
    self.lstm_hidden_size = lstm_hidden_size
    self.lstm_e_keep = lstm_e_keep
    self.lstm_x_keep = lstm_x_keep
    self.lstm_h_keep = lstm_h_keep
    self.lstm_o_keep = lstm_o_keep
    self.lstm_l_skip = lstm_l_skip
    self.vocab_size = vocab_size
    self.lr_init = lr_init
    self.l2_reg = l2_reg
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound

    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas

    self.name = name
    self.seed = seed
    
    self.global_step = None
    self.valid_loss = None
    self.test_loss = None

    print "Build data ops"
    # training data
    self.x_train, self.y_train, self.num_train_batches = ptb_input_producer(
      x_train, self.batch_size, self.bptt_steps)
    self.y_train = tf.reshape(self.y_train, [self.batch_size * self.bptt_steps])

    self.lr_dec_start = lr_dec_start * self.num_train_batches
    self.lr_dec_every = lr_dec_every * self.num_train_batches
    self.lr_dec_rate = lr_dec_rate

    # valid data
    self.x_valid, self.y_valid, self.num_valid_batches = ptb_input_producer(
      x_valid, self.batch_size, self.bptt_steps)
    self.y_valid = tf.reshape(self.y_valid, [self.batch_size * self.bptt_steps])

    # test data
    self.x_test, self.y_test, self.num_test_batches = ptb_input_producer(
      x_test, 1, 1)
    self.y_test = tf.reshape(self.y_test, [1])

    self.x_valid_raw = x_valid

    self._build_params()
    self._build_train()
    self._build_valid()
    self._build_test()

  def eval_once(self, sess, eval_set, feed_dict=None, verbose=False):
    """Expects self.acc and self.global_step to be defined.

    Args:
      sess: tf.Session() or one of its wrap arounds.
      feed_dict: can be used to give more information to sess.run().
      eval_set: "valid" or "test"
    """

    assert self.global_step is not None, "TF op self.global_step not defined."
    global_step = sess.run(self.global_step)
    print "Eval at {}".format(global_step)
   
    if eval_set == "valid":
      assert self.valid_loss is not None, "TF op self.valid_loss is not defined."
      num_batches = self.num_valid_batches
      loss_op = self.valid_loss
      reset_op = self.valid_reset
      batch_size = self.batch_size
      bptt_steps = self.bptt_steps
    elif eval_set == "test":
      assert self.test_loss is not None, "TF op self.test_loss is not defined."
      num_batches = self.num_test_batches
      loss_op = self.test_loss
      reset_op = self.test_reset
      batch_size = 1
      bptt_steps = 1
    else:
      raise ValueError("Unknown eval_set '{}'".format(eval_set))

    sess.run(reset_op)
    total_loss = 0
    for batch_id in xrange(num_batches):
      total_loss += sess.run(loss_op, feed_dict=feed_dict)
      ppl_sofar = np.exp(
        total_loss / (bptt_steps * batch_size * (batch_id + 1)))
      if verbose and (batch_id + 1) % 1000 == 0:
        print "{:<5d} {:<6.2f}".format(batch_id + 1, ppl_sofar)
    if verbose:
      print ""
    ppl = np.exp(total_loss / (num_batches * batch_size * bptt_steps))
    print "{}_ppl: {:<6.2f}".format(eval_set, ppl)

  def _build_train(self):
    print "Build train graph"
    logits, self.train_reset = self._model(self.x_train, True, False)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = tf.reduce_sum(log_probs) / tf.to_float(self.batch_size)
    self.train_ppl = tf.exp(tf.reduce_mean(log_probs))

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print "-" * 80
    for var in tf_variables:
      print var
    print "Model has {} parameters".format(self.num_vars)

    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")
    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.global_step,
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

  def _build_valid(self):
    print "-" * 80
    print "Build valid graph"
    logits, self.valid_reset = self._model(self.x_valid, False, False)
    logits = tf.stop_gradient(logits)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_valid)
    self.valid_loss = tf.reduce_sum(log_probs)

  def _build_test(self):
    print "-" * 80
    print "Build test graph"
    logits, self.test_reset = self._model(self.x_test, False, True)
    logits = tf.stop_gradient(logits)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_test)
    self.test_loss = tf.reduce_sum(log_probs)

  def build_valid_rl(self, shuffle=False):
    print "-" * 80
    print "Build valid graph on shuffled data"

  def _model(self, x, is_training, is_test):
    if is_test:
      start_c, start_h = self.test_start_c, self.test_start_h
      num_steps = 1
      batch_size = 1
    else:
      start_c, start_h = self.start_c, self.start_h
      num_steps = self.bptt_steps
      batch_size = self.batch_size

    all_h = tf.TensorArray(tf.float32, size=num_steps, infer_shape=True)
    embedding = tf.nn.embedding_lookup(self.w_emb, x)

    if is_training:
      def _gen_mask(shape, keep_prob):
        _mask = tf.random_uniform(shape, dtype=tf.float32)
        _mask = tf.floor(_mask + keep_prob) / keep_prob
        return _mask

      # variational dropout in the embedding layer
      e_mask = _gen_mask([batch_size, num_steps], self.lstm_e_keep)
      first_e_mask = e_mask
      zeros = tf.zeros_like(e_mask)
      ones = tf.ones_like(e_mask)
      r = [tf.constant([[False]] * batch_size, dtype=tf.bool)]  # more zeros to e_mask
      for step in xrange(1, num_steps):
        should_zero = tf.logical_and(
          tf.equal(x[:, :step], x[:, step:step+1]),
          tf.equal(e_mask[:, :step], 0))
        should_zero = tf.reduce_any(should_zero, axis=1, keep_dims=True)
        r.append(should_zero)
      r = tf.concat(r, axis=1)
      e_mask = tf.where(r, tf.zeros_like(e_mask), e_mask)
      e_mask = tf.reshape(e_mask, [batch_size, num_steps, 1])
      embedding *= e_mask

      # variational dropout in the hidden layers
      x_mask, h_mask = [], []
      for layer_id in xrange(self.lstm_num_layers):
        x_mask.append(
          _gen_mask([batch_size, self.lstm_hidden_size], self.lstm_x_keep))
        h_mask.append(
          _gen_mask([batch_size, self.lstm_hidden_size], self.lstm_h_keep))

      # variational dropout in the output layer
      o_mask = _gen_mask([batch_size, self.lstm_hidden_size], self.lstm_o_keep)

    def condition(step, *args):
      return tf.less(step, num_steps)

    def body(step, prev_c, prev_h, all_h):
      next_c, next_h = [], []
      for layer_id, (p_c, p_h, w) in enumerate(zip(prev_c, prev_h, self.w_lstm)):
        if layer_id == 0:
          inputs = embedding[:, step, :]
        else:
          inputs = next_h[-1]

        if is_training:
          inputs *= x_mask[layer_id]
          p_h *= h_mask[layer_id]

        ifog = tf.matmul(tf.concat([inputs, p_h], axis=1), w)
        i, f, o, g = tf.split(ifog, 4, axis=1)

        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        g =    tf.tanh(g)

        curr_c = i * g + f * p_c
        curr_h = o * tf.tanh(curr_c)

        if self.lstm_l_skip:
          curr_h += inputs

        next_c.append(curr_c)
        next_h.append(curr_h)

      out_h = next_h[-1]
      if is_training:
        out_h *= o_mask
      all_h = all_h.write(step, out_h)
      return step + 1, next_c, next_h, all_h
    
    loop_vars = [
      tf.constant(0, dtype=tf.int32), start_c, start_h, all_h
    ]
    loop_outputs = tf.while_loop(condition, body, loop_vars, back_prop=True)
    next_c = loop_outputs[-3]
    next_h = loop_outputs[-2]
    all_h = loop_outputs[-1].stack()
    all_h = tf.transpose(all_h, [1, 0, 2])
    all_h = tf.reshape(all_h, [batch_size * num_steps, self.lstm_hidden_size])
    
    carry_states = []
    reset_states = []
    for layer_id, (s_c, s_h, n_c, n_h) in enumerate(
        zip(start_c, start_h, next_c, next_h)):
      reset_states.append(tf.assign(s_c, tf.zeros_like(s_c), use_locking=True))
      reset_states.append(tf.assign(s_h, tf.zeros_like(s_h), use_locking=True))
      carry_states.append(tf.assign(s_c, tf.stop_gradient(n_c), use_locking=True))
      carry_states.append(tf.assign(s_h, tf.stop_gradient(n_h), use_locking=True))

    logits = tf.matmul(all_h, self.w_emb, transpose_b=True)

    with tf.control_dependencies(carry_states):
      logits = tf.identity(logits)

    return logits, reset_states

  def _build_params(self):
    # initializer = tf.contrib.keras.initializers.he_normal(self.seed)
    initializer = tf.random_uniform_initializer(minval=-0.04, maxval=0.04)
    with tf.variable_scope(self.name, initializer=initializer):
      with tf.variable_scope("variational_lstm"):
        self.w_lstm = []
        for layer_id in xrange(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w = tf.get_variable("w", [2 * self.lstm_hidden_size, 4 * self.lstm_hidden_size])
            self.w_lstm.append(w)

      with tf.variable_scope("embedding"):
        self.w_emb = tf.get_variable(
          "w", [self.vocab_size, self.lstm_hidden_size])

      with tf.variable_scope("starting_states"):
        zeros = np.zeros([self.batch_size, self.lstm_hidden_size],
                         dtype=np.float32)
        self.start_c = [tf.Variable(zeros, trainable=False) for _ in
                        xrange(self.lstm_num_layers)]
        self.start_h = [tf.Variable(zeros, trainable=False) for _ in
                        xrange(self.lstm_num_layers)]

        self.test_start_c = [
          tf.Variable(np.zeros([1, self.lstm_hidden_size], dtype=np.float32),
                      trainable=False) for _ in xrange(self.lstm_num_layers)]

        self.test_start_h = [
          tf.Variable(np.zeros([1, self.lstm_hidden_size], dtype=np.float32),
                      trainable=False) for _ in xrange(self.lstm_num_layers)]

