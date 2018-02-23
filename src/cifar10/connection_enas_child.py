import os
import sys

import numpy as np
import tensorflow as tf

from src.cifar10.models import Model
from src.cifar10.image_ops import conv
from src.cifar10.image_ops import fully_connected
from src.cifar10.image_ops import batch_norm
from src.cifar10.image_ops import batch_norm_with_mask
from src.cifar10.image_ops import relu
from src.cifar10.image_ops import max_pool
from src.cifar10.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from src.common_ops import create_weight


class ConnectionEnasChild(Model):
  def __init__(self,
               images,
               labels,
               fixed_arc=None,
               num_layers=2,
               filter_size=3,
               out_filters=24,
               keep_prob=1.0,
               batch_size=32,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=10000,
               lr_dec_rate=0.1,
               lr_cosine=False,
               lr_max=None,
               lr_min=None,
               lr_T_0=None,
               lr_T_mul=None,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NHWC",
               name="enas_child",
              ):
    """
    """

    super(self.__class__, self).__init__(
      images,
      labels,
      batch_size=batch_size,
      clip_mode=clip_mode,
      grad_bound=grad_bound,
      l2_reg=l2_reg,
      lr_init=lr_init,
      lr_dec_start=lr_dec_start,
      lr_dec_every=lr_dec_every,
      lr_dec_rate=lr_dec_rate,
      keep_prob=keep_prob,
      optim_algo=optim_algo,
      sync_replicas=sync_replicas,
      num_aggregate=num_aggregate,
      num_replicas=num_replicas,
      data_format=data_format,
      name=name)

    self.lr_cosine = lr_cosine
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul
    self.filter_size = filter_size
    self.out_filters = out_filters
    self.num_layers = num_layers

    self.pool_layers = [self.num_layers // 3 - 1, 2 * self.num_layers // 3 - 1]

    if fixed_arc is None:
      self.fixed_arc = fixed_arc
    else:
      print fixed_arc
      fixed_arc = np.array([int(_) for _ in fixed_arc.split(" ")],
                                dtype=np.int32)
      self.fixed_arc = np.array(fixed_arc, dtype=np.bool)

  def _model(self, images, is_training, arc_seq, reuse=None):
    if self.fixed_arc is None:
      layers = tf.TensorArray(tf.float32, size=self.num_layers - 1,
                              clear_after_read=False)
    else:
      layers = []
    with tf.variable_scope(self.name, reuse=reuse):
      x = images

      with tf.variable_scope("layer_0"):
        with tf.variable_scope("conv_1x1"):
          w = create_weight("w", [1, 1, 3, self.out_filters])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
          x = tf.nn.relu(x)

        with tf.variable_scope("conv_{}x{}".format(self.filter_size,
                                                   self.filter_size)):
          w = create_weight("w", [self.filter_size, self.filter_size,
                                  self.out_filters, self.out_filters])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
          x = tf.nn.relu(x)
          print x
        if self.fixed_arc is None:
          layers = layers.write(0, x)
        else:
          layers.append(x)

      with tf.variable_scope("layer_1"):
        with tf.variable_scope("conv_1x1"):
          w = create_weight("w", [1, 1, self.out_filters, self.out_filters])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
          x = tf.nn.relu(x)

        with tf.variable_scope("conv_{}x{}".format(self.filter_size,
                                                   self.filter_size)):
          w = create_weight("w", [self.filter_size, self.filter_size,
                                  self.out_filters, self.out_filters])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
          x = tf.nn.relu(x)
          print x
        if self.fixed_arc is None:
          layers = layers.write(1, x)
        else:
          layers.append(x)

      for layer_id in xrange(2, self.num_layers):
        with tf.variable_scope("layer_{}".format(layer_id)):
          if self.fixed_arc is None:
            start = (layer_id - 1) * (layer_id - 2) / 2
            end = start + layer_id - 1
            mask = arc_seq[start:end]
            mask = tf.concat([mask, tf.constant([True], dtype=tf.bool)], axis=0)
            x = self._one_layer(x, layer_id, images, layers, mask, is_training)
            if layer_id in self.pool_layers:
              x = max_pool(x, [2, 2], [2, 2], data_format=self.data_format,
                           keep_size=True)
            if layer_id < self.num_layers - 1:
              layers = layers.write(layer_id, x)
          else:
            start = (layer_id - 1) * (layer_id - 2) / 2
            end = start + layer_id - 1
            mask = self.fixed_arc[start:end]
            mask = np.concatenate([mask, [True]], axis=0)
            x = self._fixed_layer(layer_id, images, layers, mask, is_training)
            if layer_id in self.pool_layers:
              x = max_pool(x, [2, 2], [2, 2], data_format=self.data_format,
                           keep_size=False)
              pool_all_layers = []
              for layer in layers:
                pool = max_pool(layer, [2, 2], [2, 2], data_format=self.data_format)
                pool_all_layers.append(pool)
              layers = pool_all_layers
            if layer_id < self.num_layers - 1:
              layers.append(x)
          print x

      x = global_avg_pool(x, data_format=self.data_format)

      if is_training:
        x = tf.nn.dropout(x, self.keep_prob)
      with tf.variable_scope("fc"):
        w = create_weight("w", [self.out_filters, 10])
        x = tf.matmul(x, w)

    return x

  def _fixed_layer(self, layer_id, images, layers, mask, is_training):

    def _residual(L):
      inp_c = 0
      if self.data_format == "NHWC":
        for layer in L:
          inp_c += layer.get_shape()[3].value
        inp = tf.concat(L, axis=3)
      elif self.data_format == "NCHW":
        for layer in L:
          inp_c += layer.get_shape()[1].value
        inp = tf.concat(L, axis=1)
      else:
        raise ValueError("Unknown dafa_format {}".format(self.data_format))

      with tf.variable_scope("residual"):
        w = create_weight("w", [1, 1, inp_c, self.out_filters])
        out = tf.nn.conv2d(inp, w, [1, 1, 1, 1], "SAME",
        data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)
        out = tf.nn.relu(out)

      return out

    assert layer_id > 0, "only use this function for layer_id > 0"

    inputs = [layers[-1]]
    # with tf.variable_scope("conv_1"):
    #   with tf.variable_scope("conv_{}x{}".format(self.filter_size,
    #                                              self.filter_size)):
    #     w = create_weight("w", [self.filter_size, self.filter_size,
    #                             self.out_filters, self.out_filters])
    #     x = layers[-1]
    #     x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
    #     x = batch_norm(x, is_training, data_format=self.data_format)
    #     x = tf.nn.relu(x)
    #     inputs.append(x)

    with tf.variable_scope("conv_1"):
      with tf.variable_scope("conv_{}x{}".format(self.filter_size,
                                                 self.filter_size)):
        ch_mul = 4
        w_depth = create_weight("w_depth", [self.filter_size, self.filter_size,
                                self.out_filters, ch_mul])
        w_point = create_weight("w_point", [1, 1,
                                self.out_filters * ch_mul, self.out_filters])
        x = layers[-1]
        x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                   padding="SAME", data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        x = tf.nn.relu(x)
        inputs.append(x)

    prev_ = [layer for i, layer in enumerate(layers) if mask[i]] + inputs
    prev_.append(x)
    x = _residual(prev_)

    return x

  def _one_layer(self, inputs, layer_id, images, layers, mask, is_training):
    assert layer_id > 0, "only use this function for layer_id > 0"

    with tf.variable_scope("conv_{}x{}".format(self.filter_size,
                                               self.filter_size)):
      w = create_weight("w", [self.filter_size, self.filter_size,
                              self.out_filters, self.out_filters])
      x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      out = tf.nn.relu(x)

    inp_c = (layer_id + 1) * self.out_filters
    inp_n = tf.shape(images)[0]

    if self.data_format == "NHWC":
      inp_h = images.get_shape()[1].value
      inp_w = images.get_shape()[2].value

      indices = tf.boolean_mask(tf.range(0, layer_id, dtype=tf.int32), mask)
      x = layers.gather(indices)
      x = tf.transpose(x, [1, 2, 3, 0, 4])
      x = tf.reshape(x, [inp_n, inp_h, inp_w, -1])
      x = tf.concat([x, out], axis=3)
    elif self.data_format == "NCHW":
      inp_h = images.get_shape()[2].value
      inp_w = images.get_shape()[3].value

      indices = tf.boolean_mask(tf.range(0, layer_id, dtype=tf.int32), mask)
      x = layers.gather(indices)
      x = tf.transpose(x, [1, 0, 2, 3, 4])
      x = tf.reshape(x, [inp_n, -1, inp_h, inp_w])
      x = tf.concat([x, out], axis=1)
    else:
      raise ValueError("Unknown data_format {}".format(self.data_format))

    with tf.variable_scope("conv_1x1"):
      w_mask = tf.concat([mask, [True]], axis=0)
      w_mask = tf.reshape(w_mask, [layer_id + 1, 1])
      w_mask = tf.tile(w_mask, [1, self.out_filters])
      w_mask = tf.reshape(w_mask, [inp_c])
      w = create_weight("w", [inp_c, self.out_filters])
      w = tf.boolean_mask(w, w_mask)
      w = tf.reshape(w, [1, 1, -1, self.out_filters])
      x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      x = tf.nn.relu(x)

    return x

  # override
  def _build_train(self):
    print "Build train graph"
    logits = self._model(self.x_train, True, self.sample_arc)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.to_int32(self.train_preds)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.to_int32(self.train_acc)
    self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print "-" * 80
    for var in tf_variables:
      print var

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
      lr_cosine=self.lr_cosine,
      lr_max=self.lr_max,
      lr_min=self.lr_min,
      lr_T_0=self.lr_T_0,
      lr_T_mul=self.lr_T_mul,
      num_train_batches=self.num_train_batches,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

  # override
  def _build_valid(self):
    print "-" * 80
    print "Build valid graph"
    logits = self._model(self.x_valid, False, self.sample_arc, reuse=True)
    self.valid_preds = tf.argmax(logits, axis=1)
    self.valid_preds = tf.to_int32(self.valid_preds)
    self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
    self.valid_acc = tf.to_int32(self.valid_acc)
    self.valid_acc = tf.reduce_sum(self.valid_acc)

  # override
  def _build_test(self):
    print "-" * 80
    print "Build test graph"
    logits = self._model(self.x_test, False, self.sample_arc, reuse=True)
    self.test_preds = tf.argmax(logits, axis=1)
    self.test_preds = tf.to_int32(self.test_preds)
    self.test_acc = tf.equal(self.test_preds, self.y_test)
    self.test_acc = tf.to_int32(self.test_acc)
    self.test_acc = tf.reduce_sum(self.test_acc)

  # override
  def build_valid_rl(self, shuffle=False):
    print "-" * 80
    print "Build valid graph on shuffled data"
    with tf.device("/cpu:0"):
      # shuffled valid data: for choosing validation model
      if not shuffle and self.data_format == "NCHW":
        self.images["valid_original"] = np.transpose(
          self.images["valid_original"], [0, 3, 1, 2])
      x_valid_shuffle, y_valid_shuffle = tf.train.shuffle_batch(
        [self.images["valid_original"], self.labels["valid_original"]],
        batch_size=self.batch_size,
        capacity=25000,
        enqueue_many=True,
        min_after_dequeue=0,
        num_threads=16,
        seed=self.seed,
        allow_smaller_final_batch=True,
      )

      def _pre_process(x):
        x = tf.pad(x, [[4, 4], [4, 4], [0, 0]])
        x = tf.random_crop(x, [32, 32, 3], seed=self.seed)
        x = tf.image.random_flip_left_right(x, seed=self.seed)
        if self.data_format == "NCHW":
          x = tf.transpose(x, [2, 0, 1])

        return x

      if shuffle:
        x_valid_shuffle = tf.map_fn(_pre_process, x_valid_shuffle,
                                    back_prop=False)

    logits = self._model(x_valid_shuffle, False, self.sample_arc, reuse=True)
    valid_shuffle_preds = tf.argmax(logits, axis=1)
    valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      sample_arc = controller_model.sample_arc
      sample_arc = tf.cast(sample_arc, tf.bool)
      self.sample_arc = sample_arc
    else:
      self.sample_arc = None

    self._build_train()
    self._build_valid()
    self._build_test()
    print "Model has {} params".format(self.num_vars)

