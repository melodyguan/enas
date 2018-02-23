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


class MicroChild(Model):
  def __init__(self,
               images,
               labels,
               cutout_size=None,
               whole_channels=False,
               fixed_arc=None,
               out_filters_scale=1,
               num_layers=2,
               num_cell_layers=5,
               num_branches=6,
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
               name="child",
              ):
    """
    """

    super(self.__class__, self).__init__(
      images,
      labels,
      cutout_size=cutout_size,
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

    self.whole_channels = whole_channels
    self.lr_cosine = lr_cosine
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul
    self.out_filters = out_filters * out_filters_scale
    self.num_layers = num_layers
    self.num_cell_layers = num_cell_layers

    self.num_branches = num_branches
    self.fixed_arc = fixed_arc
    self.out_filters_scale = out_filters_scale

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

    self._create_params()

  def _model(self, images, is_training, reuse=False):
    with tf.variable_scope(self.name, reuse=reuse):
      layers = []
      with tf.variable_scope("conv_0"):
        w = create_weight("w", [1, 1, 3, self.out_filters])
        x = tf.nn.conv2d(images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        x = tf.nn.relu(x)
        layers.append(x)

      with tf.variable_scope("conv_1"):
        w = create_weight("w", [1, 1, 3, self.out_filters])
        x = tf.nn.conv2d(images, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        x = tf.nn.relu(x)
        layers.append(x)

      def condition(layer_id, *args):
        return tf.less(layer_id, self.num_layers)

      def body(layer_id, prev_layers):
        if self.fixed_arc is None:
          x = self._enas_layer(layer_id, prev_layers, is_training)
        else:
          x = self._fixed_layer(layer_id, prev_layers, is_training)
        prev_layers = [prev_layers[1], x]
        return layer_id + 1, prev_layers

      loop_vars = [tf.constant(0, dtype=tf.int32), layers]
      loop_outputs = tf.while_loop(condition, body, loop_vars)
      x = loop_outputs[-1][1]

      x = global_avg_pool(x, data_format=self.data_format)
      if is_training:
        x = tf.nn.dropout(x, self.keep_prob)
      with tf.variable_scope("fc"):
        w = create_weight("w", [self.out_filters, 10])
        x = tf.matmul(x, w)
    return x

  def _create_params(self):
    with tf.variable_scope(self.name):
      if self.fixed_arc is None:
        with tf.variable_scope("conv_3x3"):
          self.w_3x3 = create_weight("w", [self.num_layers,
                                           self.num_cell_layers, 3, 3,
                                           self.out_filters, self.out_filters])

        with tf.variable_scope("conv_1x5_5x1"):
          self.w_1x5 = create_weight("w_1_5", [self.num_layers,
                                               self.num_cell_layers, 1, 5,
                                               self.out_filters,
                                               self.out_filters])
          self.w_5x1 = create_weight("w_5_1", [self.num_layers,
                                               self.num_cell_layers, 5, 1,
                                               self.out_filters,
                                               self.out_filters])

        with tf.variable_scope("sep_conv_5x5"):
          ch_mul = 1
          self.w_depth = create_weight("w_depth", [self.num_layers,
                                                   self.num_cell_layers, 5, 5,
                                                   self.out_filters, ch_mul])
          self.w_point = create_weight("w_point", [self.num_layers,
                                                   self.num_cell_layers, 1, 1,
                                                   self.out_filters * ch_mul,
                                                   self.out_filters])

        with tf.variable_scope("bn"):
          self.moving_mean = tf.get_variable(
            "moving_mean", [self.num_layers, self.num_cell_layers, self.out_filters],
            trainable=False, initializer=tf.constant_initializer(0.0, dtype=tf.float32))

          self.moving_variance = tf.get_variable(
            "moving_variance", [self.num_layers, self.num_cell_layers, self.out_filters],
            trainable=False, initializer=tf.constant_initializer(0.0, dtype=tf.float32))

          self.offset = tf.get_variable(
            "offset", [self.num_layers, self.num_cell_layers, self.out_filters],
            initializer=tf.constant_initializer(0.0, dtype=tf.float32))

          self.scale = tf.get_variable(
            "scale", [self.num_layers, self.num_cell_layers, self.out_filters],
            initializer=tf.constant_initializer(1.0, dtype=tf.float32))

        def _batch_norm(x, layer_id, cell_id, is_training, epsilon=1e-3, decay=0.9):
          scale = self.scale[layer_id, cell_id]
          offset = self.offset[layer_id, cell_id]
          moving_mean = self.moving_mean[layer_id, cell_id]
          moving_variance = self.moving_variance[layer_id, cell_id]
          if is_training:
            x, mean, variance = tf.nn.fused_batch_norm(
              x, scale, offset, epsilon=epsilon, data_format=self.data_format,
              is_training=True)
            mean = (1.0 - decay) * (moving_mean - mean)
            variance = (1.0 - decay) * (moving_variance - variance)
            indices = [layer_id, cell_id]
            update_mean = tf.scatter_sub(moving_mean, indices, mean, use_locking=True)
            update_variance = tf.scatter_sub(moving_variance, indices, variance, use_locking=True)
            with tf.control_dependencies([update_mean, update_variance]):
              x = tf.identity(x)
          else:
            x, _, _ = tf.nn.fused_batch_norm(x, scale, offset,
                                             mean=self.moving_mean[layer_id, cell_id],
                                             variance=self.moving_variance[layer_id, cell_id],
                                             epsilon=epsilon, data_format=self.data_format,
                                             is_training=False)
          return x

        def _conv_3x3(x, layer_id, cell_id, is_training):
          w = self.w_3x3[layer_id, cell_id]
          x = tf.nn.conv2d(x, w_1x5, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          x = tf.nn.conv2d(x, w_5x1, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          x = _batch_norm(x, layer_id, cell_id, is_training)
          x = tf.nn.relu(x)
          return x

        def _conv_1x5_5x1(x, layer_id, cell_id, is_training):
          w_1x5 = self.w_1x5[layer_id, cell_id]
          w_5x1 = self.w_5x1[layer_id, cell_id]
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          x = _batch_norm(x, layer_id, cell_id, is_training)
          x = tf.nn.relu(x)
          return x

        def _sep_conv_5x5(x, layer_id, cell_id, is_training):
          w_depth = self.w_depth[layer_id, cell_id]
          w_point = self.w_point[layer_id, cell_id]
          x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                     padding="SAME", data_format=self.data_format)
          x = _batch_norm(x, layer_id, cell_id, is_training)
          x = tf.nn.relu(x)
          return x

        def _enas_ops(x, layer_id, cell_id, op_id, is_training):
          out = {
            tf.equal(op_id, 0): lambda: tf.identity(x),
            tf.equal(op_id, 1): lambda: _conv_3x3(x, layer_id, cell_id, is_training),
            tf.equal(op_id, 2): lambda: _conv_1x5_5x1(x, layer_id, cell_id, is_training),
            tf.equal(op_id, 3): lambda: _sep_conv_5x5(x, layer_id, cell_id, is_training),
          }
          out = tf.case(out, default=lambda: tf.constant(0, dtype=tf.float32), exclusive=True)
          return out
        self.enas_ops = _enas_ops
      else:
        pass

  def _enas_layer(self, layer_id, prev_layers, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """
    layers = tf.TensorArray(tf.float32, size=self.num_cell_layers + 2, clear_after_read=False)
    layers = layers.write(0, prev_layers[0])
    layers = layers.write(1, prev_layers[1])

    def _condition(cell_id, *args):
      return tf.less(cell_id, self.num_cell_layers)

    def _body(cell_id, layers):
      x_id = self.sample_arc[4 * cell_id]
      x_op = self.sample_arc[4 * cell_id + 1]
      x = layers.read(x_id)
      x = self.enas_ops(x, layer_id, cell_id, x_op, is_training)

      y_id = self.sample_arc[4 * cell_id + 2]
      y_op = self.sample_arc[4 * cell_id + 3]
      y = layers.read(y_id)
      y = self.enas_ops(y, layer_id, cell_id, y_op, is_training)

      out = x + y
      out = tf.nn.relu(out)

      layers = layers.write(cell_id + 2, out)
      return cell_id + 1, layers

    loop_vars = [tf.constant(0, dtype=tf.int32), layers]
    loop_outputs = tf.while_loop(_condition, _body, loop_vars)

    out = loop_outputs[-1].read(self.num_cell_layers + 1)
    out.set_shape(prev_layers[0].get_shape())

    return out

  def _fixed_layer(self, layer_id, prev_layers, start_idx, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """
    pass

  # override
  def _build_train(self):
    print "-" * 80
    print "Build train graph"
    logits = self._model(self.x_train, is_training=True)
    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)
    self.loss = tf.reduce_mean(log_probs)

    self.train_preds = tf.argmax(logits, axis=1)
    self.train_preds = tf.to_int32(self.train_preds)
    self.train_acc = tf.equal(self.train_preds, self.y_train)
    self.train_acc = tf.to_int32(self.train_acc)
    self.train_acc = tf.reduce_sum(self.train_acc)

    tf_variables = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print "Model has {} params".format(self.num_vars)

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
    logits = self._model(self.x_valid, False, reuse=True)
    self.valid_preds = tf.argmax(logits, axis=1)
    self.valid_preds = tf.to_int32(self.valid_preds)
    self.valid_acc = tf.equal(self.valid_preds, self.y_valid)
    self.valid_acc = tf.to_int32(self.valid_acc)
    self.valid_acc = tf.reduce_sum(self.valid_acc)

  # override
  def _build_test(self):
    print "-" * 80
    print "Build test graph"
    logits = self._model(self.x_test, False, reuse=True)
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
        x_valid_shuffle = tf.map_fn(_pre_process, x_valid_shuffle, back_prop=False)

    logits = self._model(x_valid_shuffle, False, reuse=True)
    valid_shuffle_preds = tf.argmax(logits, axis=1)
    valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      self.sample_arc = controller_model.sample_arc
    else:
      fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
      self.sample_arc = fixed_arc

    self._build_train()
    self._build_valid()
    self._build_test()

