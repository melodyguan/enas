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


class GeneralChild(Model):
  def __init__(self,
               images,
               labels,
               cutout_size=None,
               whole_channels=False,
               fixed_arc=None,
               out_filters_scale=1,
               num_layers=2,
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

    self.num_branches = num_branches
    self.fixed_arc = fixed_arc
    self.out_filters_scale = out_filters_scale

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]

  def _model(self, images, is_training, reuse=False):
    with tf.variable_scope(self.name, reuse=reuse):
      layers = []
      with tf.variable_scope("layer_0"):
        if self.fixed_arc is None:
          x = self._enas_layer(0, [images], 0, is_training)
        else:
          x = self._fixed_layer(0, [images], 0, is_training)
        layers.append(x)
        print layers[-1]

      if self.whole_channels:
        start_idx = 1
      else:
        start_idx = 2 * self.num_branches
      for layer_id in xrange(1, self.num_layers):
        with tf.variable_scope("layer_{}".format(layer_id)):
          if self.fixed_arc is None:
            x = self._enas_layer(layer_id, layers, start_idx, is_training)
          else:
            x = self._fixed_layer(layer_id, layers, start_idx, is_training)
          layers.append(x)
          if layer_id in self.pool_layers:
            layers = [max_pool(layer, [2, 2], [2, 2], data_format=self.data_format)
                      for layer in layers]
        if self.whole_channels:
          start_idx += 1 + layer_id
        else:
          start_idx += 2 * self.num_branches + layer_id
        print layers[-1]

      x = global_avg_pool(x, data_format=self.data_format)
      if is_training:
        x = tf.nn.dropout(x, self.keep_prob)
      with tf.variable_scope("fc"):
        w = create_weight("w", [self.out_filters, 10])
        x = tf.matmul(x, w)
    return x

  def _enas_layer(self, layer_id, prev_layers, start_idx, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """

    inputs = prev_layers[-1]
    if self.whole_channels:
      if self.data_format == "NHWC":
        inp_h = inputs.get_shape()[1].value
        inp_w = inputs.get_shape()[2].value
        inp_c = inputs.get_shape()[3].value
      elif self.data_format == "NCHW":
        inp_c = inputs.get_shape()[1].value
        inp_h = inputs.get_shape()[2].value
        inp_w = inputs.get_shape()[3].value

      count = self.sample_arc[start_idx]
      branches = {}
      with tf.variable_scope("branch_0"):
        y = self._conv_branch(
          inputs, 3, is_training, self.out_filters, start_idx=0)
        branches[tf.equal(count, 0)] = lambda: y
      with tf.variable_scope("branch_1"):
        y = self._conv_branch(
          inputs, 3, is_training, self.out_filters, start_idx=0, separable=True)
        branches[tf.equal(count, 1)] = lambda: y
      with tf.variable_scope("branch_2"):
        y = self._conv_branch(
          inputs, 5, is_training, self.out_filters, start_idx=0)
        branches[tf.equal(count, 2)] = lambda: y
      with tf.variable_scope("branch_3"):
        y = self._conv_branch(
          inputs, 5, is_training, self.out_filters, start_idx=0, separable=True)
        branches[tf.equal(count, 3)] = lambda: y
      out = tf.case(branches, default=lambda: tf.constant(0, tf.float32), exclusive=True)

      if self.data_format == "NHWC":
        out.set_shape([None, inp_h, inp_w, self.out_filters])
      elif self.data_format == "NCHW":
        out.set_shape([None, self.out_filters, inp_h, inp_w])
    else:
      count = self.sample_arc[start_idx:start_idx + 2*self.num_branches]
      branches = []
      with tf.variable_scope("branch_0"):
        branches.append(self._conv_branch(inputs, 3, is_training, count[1], start_idx=count[0]))
      with tf.variable_scope("branch_1"):
        branches.append(self._conv_branch(inputs, 3, is_training, count[3], start_idx=count[2], separable=True))
      with tf.variable_scope("branch_2"):
        branches.append(self._conv_branch(inputs, 5, is_training, count[5], start_idx=count[4]))
      with tf.variable_scope("branch_3"):
        branches.append(self._conv_branch(inputs, 5, is_training, count[7], start_idx=count[6], separable=True))
      if self.num_branches >= 5:
        with tf.variable_scope("branch_4"):
          branches.append(self._pool_branch(inputs, is_training, count[9], "avg", start_idx=count[8]))
      if self.num_branches >= 6:
        with tf.variable_scope("branch_5"):
          branches.append(self._pool_branch(inputs, is_training, count[11], "max", start_idx=count[10]))

      with tf.variable_scope("final_conv"):
        w = create_weight("w", [self.num_branches * self.out_filters, self.out_filters])
        w_mask = tf.constant([False] * (self.num_branches * self.out_filters), tf.bool)
        new_range = tf.range(0, self.num_branches * self.out_filters, dtype=tf.int32)
        for i in xrange(self.num_branches):
          start = self.out_filters * i + count[2 * i]
          new_mask = tf.logical_and(start <= new_range, new_range < start + count[2 * i + 1])
          w_mask = tf.logical_or(w_mask, new_mask)
        w = tf.boolean_mask(w, w_mask)
        w = tf.reshape(w, [1, 1, -1, self.out_filters])

        inp = prev_layers[-1]
        if self.data_format == "NHWC":
          branches = tf.concat(branches, axis=3)
        elif self.data_format == "NCHW":
          branches = tf.concat(branches, axis=1)
          N = tf.shape(inp)[0]
          H = inp.get_shape()[2].value
          W = inp.get_shape()[3].value
          branches = tf.reshape(branches, [N, -1, H, W])
        out = tf.nn.conv2d(branches, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)
        out = tf.nn.relu(out)

    if layer_id > 0:
      if self.whole_channels:
        skip_start = start_idx + 1
      else:
        skip_start = start_idx + 2 * self.num_branches
      skip = self.sample_arc[skip_start: skip_start + layer_id]
      with tf.variable_scope("skip"):
        # res_layers = []
        # res_mask = []
        # for i in xrange(layer_id):
        #   mask = tf.cond(tf.equal(skip[i], 1),
        #                  lambda: tf.fill([self.out_filters], True),
        #                  lambda: tf.fill([self.out_filters], False))
        #   res_mask.append(mask)
        #   res_layers.append(prev_layers[i])
        # res_mask.append(tf.constant([True] * self.out_filters, dtype=tf.bool))
        # res_mask = tf.concat(res_mask, axis=0)
        # w = create_weight("w", [(layer_id + 1) * self.out_filters, self.out_filters])
        # w_mask = tf.reshape(res_mask, [(layer_id + 1) * self.out_filters, 1])
        # w_mask = tf.tile(w_mask, [1, self.out_filters])
        # w = tf.boolean_mask(w, w_mask)
        # w = tf.reshape(w, [1, 1, -1, self.out_filters])
        # prev = res_layers + [out]
        # if self.data_format == "NHWC":
        #   prev = tf.concat(prev, axis=3)
        #   prev = tf.transpose(prev, [3, 0, 1, 2])
        #   prev = tf.boolean_mask(prev, res_mask)
        #   prev = tf.transpose(prev, [1, 2, 3, 0])
        # elif self.data_format == "NCHW":
        #   prev = tf.concat(prev, axis=1)
        #   prev = tf.transpose(prev, [1, 0, 2, 3])
        #   prev = tf.boolean_mask(prev, res_mask)
        #   prev = tf.transpose(prev, [1, 0, 2, 3])

        # out = tf.nn.conv2d(prev, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        res_layers = []
        for i in xrange(layer_id):
          res_layers.append(tf.cond(tf.equal(skip[i], 1),
                                    lambda: prev_layers[i],
                                    lambda: tf.zeros_like(prev_layers[i])))
        res_layers.append(out)
        out = tf.add_n(res_layers)
        out = batch_norm(out, is_training, data_format=self.data_format)
        # out = tf.nn.relu(out)

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

    inputs = prev_layers[-1]
    if self.whole_channels:
      if self.data_format == "NHWC":
        inp_c = inputs.get_shape()[3].value
      elif self.data_format == "NCHW":
        inp_c = inputs.get_shape()[1].value

      count = self.sample_arc[start_idx]
      size = [3, 3, 5, 5]
      filter_size = size[count]

      with tf.variable_scope("conv"):
        w = create_weight("w", [filter_size, filter_size, inp_c, self.out_filters])
      out = tf.nn.relu(inputs)
      out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      out = batch_norm(out, is_training, data_format=self.data_format)
    else:
      count = self.sample_arc[start_idx:start_idx + 2*self.num_branches] * self.out_filters_scale
      branches = []
      total_out_channels = 0
      with tf.variable_scope("branch_0"):
        total_out_channels += count[1]
        branches.append(self._conv_branch(inputs, 3, is_training, count[1]))
      with tf.variable_scope("branch_1"):
        total_out_channels += count[3]
        branches.append(self._conv_branch(inputs, 3, is_training, count[3], separable=True))
      with tf.variable_scope("branch_2"):
        total_out_channels += count[5]
        branches.append(self._conv_branch(inputs, 5, is_training, count[5]))
      with tf.variable_scope("branch_3"):
        total_out_channels += count[7]
        branches.append(self._conv_branch(inputs, 5, is_training, count[7], separable=True))
      if self.num_branches >= 5:
        with tf.variable_scope("branch_4"):
          total_out_channels += count[9]
          branches.append(self._pool_branch(inputs, is_training, count[9], "avg"))
      if self.num_branches >= 6:
        with tf.variable_scope("branch_5"):
          total_out_channels += count[11]
          branches.append(self._pool_branch(inputs, is_training, count[11], "max"))

      with tf.variable_scope("final_conv"):
        w = create_weight("w", [1, 1, total_out_channels, self.out_filters])
        if self.data_format == "NHWC":
          branches = tf.concat(branches, axis=3)
        elif self.data_format == "NCHW":
          branches = tf.concat(branches, axis=1)
        out = tf.nn.conv2d(branches, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)
        # out = tf.nn.relu(out)

    if layer_id > 0:
      if self.whole_channels:
        skip_start = start_idx + 1
      else:
        skip_start = start_idx + 2 * self.num_branches
      skip = self.sample_arc[skip_start: skip_start + layer_id]
      total_skip_channels = np.sum(skip) + 1
      with tf.variable_scope("skip"):
        res_layers = []
        for i in xrange(layer_id):
          if skip[i] == 1:
            res_layers.append(prev_layers[i])
        prev = res_layers + [out]

        if self.data_format == "NHWC":
          prev = tf.concat(prev, axis=3)
        elif self.data_format == "NCHW":
          prev = tf.concat(prev, axis=1)
        # out = tf.add_n(prev)

        w = create_weight("w", [3, 3, total_skip_channels * self.out_filters, self.out_filters])
        out = tf.nn.relu(prev)
        out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)

    return out

  def _conv_branch(self, inputs, filter_size, is_training, count, start_idx=None, separable=False):
    """
    Args:
      start_idx: where to start taking the output channels. if None, assuming
        fixed_arc mode
      count: how many output_channels to take.
    """

    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"

    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3].value
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value

    with tf.variable_scope("inp_conv_1"):
      w = create_weight("w", [1, 1, inp_c, self.out_filters])
      x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      x = tf.nn.relu(x)

    with tf.variable_scope("out_conv_{}".format(filter_size)):
      if start_idx is None:
        if separable:
          w_depth = create_weight("w_depth", [self.filter_size, self.filter_size, self.out_filters, ch_mul])
          w_point = create_weight("w_point", [1, 1, self.out_filters * ch_mul, count])
          x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                     padding="SAME", data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
        else:
          w = create_weight("w", [filter_size, filter_size, inp_c, count])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
      else:
        if separable:
          ch_mul = 1
          w_depth = create_weight("w_depth", [filter_size, filter_size, self.out_filters, ch_mul])
          w_point = create_weight("w_point", [self.out_filters, self.out_filters * ch_mul])
          w_point = w_point[start_idx:start_idx+count, :]
          w_point = tf.transpose(w_point, [1, 0])
          w_point = tf.reshape(w_point, [1, 1, self.out_filters * ch_mul, count])

          x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                     padding="SAME", data_format=self.data_format)
          mask = tf.range(0, self.out_filters, dtype=tf.int32)
          mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
          x = batch_norm_with_mask(
            x, is_training, mask, self.out_filters, data_format=self.data_format)
        else:
          w = create_weight("w", [filter_size, filter_size, self.out_filters, self.out_filters])
          w = tf.transpose(w, [3, 0, 1, 2])
          w = w[start_idx:start_idx+count, :, :, :]
          w = tf.transpose(w, [1, 2, 3, 0])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
          mask = tf.range(0, self.out_filters, dtype=tf.int32)
          mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
          x = batch_norm_with_mask(
            x, is_training, mask, self.out_filters, data_format=self.data_format)
      x = tf.nn.relu(x)
    return x

  def _pool_branch(self, inputs, is_training, count, avg_or_max, start_idx=None):
    """
    Args:
      start_idx: where to start taking the output channels. if None, assuming
        fixed_arc mode
      count: how many output_channels to take.
    """

    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"

    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3].value
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value

    with tf.variable_scope("conv_1"):
      w = create_weight("w", [1, 1, inp_c, self.out_filters])
      x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
      x = batch_norm(x, is_training, data_format=self.data_format)
      x = tf.nn.relu(x)

    with tf.variable_scope("pool"):
      if self.data_format == "NHWC":
        actual_data_format = "channels_last"
      elif self.data_format == "NCHW":
        actual_data_format = "channels_first"

      if avg_or_max == "avg":
        x = tf.layers.average_pooling2d(x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
      elif avg_or_max == "max":
        x = tf.layers.max_pooling2d(x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
      else:
        raise ValueError("Unknown pool {}".format(avg_or_pool))

      if start_idx is not None:
        if self.data_format == "NHWC":
          x = x[:, :, :, start_idx : start_idx+count]
        elif self.data_format == "NCHW":
          x = x[:, start_idx : start_idx+count, :, :]

    return x

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

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    print "-" * 80
    for var in tf_variables:
      print var
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

