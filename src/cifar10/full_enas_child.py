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


class FullEnasChild(Model):
  def __init__(self,
               images,
               labels,
               fixed_arc=None,
               num_layers=4,
               filter_size=3,
               out_filters=24,
               keep_prob=1.0,
               batch_size=32,
               pool_layers=[],
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
    self.pool_layers = pool_layers

    if fixed_arc is None:
      self.fixed_arc = fixed_arc
    else:
      print fixed_arc
      fixed_arc = np.array([int(_) for _ in fixed_arc.split(" ")],
                                dtype=np.int32)
      self.fixed_arc = np.array(fixed_arc, dtype=np.bool)

  def _model(self, images, is_training, reuse=None):
    if self.fixed_arc is None:
      layer_types, layer_skips = self.sample_arc
      layers = [
        tf.TensorArray(tf.float32, size=self.num_layers,
                       clear_after_read=False, name="pool_0"),
        tf.TensorArray(tf.float32, size=self.num_layers,
                       clear_after_read=False, name="pool_1"),
        tf.TensorArray(tf.float32, size=self.num_layers,
                       clear_after_read=False, name="pool_2"),
      ]
    else:
      raise NotImplementedError

    inp_n = tf.shape(images)[0]
    with tf.variable_scope(self.name, reuse=reuse):
      x = images

      for layer_id in xrange(self.num_layers):
        with tf.variable_scope("layer_{}".format(layer_id)):
          layer_type = layer_types[layer_id]

          x = self._one_layer(x, layer_type, is_training)

          # residual
          if layer_id > 0:
            layer_skip = layer_skips[layer_id, :layer_id]
            layer_skip.set_shape([layer_id])

            if self.data_format == "NHWC":
              inp_h = x.get_shape()[1].value
              inp_w = x.get_shape()[2].value

              if inp_h == 32:
                prev_layers = layers[0]
              elif inp_h == 16:
                prev_layers = layers[1]
              elif inp_h == 8:
                prev_layers = layers[2]
              else:
                raise ValueError("Wrong value inp_h={}".format(inp_h))

              indices = tf.boolean_mask(tf.range(0, layer_id, dtype=tf.int32),
                                        layer_skip)
              x = prev_layers.gather(indices)
              x = tf.transpose(x, [1, 2, 3, 0, 4])
              x = tf.reshape(x, [inp_n, inp_h, inp_w, -1])
            elif self.data_format == "NCHW":
              inp_h = x.get_shape()[2].value
              inp_w = x.get_shape()[3].value

              if inp_h == 32:
                prev_layers = layers[0]
              elif inp_h == 16:
                prev_layers = layers[1]
              elif inp_h == 8:
                prev_layers = layers[2]
              else:
                raise ValueError("Wrong value inp_h={}".format(inp_h))

              indices = tf.boolean_mask(tf.range(0, layer_id, dtype=tf.int32),
                                        layer_skip)
              x = prev_layers.gather(indices)
              x = tf.transpose(x, [1, 0, 2, 3, 4])
              x = tf.reshape(x, [inp_n, -1, inp_h, inp_w])
            else:
              raise ValueError("Unknown data_format {}".format(self.data_format))

            with tf.variable_scope("residual"):
              w_mask = tf.reshape(layer_skip[:layer_id], [layer_id, 1])
              w_mask = tf.tile(w_mask, [1, self.out_filters])
              w_mask = tf.reshape(w_mask, [layer_id * self.out_filters])
              w = create_weight("w",
                                [layer_id * self.out_filters, self.out_filters])
              w = tf.boolean_mask(w, w_mask)
              w = tf.reshape(w, [1, 1, -1, self.out_filters])

              x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                               data_format=self.data_format)
              x = batch_norm(x, is_training, data_format=self.data_format)
              x = tf.nn.relu(x)

              if self.data_format == "NHWC":
                x.set_shape([None, inp_h, inp_w, self.out_filters])
              elif self.data_format == "NCHW":
                x.set_shape([None, self.out_filters, inp_h, inp_w])
              else:
                raise ValueError("Unknown data_format {}".format(self.data_format))

          if layer_id in self.pool_layers:
            x = max_pool(x, [2, 2], [2, 2], "SAME", data_format=self.data_format)

          pool = x
          if layer_id < self.pool_layers[0]:
            layers[0] = layers[0].write(layer_id, pool)
            pool = max_pool(pool, [2, 2], [2, 2], data_format=self.data_format)
          if layer_id < self.pool_layers[1]:
            layers[1] = layers[1].write(layer_id, pool)
            pool = max_pool(pool, [2, 2], [2, 2], data_format=self.data_format)
          layers[2] = layers[2].write(layer_id, pool)

          print "{}".format(x)

      x = global_avg_pool(x, data_format=self.data_format)
      if is_training: x = tf.nn.dropout(x, self.keep_prob)
      with tf.variable_scope("fc"):
        w = create_weight("w", [self.out_filters, 10])
        logits = tf.matmul(x, w)

    return logits

  def _fixed_layer(self, layer_id, images, layers, mask, is_training):
    raise NotImplementedError("Bite me!")

  def _one_layer(self, inputs, layer_type, is_training, sep_c_mul=1):
    filter_sizes = [1, 3, 5, 7]
    outputs = {}

    if self.data_format == "NHWC":
      inp_h = inputs.get_shape()[1].value
      inp_w = inputs.get_shape()[2].value
      inp_c = inputs.get_shape()[3].value
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1].value
      inp_h = inputs.get_shape()[2].value
      inp_w = inputs.get_shape()[3].value
    else:
      raise ValueError("Unknown data_format {}".format(self.data_format))

    for branch_id, filter_size in enumerate(filter_sizes):
      with tf.variable_scope("conv_{}".format(filter_size)):
        w = create_weight(
          "w", [filter_size, filter_size, inp_c, self.out_filters])
        x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME",
                         data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        x = tf.nn.relu(x)
        outputs[tf.equal(layer_type, branch_id)] = lambda: x

      with tf.variable_scope("sep_conv_{}".format(filter_size)):
        w_depth = create_weight("w_depth",
          [filter_size, filter_size, inp_c, sep_c_mul])
        w_point = create_weight(
          "w_point", [1, 1,  sep_c_mul * inp_c, self.out_filters])
        x = tf.nn.separable_conv2d(inputs, w_depth, w_point, [1, 1, 1, 1],
                                   "SAME", data_format=self.data_format)
        x = batch_norm(x, is_training, data_format=self.data_format)
        x = tf.nn.relu(x)
        outputs[tf.equal(layer_type, branch_id + len(filter_sizes))] = lambda: x

    outputs = tf.case(outputs, default=lambda: tf.to_float(0), exclusive=True)

    if self.data_format == "NHWC":
      outputs.set_shape([None, inp_h, inp_w, self.out_filters])
    elif self.data_format == "NCHW":
      outputs.set_shape([None, self.out_filters, inp_h, inp_w])
    else:
      raise ValueError("Unknown data_format {}".format(self.data_format))

    return outputs

  # override
  def _build_train(self):
    print "-" * 80
    print "Build train graph"
    self.loss = []
    self.train_preds = []
    self.train_acc = []
    self.train_op = []
    self.lr = []
    self.grad_norm = []
    self.optimizer = []
    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")

    logits = self._model(self.x_train, True)

    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    self.num_vars = count_model_params(tf_variables)
    print "-" * 80
    for var in tf_variables:
      print var

    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=self.y_train)

    loss = tf.reduce_mean(log_probs)
    self.loss.append(loss)

    train_preds = tf.argmax(logits, axis=1)
    train_preds = tf.to_int32(train_preds)
    self.train_preds.append(train_preds)

    train_acc = tf.equal(train_preds, self.y_train)
    train_acc = tf.to_int32(train_acc)
    train_acc = tf.reduce_sum(train_acc)
    self.train_acc.append(train_acc)

    train_op, lr, grad_norm, optimizer = get_train_ops(
      loss,
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
    self.train_op.append(train_op)
    self.lr.append(lr)
    self.grad_norm.append(grad_norm)
    self.optimizer.append(optimizer)

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
  def build_valid_rl(self, augment=False):
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

      if augment:
        x_valid_shuffle = tf.map_fn(_pre_process, x_valid_shuffle, back_prop=False)

    logits = self._model(x_valid_shuffle, False, reuse=True)
    valid_shuffle_preds = tf.argmax(logits, axis=1)
    valid_shuffle_preds = tf.to_int32(valid_shuffle_preds)
    self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, y_valid_shuffle)
    self.valid_shuffle_acc = tf.to_int32(self.valid_shuffle_acc)
    self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

  def connect_controller(self, controller_model):
    if self.fixed_arc is None:
      layer_types = controller_model.layer_types
      layer_skips = controller_model.layer_skips
      layer_skips = tf.cast(layer_skips, tf.bool)
      self.sample_arc = (layer_types, layer_skips)
    else:
      self.sample_arc = None

    self._build_train()
    self._build_valid()
    self._build_test()
    print "Model has {} params".format(self.num_vars)

