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


class EnasChild(Model):
  def __init__(self,
               images,
               labels,
               num_branches=6,
               out_filters=24,
               block_size=12,
               num_layers=2,
               keep_prob=1.0,
               batch_size=32,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=10000,
               lr_dec_rate=0.1,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               skip_pattern=None,
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

    self.num_branches = num_branches
    self.out_filters = out_filters
    self.block_size = block_size
    self.num_blocks = out_filters / block_size
    self.total_out_filters = num_branches * out_filters
    self.num_layers = num_layers
    self.skip_pattern = skip_pattern
    self.pool_layers = [self.num_layers // 3, 2 * self.num_layers // 3]

    self._build_skip_connections()

  def _build_skip_connections(self):
    skip_connections = [[] for _ in xrange(self.num_layers)]

    if self.skip_pattern == "dense":
      prev_pool = 0
      for next_pool in (self.pool_layers + [self.num_layers]):
        for i in xrange(prev_pool, next_pool):
          for j in xrange(prev_pool, i):
            skip_connections[i].append(j)
        prev_pool = next_pool
      self.skip_connections = skip_connections
    else:
      assert self.skip_pattern is None, ("Unknown skip_pattern "
                                         "{}").format(self.skip_pattern)

  def _model(self, images, is_training, arc_seq, reuse=None):
    layers = []
    with tf.variable_scope(self.name, reuse=reuse):
      x = images
      for layer_id in xrange(self.num_layers):
        if layer_id == 0:
          inp_masks = tf.constant([True, True, True], dtype=tf.bool)
        else:
          inp_masks = out_masks
          skip_connections = self.skip_connections[layer_id]

          if len(skip_connections) > 0:
            skip_masks = tf.nn.embedding_lookup(arc_seq, skip_connections)
            skip_masks = tf.reshape(skip_masks, [-1])
            inp_masks = tf.concat([inp_masks, skip_masks], axis=0)

          x = [x] + [layers[i] for i in skip_connections]
          if self.data_format == "NHWC":
            x_concat_axis = 3
          elif self.data_format == "NCHW":
            x_concat_axis = 1
          else:
            raise ValueError("Unknown data_format {}".format(self.data_format))
          x = tf.concat(x, axis=x_concat_axis)

        out_masks = tf.slice(arc_seq, [layer_id, 0], [1, -1])
        out_masks = tf.reshape(out_masks, [self.total_out_filters])

        with tf.variable_scope("layer_{}".format(layer_id)):
          c_scale = 1 + len(self.skip_connections[layer_id])
          x = self._one_layer(x, layer_id, inp_masks, out_masks, is_training,
                              inp_c_scale=c_scale)
          if layer_id in self.pool_layers:
            x = max_pool(x, [2, 2], [2, 2], data_format=self.data_format)
          layers.append(x)
          print x

      x = global_avg_pool(x, data_format=self.data_format)

      if is_training:
        x = tf.nn.dropout(x, self.keep_prob)
      with tf.variable_scope("fc"):

        w = create_weight("w", [self.total_out_filters, 10])
        last_mask = tf.slice(arc_seq, [self.num_layers - 1, 0], [1, -1])
        last_mask = tf.reshape(last_mask, [self.total_out_filters])
        w = tf.boolean_mask(w, last_mask)
        x = tf.matmul(x, w)

    return x

  def _one_layer(self, inputs, layer_id, inp_masks, out_masks, is_training,
                 inp_c_scale=1):

    def _get_inp_masked_weight(w, mask):
      """

      Args:
        w: tensor of size [f_size, f_size, inp_filers, out_filters]
        mask: boolean tensor of size [out_filters]
      """
      w = tf.transpose(w, [2, 0, 1, 3])
      w = tf.boolean_mask(w, mask)
      w = tf.transpose(w, [1, 2, 0, 3])
      return w

    def _get_out_masked_weight(w, mask):
      """

      Args:
        w: tensor of size [f_size, f_size, inp_filers, out_filters]
        mask: boolean tensor of size [out_filters]
      """
      w = tf.transpose(w, [3, 0, 1, 2])
      w = tf.boolean_mask(w, mask)
      w = tf.transpose(w, [1, 2, 3, 0])
      return w

    out_masks = tf.split(out_masks, self.num_branches, axis=0)
    with tf.variable_scope(self.name):
      batch_size = tf.shape(inputs)[0]

      if layer_id == 0:
        inp_c = 3
      else:
        inp_c = self.total_out_filters

      outputs = []
      with tf.variable_scope("branch_0"):
        with tf.variable_scope("conv"):
          if layer_id > 0:
            x = tf.nn.relu(inputs)
          else:
            x = inputs
          w = create_weight("w", [1, 1, inp_c * inp_c_scale, self.out_filters])
          if layer_id > 0:
            w = _get_inp_masked_weight(w, inp_masks)
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
        with tf.variable_scope("conv_1x1"):
          x = tf.nn.relu(x)
          w = create_weight("w", [1, 1, self.out_filters, self.out_filters])
          w = _get_out_masked_weight(w, out_masks[0])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm_with_mask(x, is_training, out_masks[0],
                                   self.out_filters,
                                   data_format=self.data_format)
          outputs.append(x)

      with tf.variable_scope("branch_1"):
        with tf.variable_scope("conv"):
          x = tf.nn.relu(inputs)
          w = create_weight("w", [1, 1, inp_c * inp_c_scale, self.out_filters])
          if layer_id > 0:
            w = _get_inp_masked_weight(w, inp_masks)
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
        with tf.variable_scope("conv_3x3"):
          x = tf.nn.relu(x)
          w = create_weight("w", [3, 3, self.out_filters, self.out_filters])
          w = _get_out_masked_weight(w, out_masks[1])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm_with_mask(x, is_training, out_masks[1],
                                   self.out_filters,
                                   data_format=self.data_format)
          outputs.append(x)

      with tf.variable_scope("branch_2"):
        with tf.variable_scope("conv"):
          x = tf.nn.relu(inputs)
          w = create_weight("w", [1, 1, inp_c * inp_c_scale, self.out_filters])
          if layer_id > 0:
            w = _get_inp_masked_weight(w, inp_masks)
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
        with tf.variable_scope("conv_5x5"):
          x = tf.nn.relu(x)
          w = create_weight("w", [5, 5, self.out_filters, self.out_filters])
          w = _get_out_masked_weight(w, out_masks[2])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm_with_mask(x, is_training, out_masks[2],
                                   self.out_filters,
                                   data_format=self.data_format)
          outputs.append(x)

      with tf.variable_scope("branch_3"):
        with tf.variable_scope("conv"):
          x = tf.nn.relu(inputs)
          w = create_weight("w", [1, 1, inp_c * inp_c_scale, self.out_filters])
          if layer_id > 0:
            w = _get_inp_masked_weight(w, inp_masks)
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm(x, is_training, data_format=self.data_format)
        with tf.variable_scope("conv_7x7"):
          x = tf.nn.relu(x)
          w = create_weight("w", [7, 7, self.out_filters, self.out_filters])
          w = _get_out_masked_weight(w, out_masks[3])
          x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
          x = batch_norm_with_mask(x, is_training, out_masks[3],
                                   self.out_filters,
                                   data_format=self.data_format)
          outputs.append(x)

      if self.num_branches >= 5:
        with tf.variable_scope("branch_4"):
          with tf.variable_scope("conv"):
            x = tf.nn.relu(inputs)
            w = create_weight("w", [1, 1, inp_c * inp_c_scale,
                                    self.out_filters])
            if layer_id > 0:
              w = _get_inp_masked_weight(w, inp_masks)
            w = _get_out_masked_weight(w, out_masks[4])
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                             data_format=self.data_format)
            x = batch_norm_with_mask(x, is_training, out_masks[4],
                                     self.out_filters,
                                     data_format=self.data_format)
          with tf.variable_scope("avg_pool"):
            if self.data_format == "NHWC":
              actual_data_format = "channels_last"
            elif self.data_format == "NCHW":
              actual_data_format = "channels_first"
            else:
              raise ValueError("Unknown data_format {}".format(data_format))
            x = tf.layers.average_pooling2d(x, [3, 3], [1, 1], "SAME",
                                            data_format=actual_data_format)
            outputs.append(x)

      if self.num_branches >= 6:
        with tf.variable_scope("branch_5"):
          with tf.variable_scope("conv"):
            x = tf.nn.relu(inputs)
            w = create_weight("w", [1, 1, inp_c * inp_c_scale,
                                    self.out_filters])
            if layer_id > 0:
              w = _get_inp_masked_weight(w, inp_masks)
            w = _get_out_masked_weight(w, out_masks[5])
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                             data_format=self.data_format)
            x = batch_norm_with_mask(x, is_training, out_masks[5],
                                     self.out_filters,
                                     data_format=self.data_format)
          with tf.variable_scope("max_pool"):
            if self.data_format == "NHWC":
              actual_data_format = "channels_last"
            elif self.data_format == "NCHW":
              actual_data_format = "channels_first"
            else:
              raise ValueError("Unknown data_format {}".format(data_format))
            x = tf.layers.max_pooling2d(x, [3, 3], [1, 1], "SAME",
                                        data_format=actual_data_format)
            outputs.append(x)

      if self.data_format == "NHCW":
        outputs = tf.concat(outputs, axis=3)
      elif self.data_format == "NCHW":
        outputs = tf.concat(outputs, axis=1)
      else:
        raise ValueError("Unknown data_format {}".format(self.data_format))

      return outputs

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
    # self.arc_seq = tf.random_uniform([self.num_layers, self.total_out_filters],
    #                                  minval=0, maxval=2, dtype=tf.int32)

    # Pick a random architecture and stick with it
    # arc_seq = np.random.uniform(0, 2, [self.num_layers,
    #                                    self.total_out_filters]).astype(np.int32)
    # print arc_seq
    # self.arc_seq = tf.constant( arc_seq, dtype=tf.int32)
    # self.arc_seq = tf.cast(self.arc_seq, tf.bool)

    def _build_mask(num_blocks):
      """Returns a tensor of size [2^num_block - 1, num_blocks], where each row
        is the binary representation of (row_id + 1), skipping 0.
      """
      num_rows = (2 ** num_blocks) - 1
      num_cols = num_blocks
      block_configs = np.zeros([num_rows, num_cols], dtype=np.bool_)
      for state in xrange(num_rows):
        for i in xrange(num_cols):
          block_configs[state, i] = ((state+1) >> i) & 1 == 1
      return tf.constant(block_configs, dtype=tf.bool)

    block_configs = _build_mask(self.num_blocks)
    # sample_arc = controller_model.sample_arc

    # sample_arc = (
    #    "7 7 5 7 5 7"
    #   " 5 7 5 7 5 7"
    #   " 5 7 7 5 7 7"
    #   " 7 7 7 5 5 7"
    #   " 7 7 5 7 4 5"
    #   " 5 5 5 7 7 5"
    #   " 7 7 7 5 7 7"
    #   " 7 7 5 7 5 7"
    #   " 7 5 5 7 5 7")
    # sample_arc = [int(_) for _ in sample_arc.split(" ")]
    # sample_arc = np.array(sample_arc, dtype=np.int32) - 1

    sample_arc = ("6 6 7 6 6 6 6 7 6 6 2 6 6 6 6 7 7 6 7 7 7 6 6 7 7 7 7 3 6 6 7"
                  " 7 7 7 6 7 7 6 7 7 6 7 6 7 7 6 6 6 7 6 7 7 7 7")
    sample_arc = [int(_) for _ in sample_arc.split(" ")]
    sample_arc = np.array(sample_arc, dtype=np.int32) - 1

    sample_arc = tf.nn.embedding_lookup(block_configs, sample_arc)
    sample_arc = tf.reshape(
      sample_arc, [self.num_layers * self.num_branches * self.num_blocks, 1])
    sample_arc = tf.tile(sample_arc, [1, self.out_filters / self.num_blocks])
    sample_arc = tf.reshape(sample_arc, [self.num_layers,
                                         self.total_out_filters])
    self.sample_arc = sample_arc

    self._build_train()
    self._build_valid()
    self._build_test()
    print "Model has {} params".format(self.num_vars)

