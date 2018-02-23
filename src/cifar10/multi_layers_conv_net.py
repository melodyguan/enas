import os
import sys

import numpy as np
import tensorflow as tf

from src.cifar10.models import Model
from src.cifar10.image_ops import conv
from src.cifar10.image_ops import fully_connected
from src.cifar10.image_ops import batch_norm
from src.cifar10.image_ops import relu
from src.cifar10.image_ops import max_pool
from src.cifar10.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops


class MultiLayerConvNet(Model):
  def __init__(self,
               images,
               labels,
               num_layers=2,
               filter_size=3,
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
               data_format="NHWC",
               name="multi_layer_conv_net",
              ):
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
      optim_algo=optim_algo,
      sync_replicas=sync_replicas,
      num_aggregate=num_aggregate,
      num_replicas=num_replicas,
      data_format=data_format,
      name=name)

    self.num_layers = num_layers
    self.keep_prob = keep_prob
    self.filter_size = filter_size
    with tf.variable_scope(self.name):
      self._build_train()
      tf.get_variable_scope().reuse_variables()
      self._build_valid()
      self._build_test()
      print "Model has {} params".format(self.num_vars)

  def _model(self, images, is_training, reuse=None):
    pool_layers = [self.num_layers // 3, 2 * self.num_layers // 3]
    with tf.variable_scope(self.name, reuse=reuse):
      x = images
      for layer_id in xrange(self.num_layers):
        with tf.variable_scope("layer_{}".format(layer_id)):
          x = conv(x, self.filter_size, 64, [1, 1],
                   data_format=self.data_format, seed=self.seed)
          x = batch_norm(x, is_training, data_format=self.data_format)
          x = relu(x)
          if layer_id in pool_layers:
            x = max_pool(x, [2, 2], [2, 2], data_format=self.data_format)
          print x

      x = global_avg_pool(x, data_format=self.data_format)
      if is_training:
        x = tf.nn.dropout(x, self.keep_prob, seed=self.seed)
      x = fully_connected(x, 10, seed=self.seed)
    return x
