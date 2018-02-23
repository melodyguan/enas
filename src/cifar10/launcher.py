import os
import sys
import time
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("script_folder", None, "Path to the script folder")
flags.DEFINE_string("exp_name", None, "")

exp_version = 0

DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")

DEFINE_integer("batch_size", 32, "")
DEFINE_boolean("sync_replicas", False, "To sync or not to sync.")
DEFINE_integer("num_aggregate", None, "")
DEFINE_integer("num_replicas", None, "")

DEFINE_integer("num_epochs", 300, "")
DEFINE_integer("child_lr_dec_every", 100, "Number of epochs")
DEFINE_integer("child_num_layers", 5, "")
DEFINE_float("child_lr", 0.1, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.5, "")
DEFINE_float("child_l2_reg", 1e-4, "")


def generate_script(params_dict, params_used):
  """
  Args:
    params_dict: the dictionary of available params.
    params_used: the set of iterated params.
  """
  if num_used_params >= num_total_params:
    command = ""

    global exp_version
    script_name = os.path.join(
        FLAGS.script_folder, "{}_v{}".format(FLAGS.exp_name, exp_version))
    exp_version += 1
    with open(script_name, "w") as fout:
      fout.write(command)
    return
  
  # TODO: check which params were not used and iterate them only
  for param_name in sorted(params_dict.iteritems()):
    pass


def main(_):
  params = {
      "batch_size": [384],
      "child_lr": [0.375, 0.75],
      "child_num_layers": [5, 7],
  }

if __name__ == "__main__":
  tf.app.run()
