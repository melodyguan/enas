# Efficient Neural Architecture Search via Parameter Sharing

Authors' implementation of "Efficient Neural Architecture Search via Parameter Sharing" (2018) in TensorFlow.

Includes code for CIFAR-10 image classification and Penn Tree Bank language modeling tasks.

Paper: https://arxiv.org/abs/1802.03268

Authors: Hieu Pham*, Melody Y. Guan*, Barret Zoph, Quoc V. Le, Jeff Dean

_This is not an official Google product._

## Penn Treebank

**IMPORTANT ERRATA**: The implementation of Language Model on this repository is wrong. Please do not use it. The correct implementation is at the [new repository](https://github.com/google-research/google-research/tree/master/enas_lm). We apologize for the inconvenience.

## CIFAR-10

To run the experiments on CIFAR-10, please first download the [dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Again, all hyper-parameters are specified in the scripts that we descibe below.

To run the ENAS experiments on the _macro search space_ as described in our paper, please use the following scripts:
```
./scripts/cifar10_macro_search.sh
./scripts/cifar10_macro_final.sh
```

A macro architecture for a neural network with `N` layers consists of `N` parts, indexed by `1, 2, 3, ..., N`. Part `i` consists of:

* A number in `[0, 1, 2, 3, 4, 5]` that specifies the operation at layer `i`-th, corresponding to `conv_3x3`, `separable_conv_3x3`, `conv_5x5`, `separable_conv_5x5`, `average_pooling`, `max_pooling`.
* A sequence of `i - 1` numbers, each is either `0` or `1`, indicating whether a skip connection should be formed from a the corresponding past layer to the current layer.

A concrete example can be found in our script `./scripts/cifar10_macro_final.sh`.

To run the ENAS experiments on the _micro search space_ as described in our paper, please use the following scripts:
```
./scripts/cifar10_micro_search.sh
./scripts/cifar10_micro_final.sh
```

A micro cell with `B + 2` blocks can be specified using `B` blocks, corresponding to blocks numbered `2, 3, ..., B+1`, each block consists of `4` numbers
```
index_1, op_1, index_2, op_2
```
Here, `index_1` and `index_2` can be any previous index. `op_1` and `op_2` can be `[0, 1, 2, 3, 4]`, corresponding to `separable_conv_3x3`, `separable_conv_5x5`, `average_pooling`, `max_pooling`, `identity`.

A micro architecture can be specified by two sequences of cells concatenated after each other, as shown in our script `./scripts/cifar10_micro_final.sh`

## Citations

If you happen to use our work, please consider citing our paper.
```
@inproceedings{enas,
  title     = {Efficient Neural Architecture Search via Parameter Sharing},
  author    = {Pham, Hieu and
               Guan, Melody Y. and
               Zoph, Barret and
               Le, Quoc V. and
               Dean, Jeff
  },
  booktitle = {ICML},
  year      = {2018}
}
```
