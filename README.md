# Efficient Neural Architecture Search via Parameter Sharing

Authors' implementation of "Efficient Neural Architecture Search via Parameter Sharing" (2018) in TensorFlow.

Includes code for CIFAR-10 image classification and Penn Tree Bank language modeling tasks.

Paper: https://arxiv.org/abs/1802.03268

Authors: Hieu Pham*, Melody Y. Guan*, Barret Zoph, Quoc V. Le, Jeff Dean

## Penn Treebank

The Penn Treebank dataset is included at `data/ptb`. Depending on the system, you may want to run the script `data/ptb/process.py` to create the `pkl` version. All hyper-parameters are specified in these scripts.

To run the ENAS search process on Penn Treebank, please use the script
```
./scripts/ptb_search.sh
```

To run ENAS with a determined architecture, you have to specify the archiecture using a string. The following script provides an exampling of using the architecture we described in our paper.
```
./scripts/ptb_final.sh
```
A sequence of architecture for a cell with `N` nodes can be specified using a sequence `a` of `2N + 1` tokens

* `a[0]` is a number in `[0, 1, 2, 3]`, specifying the activation function to use at the first cell: `tanh`, `ReLU`, `identity`, and `sigmoid`.
* For each `i`, `a[2*i]` specifies a previous index and `a[2*i+1]` specifies the activation function at the `i`-th cell.

For a concrete example, the following sequence specifies the architecture we visualize in our paper

```
0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1
```

<img src="https://github.com/melodyguan/enas/blob/master/img/enas_rnn_cell.png" width="50%"/>

## CIFAR-10

To run the experiments on CIFAR-10, please first download the [dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

To run the ENAS experiments on the _macro space_ as described in our paper, please use the following scripts:
```
./scripts/cifar10_macro_search.sh
./scripts/cifar10_macro_final.sh
```

A macro architecture for a neural network with `N` layers consists of `N` parts, indexed by `1, 2, 3, ..., N`. Part `i` consists of

* A number in `[0, 1, 2, 3, 4, 5]` that specifies the operation at layer `i`-th, corresponding to `conv_3x3`, `separable_conv_3x3`, `conv_5x5`, `separable_conv_5x5`, `average_pooling`, `max_pooling`.

To run the ENAS experiments on the _micro space_ as described in our paper, please use the following scripts:
```
./scripts/cifar10_micro_search.sh
./scripts/cifar10_micro_final.sh
```

Again, all hyper-parameters are specified in these scripts.

