# NotebookLM Source — Chapter 3: Adding Layers

<!--
HOW TO USE: Upload as a source in a NotebookLM notebook for Chapter 3. Clean self-contained
prose so NotebookLM can ground answers and the audio overview. Pair with study-guide.md.
Original explanatory prose, not a copy of the book text.
-->

## Overview

This chapter makes two important transitions. First, it stacks layers so that one layer's
output becomes the next layer's input, which is what actually makes a network deep. Second, it
stops hand-typing both the parameters and the data: a layer is packaged into a reusable class
with random weight initialization, and the training data now comes from a small helper package
that generates a genuinely non-linear dataset. Together these changes move the project from
loose demonstration code toward the reusable components a real network is built from.

## Hidden layers and the meaning of deep

A neural network is called deep when it has two or more hidden layers. A hidden layer is any
layer that is neither the input layer nor the output layer. The name reflects the fact that you
do not directly consume the values these layers produce, the way you do with the final output.
It does not mean the values are inaccessible; in fact you can and often do inspect them, for
example to diagnose problems or improve the network. At this stage the layers being built are
treated as hidden layers, with the output layer still to come.

## Stacking layers manually

To add a second layer, its expected input size must match the first layer's output size. The
number of neurons in a layer equals the number of weight sets and the number of biases it has.
So if the first layer has three neurons, then every weight set in the second layer must contain
exactly three weights, one for each value arriving from the previous layer. In code, the first
layer computes its outputs from the inputs, and then the second layer computes its outputs from
the first layer's outputs. Importantly, there is only one set of input data but two separate
sets of weights and biases, because the second layer's input is simply the first layer's output
rather than the original data.

## Generating training data with the nnfs package

Rather than typing data by hand, the chapter uses a small Python package, installable with pip,
called nnfs. Its datasets module provides a spiral_data function that creates a non-linear
dataset with a chosen number of classes and a chosen number of points per class. The package
also provides an init function. Calling nnfs init does three things: it sets the random seed to
zero, it makes float32 the default data type, and it overrides NumPy's dot product, all to make
results reproducible so that everyone following along sees the same numbers.

The distinction between linear and non-linear data matters here. Linear data can be fit, or
represented, by a straight line. Non-linear data cannot be captured well by a straight line. The
spiral dataset is deliberately non-linear: its classes interleave in a way no straight cut can
separate. Simpler machine learning models struggle with such data, which is exactly the kind of
problem neural networks are built to solve. In the generated data, each point is a sample with
two features, which are its two coordinates, and a separate array holds each point's class label,
such as zero, one, or two. The network never sees the class colors used in plots; those are only
for human understanding, and the labels are what the model will eventually learn to predict.

## The dense layer class

A dense layer, also called a fully connected layer and sometimes abbreviated as fc, is one in
which every input connects to every neuron. The chapter packages this into a class with two
methods: an initialization method and a forward method.

The initialization method takes the number of inputs and the number of neurons. It sets the
weights to small random numbers and the biases to zeros. The weights are created by multiplying
the output of a Gaussian random number generator by 0.01. The Gaussian generator produces values
centered on zero with a spread of one, giving a mix of small positive and negative numbers, and
multiplying by 0.01 keeps them small. Small starting weights are preferred because the network
makes tiny adjustments during training, and if the initial weights were large they would dominate
those adjustments and slow learning; networks generally work best with values roughly between
negative one and one.

A subtle but important detail is the shape of the weights. They are created with shape number of
inputs by number of neurons, rather than the reverse. This is done deliberately so that the
forward pass can compute the dot product of the inputs and the weights directly, without needing
to transpose the weights on every single forward pass, which was the operation discussed in the
previous chapter.

The biases are initialized to zero, with a shape of one by the number of neurons, that is, a row
vector. The row-vector shape lets the bias be added directly to the result of the dot product
without any extra transposition. Zero is the most common bias initialization, but it is not
always ideal. One situation to watch for is the dead neuron. If a neuron's weighted sum plus bias
fails to exceed the threshold of its activation for every single input sample, the neuron always
outputs zero. Since anything multiplied by that zero is also zero, such a neuron contributes
nothing onward and, as becomes clear once backpropagation is covered, cannot learn. It is
effectively dead. Choosing a non-zero bias initialization is one way to reduce this risk,
although zero initialization is usually fine.

The forward method simply computes the dot product of the inputs and the weights and adds the
biases, storing the result. This is the same calculation as before, now wrapped in a method.

## Putting it together

With the class ready, a dataset is generated with the spiral function, a dense layer is created
specifying two input features and three neurons, and the data is passed through the layer's
forward method. The resulting output has one row per sample and three values per row, one from
each neuron. Because the random seed is fixed, the numbers are reproducible. The very first row
is all zeros because the first generated point sits at the origin, so every weighted sum is zero.
At this point the network has layers of neurons but still lacks activation functions, which is
the next missing ingredient before it truly behaves like a neural network.

## Connections to astrophysics

The spiral dataset is a convenient stand-in for the non-linearly separable classes that appear
in real astronomy, such as star, galaxy, and quasar populations that overlap in a color-color
diagram, where no straight cut cleanly separates them. The dense layer, parameterized by the
number of input features and the number of neurons, is exactly the reusable building block one
would stack to construct such a classifier. The data layout, with shape number of objects by
number of features, is the same design-matrix arrangement used by astronomical table libraries
and by standard machine learning tools. Finally, the discipline of fixing the random seed and
keeping initialization small and controlled is the same reproducibility hygiene that a careful
research pipeline should follow, so that any run can be repeated exactly.

## Glossary

- **Hidden layer**: any layer that is not the input or output layer.
- **Deep network**: a network with two or more hidden layers.
- **Dense / fully connected layer**: a layer where every input connects to every neuron.
- **Forward pass**: passing data through the network from input to output.
- **Weight initialization**: setting starting weights, here small random Gaussian values scaled by 0.01.
- **np.random.randn**: draws from a Gaussian distribution with mean zero and variance one.
- **np.zeros**: creates an array of a given shape filled with zeros (used for biases).
- **Dead neuron**: a neuron that outputs zero for all samples and therefore contributes nothing and cannot learn.
- **spiral_data**: an nnfs helper that generates a non-linear, multi-class dataset.
- **nnfs.init**: sets seed zero, default float32, and a fixed dot product for reproducibility.
