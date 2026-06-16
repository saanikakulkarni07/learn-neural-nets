# NotebookLM Source — Chapter 2: Coding Our First Neurons

<!--
HOW TO USE: Upload as a source in a NotebookLM notebook for Chapter 2. Clean self-contained
prose so NotebookLM can ground answers and the audio overview. Pair with study-guide.md.
Original explanatory prose, not a copy of the book text.
-->

## Overview

This is the first chapter that writes code. It builds a neural network's most basic unit, the
neuron, in plain Python, then grows it into a layer, and finally rewrites everything using the
NumPy library so that an entire layer can process an entire batch of data in a single, fast
operation. The whole chapter is really about one idea expressed at increasing levels of
sophistication: a neuron multiplies its inputs by weights, sums them, and adds a bias, and all
of the apparent complexity later in the book is built by stacking and vectorizing this one
operation.

## A single neuron

A neuron receives several inputs. Each input has an associated weight, and the neuron has a
single bias. The neuron multiplies each input by its corresponding weight, adds all of those
products together, and then adds the bias. For example, with four inputs equal to 1.0, 2.0,
3.0, and 2.5, weights equal to 0.2, 0.8, negative 0.5, and 1.0, and a bias of 2.0, the output
is 1.0 times 0.2, plus 2.0 times 0.8, plus 3.0 times negative 0.5, plus 2.5 times 1.0, plus
2.0, which equals 4.8. A neuron always has exactly as many weights as it has inputs, and exactly
one bias. In this chapter the weights and biases are simply made up; in a real network the
weights are typically initialized randomly and the biases to zero, and then both are adjusted
during training.

## A layer of neurons

A layer is simply a group of neurons that all receive the same inputs. What makes each neuron
distinct is that each has its own set of weights and its own bias, so each produces its own
output. The output of the layer is therefore a list of values, one per neuron. With three
neurons receiving four shared inputs, the layer produces three outputs. In plain Python this is
done with two nested loops: an outer loop over each neuron's weights and bias, and an inner loop
over the paired inputs and weights, accumulating the weighted sum before adding the bias. The
built-in zip function pairs the iterables together so the same code works for any number of
inputs or neurons. This arrangement, in which every input connects to every neuron, is called a
fully connected, or dense, layer.

## Tensors, arrays, and vectors

The chapter clarifies some vocabulary. A Python list becomes an array when it is homologous,
meaning every sublist along a given dimension has the same length. The shape of an array
describes its size in each dimension, read from the outermost brackets inward; for instance a
list of three two-element lists has shape three by two. A matrix is a two-dimensional array with
rows and columns. A vector, in this machine-learning context, is just a one-dimensional array,
which in plain Python is an ordinary list. The book offers a deliberately simple working
definition of a tensor: a tensor is an object that can be represented as an array. For the
purposes of writing code, tensors can simply be treated as arrays.

## The dot product

The dot product of two equal-length vectors is the sum of the products of their corresponding
elements, and it produces a single number, a scalar. This is exactly the operation a neuron
performs before adding its bias: multiply each input by its weight and sum. So a neuron's
pre-bias output is literally a dot product of the input vector and the weight vector. Vector
addition, by contrast, is performed element by element on two vectors of the same length and
produces another vector; this is how a layer's vector of biases is added to its vector of
dot-product results.

## A layer with NumPy

Using NumPy, the weights of a layer form a matrix, where each row is one neuron's weight vector,
and the inputs form a vector. NumPy's dot function, given a matrix and a vector, treats the
matrix as a list of vectors and computes the dot product of each row with the input vector,
returning one output per neuron. Adding the bias vector to that result, again element by
element, completes the layer. This replaces the explicit nested loops with a single concise and
much faster expression.

## Batches, the matrix product, and transposition

In practice a network processes many samples at once, called a batch. Training in batches is
faster because the work is done in parallel, and it improves generalization, because adjusting
the weights to fit one sample at a time tends to overfit to individual samples rather than
learning the overall pattern. When the inputs become a batch, they form a matrix rather than a
single vector, and the weights are also a matrix, so the operation needed is the matrix product:
the dot products of every row of the first matrix with every column of the second.

The matrix product has a strict rule: the second dimension of the left matrix must equal the
first dimension of the right matrix. Here the batch of inputs has shape three samples by four
features, and the weights have shape three neurons by four features. These do not line up for a
matrix product because both have four as their second dimension. The fix is transposition, an
operation that turns rows into columns and columns into rows. Transposing the weight matrix
changes its shape from three by four to four by three, and now the input matrix, shaped three by
four, can be multiplied by it, producing a result of shape three by three.

Crucially, the inputs are placed first and the transposed weights second. This choice makes the
result sample-related: each row of the output corresponds to one input sample, and the columns
within that row are that sample's outputs from each neuron. This ordering matters because the
output will be passed onward, and the next layer expects a batch of samples arranged as rows.
The bias vector is then added to every row. In code this is written as np dot of inputs and the
transposed weight array, plus the biases. Plain Python lists do not support the transpose
operation, so the weights are wrapped in a NumPy array first; the biases can remain a plain list
because NumPy converts them automatically.

The key takeaway is that the output of a network is always a list of predictions, one per input
sample, even when only a single sample is provided. This is why deep learning libraries expect a
batch dimension and return a list of predictions even for one input.

## Connections to astrophysics

A batch is naturally a catalog: each row is one astronomical source, and each column is a
feature such as a magnitude in a photometric band. The single expression that multiplies the
input batch by the transposed weights and adds the biases pushes an entire catalog through a
network layer in one vectorized operation, which is exactly how one would score many sources at
once. The convention that the data matrix has shape number of objects by number of features is
the same layout used throughout astronomical machine learning and by table libraries, so getting
the shape and transpose conventions right here pays off directly later. Replacing per-object
Python loops with a single array operation is the same performance argument that motivates
vectorized operations on large tables at survey scale.

## Glossary

- **Neuron**: a unit that computes a weighted sum of its inputs plus a bias.
- **Weight**: a multiplier on one input; a neuron has one weight per input.
- **Bias**: a single additive constant per neuron.
- **Layer**: a group of neurons sharing the same inputs, each with its own weights and bias.
- **Dense / fully connected layer**: a layer in which every input connects to every neuron.
- **Dot product**: sum of element-wise products of two equal-length vectors; returns a scalar.
- **Matrix product**: dot products of all row-column combinations of two matrices; returns a matrix.
- **Shape**: the size of an array along each of its dimensions.
- **Vector / matrix / tensor**: one-dimensional array / two-dimensional array / object representable as an array.
- **Batch**: a group of samples processed together.
- **Transposition**: swapping an array's rows and columns; written with a trailing .T in NumPy.
