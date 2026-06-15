# NotebookLM Source — Chapter 1: Introducing Neural Networks

<!--
HOW TO USE: Create a NotebookLM notebook for Chapter 1 and upload this file as a source
(you said you'll set up the plugin). It's written as clean, self-contained prose so NotebookLM
can ground its answers and the audio overview on it. Pair it with study-guide.md for prompts.
This is original explanatory prose, not a copy of the book text.
-->

## Overview

This is the conceptual on-ramp to neural networks. No complete network is built yet; the
purpose is to establish vocabulary and intuition so the later, code-heavy chapters make sense.
A neural network is a collection of very simple computational units, called neurons, connected
together in layers. Individually a neuron does something almost trivial. Collectively, after
their internal numbers are tuned on data, they can approximate remarkably complex relationships
between inputs and outputs.

## Where neural networks fit in artificial intelligence

There is a nested hierarchy. Artificial intelligence is the broadest field. Inside it is
machine learning, which is the study of algorithms that improve from data rather than from
hand-written rules. Inside machine learning are neural networks. And inside neural networks is
deep learning. A neural network is described as "deep" when it contains two or more hidden
layers, where a hidden layer is any layer that is neither the input nor the output and whose
values the network itself controls. In practice, most neural networks in use today are deep.

## Supervised and unsupervised learning

In supervised learning we train on data that already carries the correct answers, called labels
or targets or ground truths. A worked example is server monitoring: sensors record upload and
download rates, temperature, and humidity every ten minutes. We label stretches of data as
"normal" or as "preceding a failure," and the model learns to predict, from new sensor readings,
whether failure is imminent. In unsupervised learning there are no labels; the algorithm finds
structure on its own, for example by clustering similar samples together. This book focuses on
supervised learning, specifically classification and regression.

## Features, samples, and labels

Each individual measured quantity, such as one sensor's temperature reading, is called a
feature. The complete group of features describing a single observation is a feature set, and
one such set of values is a sample. Samples are represented as vectors or arrays of numbers.
The correct answer attached to a sample is its label. Classification means predicting a discrete
category, such as "normal" versus "failure." Regression means predicting a continuous numerical
value, such as a price. Both are supervised tasks because both rely on labeled examples.

## The neuron

A single neuron performs a weighted sum of its inputs and then adds a bias. If the inputs are
x1, x2, x3 and their corresponding weights are w1, w2, w3, and the bias is b, then the neuron's
output before any further processing is equal to x1 times w1, plus x2 times w2, plus x3 times
w3, all added together, and then with b added on. Written compactly, the output equals the sum
over all inputs of each input multiplied by its weight, plus the bias.

Two kinds of tunable numbers appear here, and they behave differently. Weights multiply the
inputs, so a weight controls the steepness, or slope, of the neuron's response, and changing a
weight's sign flips that response from increasing to decreasing. The bias is added on, so it
shifts the entire output up or down without changing its slope. This is exactly the equation of
a straight line, output equals weight times input plus bias, which has the same form as y equals
m x plus b. Both weights and biases are needed: weights alone could only scale, and biases alone
could only shift.

These tunable numbers, the weights and the biases together, are called the parameters of the
network. A realistic network has thousands, or even millions, of parameters. Training is the
process of finding values for all of them that make the network produce the outputs we want.

## Activation functions

After computing its weighted sum, a neuron usually passes the result through an activation
function. A simple historical choice is the step function, which outputs one when its input is
greater than zero and outputs zero otherwise. This imitates a biological neuron either firing or
not firing, like an on-off switch. Modern networks instead use more informative activation
functions, such as the rectified linear unit, which outputs the input when it is positive and
zero otherwise, and the sigmoid and softmax functions. The deep reason activation functions
matter is that they introduce nonlinearity. Without a nonlinear activation, stacking many layers
is pointless, because a chain of purely linear operations is mathematically equivalent to a
single linear operation, no matter how many layers there are. Activation functions are covered
in detail in Chapter 4.

## Layers

A network is organized into layers. The input layer holds the actual input data, which is
usually preprocessed first; raw values are commonly scaled or normalized into a small range such
as zero to one or negative one to one, and they must be numeric. The hidden layers are the
network's internal representation, and having two or more of them is what makes a network deep.
The output layer's shape depends on the task. For classification with several classes, there is
typically one output neuron per class, and the class whose neuron has the highest value becomes
the prediction. For binary decisions or for regression, a single output neuron can suffice. The
most common layer type is the dense, or fully connected, layer, in which every neuron is
connected to every neuron in the next layer, and each connection carries its own weight.

## The forward pass

Passing data from the input layer through to the output layer is called the forward pass. If you
write the entire forward pass of even a small network as a single mathematical formula, it looks
intimidating: nested summations, exponentials, and maximum operations. But it decomposes into a
sequence of individually simple steps, including multiplication, summation, the dot product,
transposition, the maximum operation, the exponential, and the logarithm. None of these requires
mathematics beyond basic algebra. The central message of the book is that an apparently complex
system is just a large number of simple parts combined.

## Training and generalization

A network learns by being shown labeled examples, measuring how wrong its predictions are with a
function called the loss, and then slowly adjusting its weights and biases to reduce that loss,
repeated over many iterations. A major danger is overfitting, where the network merely memorizes
the training examples instead of learning the underlying input-output relationship. To detect
this, data is partitioned: some is kept as in-sample training data, and some is withheld as
out-of-sample data. For instance, from one hundred thousand samples, ten thousand might be set
aside, the model trained on the remaining ninety thousand, and then evaluated on the untouched
ten thousand. The objective is generalization: comparable accuracy on data the network has never
seen, which indicates it has learned the pattern rather than memorized the answers.

## Connections to astrophysics

The vocabulary maps directly onto astronomical machine learning. Separating stars from galaxies,
or classifying galaxy morphology, is classification, with one output neuron per class. Estimating
a photometric redshift from broadband fluxes is regression, with a single continuous output. When
the input is a spectrum or a light curve, each flux bin or each photometric measurement is a
feature, and one astronomical source is a sample; normalizing those fluxes into a consistent
range is precisely the input-preprocessing step described above. The warning about overfitting is
especially important in astronomy: a model that performs well on its training survey but fails on
a different field, instrument, or epoch has memorized rather than learned the underlying physics.

## Glossary

- **Feature**: a single measured quantity describing an observation.
- **Sample**: the full set of feature values for one observation, stored as a vector.
- **Label / target / ground-truth**: the known correct answer for a sample.
- **Weight**: a parameter that multiplies an input; controls slope and sign of the response.
- **Bias**: a parameter added after the weighted sum; shifts the output up or down.
- **Parameters**: all the weights and biases of the network.
- **Activation function**: a function applied to a neuron's weighted sum, introducing nonlinearity.
- **Dense layer**: a layer in which every neuron connects to every neuron in the next layer.
- **Forward pass**: the computation of outputs from inputs through the network.
- **Loss**: a measure of how wrong the network's predictions are.
- **Overfitting**: memorizing training data instead of learning the general pattern.
- **Generalization**: performing well on previously unseen data.
- **Deep network**: a network with two or more hidden layers.
