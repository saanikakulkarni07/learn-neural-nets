# NotebookLM Source — Chapter 4: Activation Functions

<!--
HOW TO USE: Upload as a source in a NotebookLM notebook for Chapter 4. Clean self-contained
prose so NotebookLM can ground answers and the audio overview. Pair with study-guide.md.
Original explanatory prose, not a copy of the book text.
-->

## Overview

This chapter adds the missing ingredient that makes a stack of layers behave like a true neural
network: the activation function. An activation function is applied to the output of a neuron or
a whole layer, transforming it. The central reason activation functions exist is nonlinearity.
If the activation function is itself nonlinear, then a network with two or more hidden layers
can learn to represent nonlinear relationships. The chapter introduces several activation
functions, then codes two of them: the rectified linear unit for the hidden layers and the
softmax function for the output layer of a classifier. By the end, a complete forward pass runs
from input through to a probability distribution over classes.

A network generally uses two kinds of activation function: one in the hidden layers, usually the
same function for all hidden neurons, and another in the output layer, chosen to match the task.

## The step, linear, sigmoid, and rectified linear functions

The step function outputs one if the weighted sum plus bias is greater than zero, and zero
otherwise. It mimics a neuron firing or not firing. It was used historically in hidden layers
but is rarely chosen today. Its weakness is that it is not informative: the output is only on or
off, so it carries no sense of how close the neuron was to switching. An input of three and an
input of three hundred thousand both produce the same output of one, which makes it hard for a
training optimizer to judge the effect of small changes.

The linear function is simply the equation of a line, where the output equals the input. It is
typically used on the output layer of a regression model, which predicts a continuous value
rather than a class.

The sigmoid function returns a smooth S-shaped curve, approaching zero for very negative inputs,
passing through one half at an input of zero, and approaching one for very positive inputs. Its
output lies between zero and one. Unlike the step function it is granular: its output retains
information about the magnitude of the input, and that smoothness makes it far easier for an
optimizer to work with. The sigmoid was historically used in hidden layers and is later used as
the output activation for binary classification.

The rectified linear unit, abbreviated ReLU, returns the input when the input is greater than
zero and returns zero otherwise. It is, in effect, the line y equals x clipped at zero on the
negative side. Despite its simplicity it is the most widely used activation function today,
mainly because it is fast and efficient to compute compared with the sigmoid, while the single
bend at zero is just enough nonlinearity to be very effective.

## Why nonlinearity is necessary

A neuron with no activation function computes a linear output, equivalent to using the linear
activation y equals x. The crucial fact is that a network built entirely from linear neurons,
no matter how many layers it has, can only ever represent a single linear function, because
composing linear functions yields another linear function. Such a network can only draw a
straight line, and so it cannot fit a curved function such as a sine wave. When the hidden layers
instead use a nonlinear activation like ReLU, the very same network architecture becomes able to
fit the sine wave. This is the entire justification for activation functions.

The way ReLU achieves this is worth understanding. For a single neuron with a single input, the
weight controls the slope of the active portion of the output, and the sign of the weight
determines whether the neuron describes an activation, when output begins to rise, or a
deactivation, when output stops. The bias shifts the activation point horizontally, controlling
where the neuron turns on. With a pair of neurons, the first can set where the output starts to
rise and the second can set where it stops, creating a bounded region the book calls an area of
effect. By combining many such pairs across the hidden layers, a network assembles many small
sloped and flat segments that together approximate an arbitrary curve. More neurons allow a finer
approximation; the book hand-tunes small layers to trace a sine wave and then shows that larger
layers trained by an optimizer fit it almost perfectly.

## Coding the rectified linear unit

ReLU is simple to implement. The most direct version loops over the inputs and appends the input
if it is greater than zero, otherwise appends zero. This can be written more compactly as taking
the maximum of zero and the value. In NumPy the same thing is done in one vectorized call that
compares each element against zero. The ReLU activation is wrapped in a class with a forward
method that stores the maximum of zero and the inputs. When applied after a dense layer, negative
pre-activation values are clipped to zero while positive values pass through unchanged.

## The softmax activation function

For a classifier we want the output layer to express which class the network believes the input
belongs to, as a set of confidence scores. The ReLU outputs are unbounded, not normalized
relative to one another, and mutually independent, so a raw output like twelve, ninety-nine,
three hundred eighteen has no probabilistic meaning. The softmax activation converts such
uncalibrated values into a normalized probability distribution: every value lies between zero and
one, and the values sum to one. The class with the highest score is the prediction, but the other
scores remain meaningful as confidences. For example, a distribution of zero point four five and
zero point five five predicts the second class but with low confidence, and a program might
choose not to act on such an uncertain prediction.

Softmax works in two steps. First it exponentiates each value, raising Euler's number e to the
power of that value. Exponentiation serves two purposes. It guarantees non-negative results,
since a negative probability would be meaningless, and it is monotonic, meaning larger inputs map
to larger outputs, so exponentiating never changes which class has the highest value. The
exponential of negative infinity is zero, the exponential of zero is one, and it grows for
positive inputs. Second, it normalizes by dividing each exponentiated value by the sum of all the
exponentiated values for that sample, turning them into fractions that add up to one.

## Numerical stability and the axis and keepdims arguments

The exponential function grows extremely fast and overflows for even moderately large inputs; the
exponential of one thousand is reported as infinity. To prevent this, the implementation subtracts
the largest value in each sample from all of that sample's values before exponentiating. This
makes the largest value zero, whose exponential is one, and all others negative, whose
exponentials fall between zero and one, so nothing overflows. Crucially, because softmax
normalizes, subtracting the same constant from every input leaves the final output unchanged, so
this stabilization is free.

The normalization needs care with array dimensions. When summing a two-dimensional array, an axis
of zero sums down the columns, while an axis of one sums across the rows. For a batch where each
row is one sample, we want to sum across the rows, using axis one, to get one total per sample.
The keepdims argument, set to true, keeps that result as a two-dimensional column vector with
shape number of samples by one, rather than collapsing it to one dimension. The column-vector
shape is what allows the per-sample sums to broadcast correctly when dividing the matrix of
exponentiated values row by row. The softmax class combines all of this: subtract the row maximum,
exponentiate, sum across rows with keepdims, and divide.

## The complete forward pass

With both activation classes ready, the chapter builds a full small classifier on the spiral
dataset: a dense layer with two inputs and three neurons, followed by a ReLU activation, then a
second dense layer with three inputs and three neurons, followed by a softmax activation. The
output is one probability distribution per sample. Because the network is still untrained, with
random weights and zero biases, the distributions are close to uniform, roughly one third for
each of the three classes, which is expected rather than a bug. To turn a distribution into a
single predicted class, an argmax is taken, returning the index of the highest confidence. The
confidence itself remains important: two distributions can share the same argmax, and therefore
the same predicted class, while differing greatly in how confident that prediction is.

The model is now complete for forward passing but produces random predictions. The next step,
beginning in the following chapter, is to quantify how wrong those predictions are using a loss
function, so that the weights and biases can be adjusted to reduce error.

## Connections to astrophysics

The areas of effect that ReLU builds up are what allow a network to carve the curved, interlocking
decision boundaries needed for classes that overlap in a color-color diagram, such as stars,
galaxies, and quasars, where a single straight cut would fail. Softmax then turns the output into
calibrated-style class confidences, for example the probabilities of star, galaxy, and quasar
that sum to one. In survey work the confidence is the actionable quantity: one might keep only
high-confidence classifications and flag ambiguous sources for follow-up, exactly the do-not-act
situation the chapter describes. Finally, subtracting the maximum before exponentiating is the
same defensive numerical habit needed when taking logarithms or exponentials of fluxes and
magnitudes that span many orders of magnitude, where overflow and underflow are routine hazards.

## Glossary

- **Activation function**: a function applied to a neuron or layer output, providing nonlinearity.
- **Step function**: outputs one if input is positive, else zero; coarse and uninformative.
- **Linear activation**: output equals input; used for regression output layers.
- **Sigmoid**: smooth S-curve mapping to the range zero to one.
- **ReLU (rectified linear unit)**: outputs the input if positive, else zero; the default hidden-layer activation.
- **Softmax**: exponentiate then normalize, producing a probability distribution over classes.
- **Confidence score**: a softmax output value interpreted as the model's certainty for a class.
- **argmax**: the index of the largest value; used to pick the predicted class.
- **Numerical stability (max subtraction)**: subtracting the row maximum before exponentiating to avoid overflow.
- **axis / keepdims**: NumPy arguments controlling which dimension is summed and whether the result stays 2-D for broadcasting.
