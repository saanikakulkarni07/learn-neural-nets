# Chapter 2: Coding Our First Neurons

> NNFS pp. 25–58 · *my notes, in my own words*

## TL;DR
The first code chapter. We build a single neuron in pure Python, scale it to a layer, then
re-express everything with NumPy. The punchline is one line — `np.dot(inputs, weights.T) + biases`
— that computes a whole **layer** of neurons on a whole **batch** of samples at once. Getting the
shapes and the transpose right is the entire game; everything later in the book stacks on this.

## The single neuron
A neuron multiplies each input by its weight, sums, adds one bias:
```python
inputs  = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias    = 2.0
output  = (inputs[0]*weights[0] + inputs[1]*weights[1] +
           inputs[2]*weights[2] + inputs[3]*weights[3] + bias)   # 4.8
```
- One neuron has **one bias** and **as many weights as it has inputs**.
- Weights/biases start made-up here; later they're randomly initialized (weights) / zero (biases)
  and *learned*.

## A layer of neurons
A layer is just several neurons sharing the **same inputs**, each with its **own** weight vector
and bias. Output = one value per neuron.
```python
inputs  = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],     # neuron 1
           [0.5, -0.91, 0.26, -0.5],  # neuron 2
           [-0.26, -0.27, 0.17, 0.87]]# neuron 3
biases  = [2.0, 3.0, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
# [4.8, 1.21, 2.385]
```
`zip()` pairs up the iterables so the loop is dynamic — change the number of inputs or neurons
and it still works. This is a **fully connected (dense)** layer: every input feeds every neuron.

## Tensors, arrays, vectors (the vocabulary)
- **List** → **array** if it's *homologous* (every sub-list along a dimension is the same length).
- **Shape**: dimensions from outermost brackets in. `[[4,2],[5,1],[8,2]]` has shape `(3, 2)`.
  A 3-D array like `lolol` has shape `(3, 2, 4)`.
- **Matrix** = 2-D array (rows × columns). **Vector** = 1-D array = a Python list.
- The book's working definition: *"A tensor is an object that can be represented as an array."*
  For us, treat tensors as arrays and move on.

## Dot product & vector addition
$$\vec a \cdot \vec b = \sum_i a_i b_i = a_1b_1 + a_2b_2 + \dots + a_nb_n$$
A dot product of two equal-length vectors is a **scalar**. That's *exactly* the "multiply
inputs by weights and sum" operation — so a neuron's pre-bias output **is** a dot product.
```python
import numpy as np
np.dot([0.2,0.8,-0.5,1.0], [1.0,2.0,3.0,2.5]) + 2.0   # 4.8  (single neuron)
```
Vector addition is element-wise (same-size vectors), which is how the **bias vector** gets added
to a layer's dot-product results.

## Layer with NumPy
With weights as a matrix and inputs as a vector, `np.dot` treats the matrix as a list of vectors
and dots each against the input → one output per neuron:
```python
weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
biases  = [2.0, 3.0, 0.5]
np.dot(weights, inputs) + biases     # [4.8, 1.21, 2.385]
```

## Batches, the matrix product, and the transpose ⭐
We feed many **samples** at once (a **batch**) — faster, and it helps generalization (one
sample at a time over-fits to that sample). Now inputs is a matrix and weights is a matrix, so
we need the **matrix product**: dot products of every row of the left with every column of the
right.

- Rule: inner dimensions must match. `(a, n) · (n, b) → (a, b)`.
- `inputs` is `(3 samples, 4 features)`; `weights` is `(3 neurons, 4 features)`. Both are
  `(_, 4)` — the inner dims don't line up. **Transpose** weights to `(4, 3)`:
```python
inputs  = [[1.0,2.0,3.0,2.5],[2.0,5.0,-1.0,2.0],[-1.5,2.7,3.3,-0.8]]   # (3, 4)
weights = [[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
biases  = [2.0, 3.0, 0.5]
layer_outputs = np.dot(inputs, np.array(weights).T) + biases
# (3, 4) · (4, 3) -> (3, 3):
# [[ 4.8 ,  1.21 , 2.385],
#  [ 8.9 , -1.81 , 0.2  ],
#  [ 1.41,  1.051, 0.026]]
```
- **Why `inputs` first, `weights.T` second?** So the result is **sample-related**: row *i* =
  outputs for sample *i*. The next layer expects a batch of inputs, so we want rows = samples.
- `.T` (transpose) swaps rows↔columns. Plain Python lists have no `.T`, so wrap in `np.array`
  first. Biases (a 1-D list) are broadcast-added to every row — NumPy converts it internally.

## Gotchas
- A **dot product** gives a scalar; a **matrix product** gives a matrix. NumPy's `np.dot` does
  both depending on the inputs' shapes — that's the main source of confusion.
- Mismatched shapes are *the* recurring bug. Always sanity-check: `(samples, features) ·
  (features, neurons) → (samples, neurons)`.
- Output is always a **list of predictions** (one row per sample) even for a single sample —
  this is why every DL library wants a batch dimension.

## Why this matters for astro 🔭
- A **batch** is literally a table of objects: rows = sources (samples), columns = features
  (e.g. `[u, g, r, i, z]` magnitudes). `np.dot(inputs, weights.T) + biases` pushes the whole
  catalog through a layer in one vectorized call — exactly how you'd score a galaxy catalog.
- Getting the transpose/shape convention right *now* saves hours later: a `(N_objects,
  N_features)` design matrix is the standard layout for astro ML (and for `sklearn`/`astropy`
  tables), and it's the same `(samples, features)` convention NumPy wants here.
- Vectorizing over a batch instead of looping per object is the same speed argument as using
  array ops over Python loops on a FITS table — minutes vs. milliseconds at survey scale.

## Questions to revisit
- When weights are *learned* (not made up), what shape does the weight matrix take for a layer
  with `n_inputs` inputs and `n_neurons` neurons? (Ch. 3 — `(n_inputs, n_neurons)`, no transpose
  needed because it's stored that way.)
- How does the bias broadcast generalize when batch size varies?
