# Chapter 1: Introducing Neural Networks

> NNFS pp. 11–24 · *my notes, in my own words*

## TL;DR
A neural network is a stack of simple units (**neurons**) wired together. Each neuron does
almost nothing on its own — multiply inputs by weights, add a bias, optionally squash through
an activation — but wire thousands together in **layers** and tune their **parameters** on
labeled data, and the whole thing becomes a flexible function approximator. This chapter is
all concepts and intuition; no full network is built yet. The job here is to know *what* the
pieces are and *why* before we start coding them in Chapter 2.

## Where neural nets sit
**AI ⊃ ML ⊃ Neural Networks ⊃ Deep Learning.** A network is "deep" when it has **two or more
hidden layers** (layers the network controls that aren't the input or output). Most networks
people use today are deep.

## Key terms
- **Feature** — one measured quantity (e.g. a sensor's temperature reading; a flux in one band).
- **Feature set / sample** — all features for one observation, stored as a vector/array. One sample = one row.
- **Label / target / ground-truth** — the known-correct answer we want the model to output.
- **Classification** — predict a discrete class (e.g. "normal" vs "failure").
- **Regression** — predict a continuous number (e.g. a price, a redshift).
- **Supervised learning** — train on labeled examples (this book's focus).
- **Unsupervised learning** — find structure with no labels (e.g. clustering).
- **Parameters** — the **weights** and **biases**; the tunable numbers. Real nets have thousands→millions.
- **Overfitting** — model memorizes training data instead of learning the underlying pattern.
- **Generalization** — the goal: perform well on unseen (out-of-sample) data.

## Core concepts

### The neuron
A neuron's forward computation is just a weighted sum plus a bias:

$$\text{output} = \Big(\sum_i x_i\, w_i\Big) + b$$

In code, before any activation:
```python
output = sum(i * w for i, w in zip(inputs, weights)) + bias
```

- **Weights** scale each input → they control the **slope** (and sign) of the neuron's response.
- **Bias** shifts the whole output up/down → it's the **offset**.
- This is literally $y = mx + b$ generalized to many inputs. A weight that flips sign flips
  the slope negative; raising the bias slides the line up.

### Activation functions (preview)
After the weighted sum we usually apply an **activation** `output = activation(output)`.
- The **step function** (`1` if `z > 0` else `0`) mimics a biological neuron "firing." Simple,
  but too coarse for learning.
- Modern nets use richer activations — **ReLU** ($\max(0, z)$), **sigmoid**, **softmax** —
  covered properly in Chapter 4. Key reason they exist: **without a nonlinear activation,
  stacking layers collapses to a single linear function** (proven hands-on in the notebook).

### Layers
- **Input layer** — your raw (usually preprocessed/normalized) features.
- **Hidden layers** — the network's internal representation; "deep" = ≥2 of these.
- **Output layer** — shape depends on the task: one neuron per class for classification
  (highest-scoring neuron = predicted class), or a single neuron for binary / regression.
- **Dense (fully-connected) layer**: every neuron connects to every neuron in the next layer.

### The forward pass
Pushing data input → output is the **forward pass**. Written as one giant equation it looks
terrifying (nested sums, exps, maxes — see the book's Fig 1.13), but it decomposes into tiny,
individually-simple operations: multiply, sum, dot product, transpose, max, exp, log. Nothing
beyond high-school algebra per step. The book's whole thesis: complexity = many simple parts.

### Training, in one breath
Show the network labeled examples, measure how wrong it is with a **loss** function, and
nudge weights and biases to reduce that loss, repeatedly. Over many iterations it (hopefully)
**generalizes**. To check for overfitting we hold out **out-of-sample** data (e.g. keep 10k of
100k samples aside) and require similar accuracy there.

## Gotchas / things to remember
- Weights vs biases aren't redundant: weights *multiply* (scale/flip), biases *add* (shift).
  You need both to fit real data.
- "Deep" is a structural claim (≥2 hidden layers), not a vibe.
- A network is a **black box** for *why* (we can't easily read its reasoning) but fully
  transparent for *how* (every operation is explicit). This repo is about owning the *how*.

## Why this matters for astro 🔭
Every supervised astro ML task maps directly onto this vocabulary:
- **Galaxy morphology / star–galaxy separation** → classification; output neuron per class.
- **Photometric redshift estimation** → regression; one continuous output neuron.
- **Spectra / light curves** → each flux bin or photometric measurement is a **feature**; one
  source is a **sample**. Normalizing fluxes into a sane range is the "preprocess your input"
  step mentioned here.
- The overfitting/generalization warning is *the* central risk in astro ML: a model that nails
  your training survey but fails on a new field or instrument hasn't learned physics, it's
  memorized. Holding out by field/time, not just randomly, is the honest version of "out-of-sample."

## Questions to revisit
- Why exactly does a nonlinearity let layers add representational power? (Ch. 4)
- How does the optimizer actually *find* good weights/biases? (Ch. 6–10)
- For imbalanced astro classes (rare transients), is plain accuracy misleading? (Ch. 5, 20)
