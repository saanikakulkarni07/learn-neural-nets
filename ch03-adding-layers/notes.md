# Chapter 3: Adding Layers

> NNFS pp. 59–71 · *my notes, in my own words*

## TL;DR
Two upgrades that turn loose code into a real (untrained) network: (1) **stack layers** — a
layer's output is the next layer's input, which is what makes a network *deep*; (2) wrap a layer
in a reusable **`Layer_Dense` class** with **random weight init** and **zero biases**, so we
stop hand-typing numbers. We also stop hand-typing *data*: the `nnfs` package's `spiral_data`
gives a non-linear dataset that a straight line can't separate — the whole reason we need
neural nets.

## Hidden layers & "deep"
- A network is **deep** when it has **≥2 hidden layers**.
- A **hidden layer** is any layer that isn't the input or the output — you don't directly
  consume its values, but you *can* inspect them (useful for debugging). The name doesn't mean
  inaccessible.
- Right now we treat our layers as hidden; the output layer comes later.

## Stacking layers manually
To add a second layer, its inputs must match the first layer's **outputs**. A layer's neuron
count = number of weight sets = number of biases. So if layer 1 has 3 neurons, every weight set
in layer 2 must have **3 weights** (one per incoming value):
```python
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases       # (3 samples, 3 neurons)
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
# layer2_outputs:
# [[ 0.5031,  -1.04185, -2.03875],
#  [ 0.2434,  -2.7332,  -5.7633 ],
#  [-0.99314,  1.41254, -0.35655]]
```
There's only **one** `inputs` but two sets of weights/biases — because layer 2's input *is*
layer 1's output.

## Training data: the `nnfs` package
```python
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()        # seed=0, default float32 dtype, and a fixed np.dot — for reproducibility
X, y = spiral_data(samples=100, classes=3)   # X: (300, 2) features, y: (300,) class labels
```
- **Linear data** can be fit by a straight line; **non-linear data** (like the spiral) cannot.
  Simpler ML models struggle with non-linear data — neural nets are built for it.
- In `X`, each point is a sample with 2 features (its x,y coords); `y` is its class (0/1/2). The
  network never sees the colors — those are just our labels to fit toward.

## The `Layer_Dense` class ⭐
A "dense" (a.k.a. fully-connected / "fc") layer as a reusable object:
```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
```

### Why these choices
- **Weights shape `(n_inputs, n_neurons)`**, *not* `(n_neurons, n_inputs)`. Storing them this
  way means the forward pass is `np.dot(inputs, self.weights)` with **no transpose every pass**
  (we paid that cost once, conceptually, in Ch. 2).
- **`0.01 * np.random.randn(...)`** — `randn` draws from a Gaussian (mean 0, variance 1),
  giving small +/− values centered on 0. Scaling by `0.01` keeps starting weights small so they
  don't dwarf the tiny updates training makes; networks like values roughly in [−1, 1].
- **`np.zeros((1, n_neurons))` biases** — start at 0; shape `(1, n_neurons)` is a row vector so
  it broadcasts cleanly onto the `(samples, neurons)` dot product, no transpose needed.
  - *Caveat — dead neurons:* with a step/ReLU-style activation, if `inputs·weights + bias ≤ 0`
    for **every** sample, the neuron always outputs 0, contributes nothing, and (foreshadowing
    backprop) can't learn — it's "dead." Non-zero bias init is one lever to avoid this; usually
    0 is fine.

### Using it
```python
X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)     # 2 input features -> 3 neurons
dense1.forward(X)
print(dense1.output[:5])       # 5 samples x 3 neuron-outputs (values tiny, ~1e-4, b/c weights*0.01)
```
Output is `(300, 3)`: one row per sample, 3 values per row (one per neuron). The first row is
all zeros because the first spiral point sits at the origin (`X[0] = [0,0]`), so `0·w + 0 = 0`.

## Gotchas
- Layer-to-layer **shape contract**: layer *k*'s `n_neurons` must equal layer *k+1*'s `n_inputs`.
  This is the thing to check when stacking.
- Weights are `(n_inputs, n_neurons)` here — opposite of how we first wrote them in Ch. 2. The
  class convention removes the per-pass `.T`.
- Outputs are still tiny right now (no activation yet) — that's expected; activations (Ch. 4)
  are the missing piece before this is a "real" network.

## Why this matters for astro 🔭
- `spiral_data` is a stand-in for the **non-linearly separable** classes you actually meet: e.g.
  star/galaxy/QSO populations that overlap in a color–color diagram, or interleaved stellar
  populations — no straight cut separates them, which is exactly why you'd reach for a network.
- `Layer_Dense(n_features, n_neurons)` is the reusable building block you'll stack for a real
  classifier; `X` of shape `(N_objects, N_features)` is the same design-matrix layout as an
  `astropy` table or an sklearn feature matrix.
- The small-random-weights / zero-bias / fixed-seed (`nnfs.init`) discipline is the same
  reproducibility hygiene you want in a research pipeline — seed everything so a run is repeatable.

## Questions to revisit
- Why does multiplying randn by exactly `0.01` matter, and what breaks if weights start large? (training dynamics; later chapters)
- Better-than-uniform init schemes (Xavier/Glorot, He) — the book uses simple `0.01*randn`; when
  would I want those instead? (general DL knowledge, beyond this book)
- The output is still linear in the input until we add an activation — confirm in Ch. 4 why
  stacking Dense layers alone still collapses to one linear map.
