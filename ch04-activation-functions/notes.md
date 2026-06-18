# Chapter 4: Activation Functions

> NNFS pp. 72–110 · *my notes, in my own words*

## TL;DR
Activations are the nonlinearity that lets a deep network fit non-linear functions — without
them, stacking layers collapses to one straight line no matter how deep. We meet four
(Step, Linear, Sigmoid, ReLU), code **ReLU** for the hidden layers, and code **Softmax** for the
output layer so the network emits a probability distribution over classes. By the end we have a
full forward pass: `Dense → ReLU → Dense → Softmax`, producing per-class confidence scores
(still random, because the net is untrained — that's Ch. 5+).

## The four activation functions
| Function | Formula | Range | Where used |
|---|---|---|---|
| **Step** | 1 if x>0 else 0 | {0,1} | historical hidden; too coarse now |
| **Linear** | y = x | (−∞,∞) | regression output layer (Ch. 17) |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0,1) | historical hidden; binary output (Ch. 16) |
| **ReLU** | max(0, x) | [0,∞) | **the** modern hidden-layer default |

- **Two roles:** hidden-layer activation (usually the same for all hidden layers — ReLU) and
  output-layer activation (task-dependent — Softmax for classification, Linear for regression,
  Sigmoid for binary).
- **Step's problem:** it's not *informative*. Output is just on/off, so the optimizer can't tell
  how *close* a neuron was to flipping — input 3 and input 300,000 both give 1. Sigmoid/ReLU are
  granular: the output retains information about the input's magnitude.

## Why nonlinearity? (the key idea)
A neuron with **no activation** is `y = x` (linear). A network of *only* linear neurons — however
many layers — is still just **one linear function** (compose lines, get a line). So it can only
draw a straight line and **cannot fit** something like `y = sin(x)`. Add a nonlinear activation
(ReLU) to the hidden layers and the same `(1, 8, 8, 1)` network *can* fit the sine wave. That's
the whole reason activations exist.

### How ReLU builds curves (intuition)
ReLU is "barely" nonlinear — `y=x` with a flat clip below 0 — but that single bend is enough.
For one neuron with one input:
- **weight** sets the **slope** of the active part (and its sign — negative weight flips it to a
  *deactivation*);
- **bias** shifts the activation point **horizontally** (where the neuron "turns on").
A **pair** of ReLU neurons gives an **"area of effect"**: the first neuron sets where output
*starts* rising, the second sets where it *stops* (deactivates). Many pairs → many little
sloped/flat segments that piece together to approximate any curve. More neurons = finer fit
(the book hand-tunes 8-neuron layers to trace a sine, then shows 64-neuron layers + an optimizer
fitting it almost perfectly).

## ReLU in code
```python
# the definition, literally
output = [max(0, x) for x in inputs]
# NumPy version (works element-wise on arrays):
np.maximum(0, inputs)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
```
Applied after a Dense layer, negative pre-activations get **clipped to 0**; positives pass
through unchanged.

## Softmax (the output-layer activation for classification)
ReLU outputs are **unbounded, un-normalized, and mutually independent** — `[12, 99, 318]` has no
"these are probabilities" meaning. For classification we want a **probability distribution**:
non-negative, summing to 1. Softmax does this:
$$S_{i,j} = \frac{e^{z_{i,j}}}{\sum_{l} e^{z_{i,l}}}$$
Two steps: **exponentiate**, then **normalize** (divide by the row's sum).
- **Why exponentiate** (`e^x`)? It's always **non-negative** (handles negative logits — a
  negative "probability" is nonsense) and **monotonic** (bigger input → bigger output, so it
  never changes which class wins). `e^{-∞}=0`, `e^0=1`.
- **Why normalize?** Turns the exponentials into fractions of their sum → values in [0,1] that
  add to 1 = **confidence scores** per class.

```python
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # stability!
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
```

### The numerical-stability trick ⭐
`np.exp` **explodes**: `exp(1000) → inf` (overflow). Fix: **subtract the row max from every
value before exponentiating**. Largest value becomes 0 (`e^0=1`), everything else negative
(`e^neg ∈ (0,1)`) — no overflow. Because Softmax normalizes, subtracting a constant from all
inputs **doesn't change the output** (`softmax([1,2,3]) == softmax([-2,-1,0])`). Free safety.

### `axis` and `keepdims` (the part that trips people up)
- `axis=None` (default): sum *everything* → scalar.
- `axis=0`: sum **down columns** → one value per column.
- `axis=1`: sum **across rows** → one value per row (this is what we want: one sum per sample).
- `keepdims=True`: keep the result 2-D as a **column vector** `(samples, 1)` instead of `(samples,)`,
  so it **broadcasts** cleanly when dividing the `(samples, classes)` exp array row-wise.

## The full forward pass
```python
X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3);  activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3);  activation2 = Activation_Softmax()
dense1.forward(X);            activation1.forward(dense1.output)
dense2.forward(activation1.output); activation2.forward(dense2.output)
print(activation2.output[:5])   # ~[0.333, 0.333, 0.333] per row
```
Output ≈ uniform `0.33` per class because the net is **random/untrained** (random weights, zero
biases). The **predicted class = `argmax`** of each row. Note: argmax of `[0.22, 0.6, 0.18]` and
`[0.32, 0.36, 0.32]` are both class 1, but 60% confidence ≫ 36% — the *confidence*, not just the
class, carries information.

## Gotchas
- **Linear-only = no power.** If you forget the activation, depth buys nothing.
- **Softmax overflow** is real and silent-ish (a RuntimeWarning, then `inf` → `nan` everywhere).
  Always subtract the max. The book bakes it into the class.
- `keepdims=True` matters: without it the sum is `(n,)` not `(n,1)`, and the division broadcasts
  wrong (or errors).
- Untrained ≈ uniform output is *expected*, not a bug.

## Why this matters for astro 🔭
- **ReLU's "areas of effect"** are why a network can carve the curved, interlocking decision
  boundaries you need in color–color space (star/galaxy/QSO) — regions a linear cut can't.
- **Softmax = calibrated-ish class confidences.** For a 3-way classifier you get
  `[P(star), P(galaxy), P(QSO)]` summing to 1. The *confidence* is the actionable part: in a
  survey you might keep only high-confidence labels and flag the `[0.4, 0.35, 0.25]` ambiguous
  sources for follow-up — exactly the "maybe don't act" case the book describes.
- The **max-subtraction stability trick** is the same defensive habit as guarding `log`/`exp` of
  fluxes/magnitudes spanning many orders of magnitude — overflow/underflow is a routine hazard in
  astro pipelines.

## Questions to revisit
- For an imbalanced 14-class transient problem, is Softmax + argmax enough, or do I need to
  threshold on confidence / reweight? (ties into Ch. 5 loss, Ch. 20 evaluation)
- ReLU vs. Leaky ReLU / GELU — the book uses plain ReLU; when do dead neurons (Ch. 3) force a
  variant? (general DL knowledge)
- Why does subtracting the max leave Softmax output identical — prove it from the formula.
