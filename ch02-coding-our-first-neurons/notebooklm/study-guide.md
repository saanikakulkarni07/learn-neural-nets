# NotebookLM Study Guide — Chapter 2: Coding Our First Neurons

> 🎧 **Live notebook:** https://notebooklm.google.com/notebook/4406c963-b4ed-4c0c-8311-8be57fc1dd11
> (private to my Google account)

Use these with the NotebookLM notebook built from `source.md`.

## Quick recall (test yourself, then ask NotebookLM to check)
1. State the single-neuron computation in words. How many weights and biases does it have?
2. What makes one neuron in a layer different from another?
3. Write the dot-product formula. What does it return — scalar or vector?
4. Dot product vs. matrix product: what does each return?
5. What is the matrix-product shape rule? Give the resulting shape of `(5,4)·(4,7)`.
6. Why do we transpose the weights when feeding a batch through a layer?
7. Why is `inputs` the first argument to `np.dot` and `weights.T` the second?
8. What is a batch, and why train in batches instead of one sample at a time?
9. What is the shape of the output for a batch of 3 samples through a layer of 3 neurons?
10. Why do deep-learning libraries return a *list* of predictions even for one input?

## Deeper prompts (paste into NotebookLM chat)
- "Walk me through why a neuron's pre-bias output is a dot product, with the 4-input example."
- "Explain the transpose step using the shapes (3,4) for inputs and (3,4) for weights — show me
  each shape before and after, and why the matrix product only works after transposing."
- "Give me three different (samples, features)/(neurons) shape scenarios and quiz me on the
  output shape and whether a transpose is needed."
- "Contrast the pure-Python nested-loop layer with the single NumPy expression. What exactly did
  vectorization replace?"
- "Why does training in batches help generalization compared to one sample at a time?"

## Audio-overview brief (use as the 'customize' instruction)
> "Generate a focused overview for a researcher comfortable with Python and NumPy but new to
> neural nets. Center it on: the single-neuron weighted-sum-plus-bias computation; how a layer is
> just many neurons with their own weights; that a neuron's pre-bias output IS a dot product; and
> especially the batch + transpose step — why (samples, features) · (features, neurons) →
> (samples, neurons) and why inputs go first. Use the photometric-catalog analogy (rows = sources,
> columns = magnitude bands). Keep it concrete and code-flavored."

## Astro tie-in prompts 🔭
- "Frame a batch as a photometric catalog with bands u, g, r, i. Walk through computing two
  color features (u−g, g−r) as a 2-neuron layer with hand-set weights."
- "Explain why the (N_objects, N_features) layout used here matches astropy tables and sklearn,
  and why vectorizing over the batch matters at survey scale."

## After studying, I should be able to…
- [ ] Code a neuron and a dense layer from scratch in pure Python.
- [ ] Explain why a neuron is a dot product plus a bias.
- [ ] Predict the output shape of a batch passing through a layer, and know when to transpose.
- [ ] Write `np.dot(inputs, weights.T) + biases` from memory and explain every piece.
- [ ] Map all of this onto scoring an astronomical catalog.
