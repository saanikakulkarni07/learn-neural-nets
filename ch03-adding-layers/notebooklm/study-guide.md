# NotebookLM Study Guide — Chapter 3: Adding Layers

> 🎧 **Live notebook:** https://notebooklm.google.com/notebook/0ac4fe48-e2f7-49c4-a9c0-88966bbae143
> (private to my Google account)

Use these with the NotebookLM notebook built from `source.md`.

## Quick recall (test yourself, then ask NotebookLM to check)
1. What makes a network "deep"? What exactly is a "hidden" layer?
2. When stacking layers, what must match between layer *k* and layer *k+1*?
3. Why is there only one `inputs` but two sets of weights/biases when stacking two layers?
4. What does `nnfs.init()` do, and why does it matter?
5. Linear vs. non-linear data — why does the spiral dataset motivate neural networks?
6. In `Layer_Dense.__init__`, what shape are the weights and why that orientation?
7. Why scale the initial weights by `0.01`?
8. Why initialize biases to zero, and what is a "dead neuron"?
9. What does `Layer_Dense.forward` compute, in one line?
10. After `dense1 = Layer_Dense(2, 3); dense1.forward(X)` on spiral data, what's the output shape, and why is the first row all zeros?

## Deeper prompts (paste into NotebookLM chat)
- "Walk me through the shape contract when stacking three dense layers with neuron counts 4 → 6 → 2. What are the weight shapes?"
- "Explain why weights are stored as (n_inputs, n_neurons) and how that removes a transpose from the forward pass."
- "What is a dead neuron, step by step, and how can bias initialization or weight scale make it more or less likely?"
- "Contrast `np.random.randn` and `np.zeros` and why each is used for weights vs. biases."
- "Why are the dense-layer outputs still essentially linear at the end of this chapter, and what fixes that next?"

## Audio-overview brief (use as the 'customize' instruction)
> "Generate a focused overview for a researcher who knows Python/NumPy and just learned how a
> single layer works. Center it on: stacking layers (output of one = input of next) and the
> shape contract; switching to the nnfs spiral dataset and why non-linear data motivates neural
> nets; and the Layer_Dense class — small random weight init scaled by 0.01, zero biases as a
> (1, n_neurons) row vector, the (n_inputs, n_neurons) weight shape that avoids per-pass
> transposes, and the dead-neuron caveat. Use the astro framing: overlapping star/galaxy/QSO
> populations in color-color space as the non-linearly-separable analogue of the spiral, and the
> (N_objects, N_features) design matrix. Keep it concrete and code-flavored."

## Astro tie-in prompts 🔭
- "Relate the spiral dataset's non-linear separability to star/galaxy/QSO classes overlapping in a color-color diagram."
- "Explain how Layer_Dense(n_features, n_neurons) and an (N_objects, N_features) matrix map onto an astropy table / sklearn feature matrix, and why fixing the seed matters for a research pipeline."

## After studying, I should be able to…
- [ ] Stack layers correctly and state the shape contract from memory.
- [ ] Write the `Layer_Dense` class (init + forward) without looking.
- [ ] Justify every initialization choice: weight shape, the 0.01 scale, zero biases.
- [ ] Explain dead neurons and reproducibility (`nnfs.init`).
- [ ] Predict the output shape of a forward pass on a spiral / catalog feature matrix.
