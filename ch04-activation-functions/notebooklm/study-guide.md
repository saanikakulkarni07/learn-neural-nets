# NotebookLM Study Guide — Chapter 4: Activation Functions

> 🎧 **Live notebook:** https://notebooklm.google.com/notebook/3d7c11db-1f45-4e75-9b02-0b0661ef764b
> (private to my Google account)

Use these with the NotebookLM notebook built from `source.md`.

## Quick recall (test yourself, then ask NotebookLM to check)
1. Name the four activation functions in this chapter and where each is typically used.
2. Why is a network of only linear neurons unable to fit a curve like `sin(x)`?
3. What's the optimizer's problem with the step function?
4. In a ReLU neuron, what does the weight control vs. the bias? What does a negative weight do?
5. How do a *pair* of ReLU neurons create an "area of effect"?
6. Write the ReLU forward pass in one line of NumPy.
7. What two properties of `e^x` make exponentiation the right first step in Softmax?
8. State the numerical-stability trick and explain why it doesn't change Softmax's output.
9. What do `axis=1` and `keepdims=True` each do in the Softmax normalization?
10. After a full untrained forward pass on 3 classes, what do the outputs look like, and why?

## Deeper prompts (paste into NotebookLM chat)
- "Walk me through, step by step, why stacking linear layers collapses to a single linear function."
- "Explain how weight (slope/sign) and bias (horizontal shift) let a pair of ReLU neurons bound an 'area of effect', and how many pairs approximate a sine wave."
- "Derive why subtracting the max from all Softmax inputs leaves the output identical."
- "I'm confused by axis vs keepdims. Give me a 3x3 example and quiz me on the shapes for axis=0, axis=1, and axis=1 with keepdims=True."
- "Why is the output of an untrained 3-class network ~0.33 per class, and what changes after training?"

## Audio-overview brief (use as the 'customize' instruction)
> "Generate a focused overview for a researcher who knows Python/NumPy and just built a forward
> pass with no activations. Center it on: why nonlinearity is non-negotiable (linear-only layers
> collapse to one line and can't fit a curve); ReLU as the hidden-layer default and how weight
> (slope/sign) and bias (horizontal shift) let pairs of ReLU neurons build 'areas of effect' that
> approximate any curve; and Softmax for the output — exponentiate (non-negative + monotonic) then
> normalize to a probability distribution, plus the max-subtraction stability trick and why it's
> free. Mention axis=1/keepdims briefly. Use astro framing: ReLU carving curved boundaries for
> star/galaxy/QSO overlap in color-color space, and Softmax giving class confidences you can
> threshold. End on: the net is still untrained (~uniform outputs), so loss comes next."

## Astro tie-in prompts 🔭
- "Relate ReLU's 'areas of effect' to carving curved decision boundaries between star/galaxy/QSO populations in a color-color diagram."
- "Explain how Softmax confidences let me keep high-confidence classifications and flag ambiguous sources in a survey, and how that ties to the chapter's 0.45/0.55 example."
- "Why is the max-subtraction stability trick analogous to guarding exp/log of fluxes spanning many orders of magnitude?"

## After studying, I should be able to…
- [ ] Explain in one sentence why activations are required for depth to matter.
- [ ] Write `Activation_ReLU` and `Activation_Softmax` (with the stability trick) from memory.
- [ ] Reason about `axis`/`keepdims` shapes without guessing.
- [ ] Read a Softmax output as class confidences and apply argmax + a confidence cut.
- [ ] Map ReLU boundaries and Softmax confidences onto an astro classification task.
