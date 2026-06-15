# NotebookLM Study Guide — Chapter 1: Introducing Neural Networks

Use these with the NotebookLM notebook built from `source.md`. They're grouped by how you'd
use NotebookLM: quick recall, deeper "explain it to me" prompts, an audio-overview brief, and
astro-specific tie-ins.

## Quick recall (test yourself, then ask NotebookLM to check)
1. Order from broadest to narrowest: deep learning, AI, neural networks, machine learning.
2. What structural feature makes a network "deep"?
3. Define feature, sample, and label using the server-monitoring example.
4. Supervised vs. unsupervised learning — one sentence each.
5. Classification vs. regression — give one example of each.
6. What does a weight control? What does a bias control?
7. Write the single-neuron output equation from memory.
8. Why can't a network be all linear layers with no activation function?
9. What is overfitting, and how does out-of-sample data detect it?
10. What is the difference between understanding *how* a network computes vs. *why* it decides?

## Deeper prompts (paste into NotebookLM chat)
- "Explain the difference between a weight and a bias using the equation of a line, and give a
  concrete numeric example where changing only the bias changes the prediction."
- "Walk me through the server-failure example step by step, labeling what the features,
  samples, and labels are."
- "Why does stacking purely linear layers give no more power than a single linear layer?
  Explain it intuitively, then with a small algebraic argument."
- "Summarize the full training loop in five sentences: forward pass, loss, adjustment,
  iteration, generalization."
- "I keep confusing classification and regression. Give me five tasks and quiz me on which is
  which."
- "What preprocessing does the chapter say input data usually needs, and why?"

## Audio-overview brief (use as the 'customize' instruction)
> "Generate a focused ~8-minute overview for a researcher who knows Python and statistics but
> is new to neural networks. Prioritize: the AI/ML/NN/deep-learning hierarchy, the single-neuron
> equation with the weight-as-slope and bias-as-offset intuition, why nonlinear activations are
> necessary, and the overfitting-vs-generalization idea. Keep the biology analogy brief. End with
> what the rest of the book will build."

## Astro tie-in prompts 🔭
- "Map every term in this chapter (feature, sample, label, classification, regression,
  overfitting) onto a photometric-redshift estimation task from broadband fluxes."
- "For classifying variable stars from light curves, what are the features and samples, and why
  might random train/test splitting overstate accuracy compared to splitting by observing field?"
- "Explain why the overfitting warning is especially serious when a model trained on one survey
  is applied to data from a different instrument."

## After studying, I should be able to…
- [ ] Sketch a neuron and label inputs, weights, bias, sum, activation, output.
- [ ] Explain deep vs. shallow in one sentence.
- [ ] Predict how a line moves when I change a weight vs. a bias.
- [ ] State why activations are non-negotiable for deep nets.
- [ ] Explain the train/hold-out split to a labmate in 30 seconds.
