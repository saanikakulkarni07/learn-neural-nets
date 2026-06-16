# Learning Neural Networks from Scratch 🧠✨

My study repo for working through **[Neural Networks from Scratch in Python](https://nnfs.io)** (NNFS)
by Harrison Kinsley & Daniel Kukieła — building toward using neural networks in my
**astrophysics research**.

The goal isn't just to *use* a deep-learning library, but to understand what every line is
doing: neurons, layers, activations, loss, gradients, backprop, and optimizers — coded from
scratch with NumPy — so that when I reach for PyTorch on real data (spectra, light curves,
galaxy shapes), the black box is a lot less black.

> ⚠️ **The book PDF is not in this repo.** It's copyrighted (© 2020 Harrison Kinsley) and my
> copy is personally watermarked. It's `.gitignore`d. These notes are my own paraphrasing and
> exercises, not a reproduction of the text. Buy the book at [nnfs.io](https://nnfs.io).

---

## How each chapter is organized

Every chapter folder contains three things:

| File | What it is |
|------|-----------|
| `notes.md` | My distilled notes on the chapter — concepts, math, and an *"Why this matters for astro"* callout. |
| `exercises.ipynb` | A Google Colab–ready notebook with exercises I work through (open via the Colab badge at the top). |
| `notebooklm/source.md` | A clean source document to drop into **[NotebookLM](https://notebooklm.google.com)** for the chapter, plus `study-guide.md` with questions & audio-overview prompts. |

A reusable version of all three lives in [`_templates/`](_templates/).

### Running the notebooks
- **In Colab (easiest):** click the "Open in Colab" badge at the top of any `exercises.ipynb`. Then run `!pip install nnfs` in the first cell.
- **Locally:** `pip install -r requirements.txt`, then `jupyter notebook`.

---

## Progress tracker

Legend: ⬜ not started · 🟦 in progress · ✅ done

| # | Chapter | Notes | Exercises | NotebookLM |
|---|---------|:-----:|:---------:|:----------:|
| 1 | [Introducing Neural Networks](ch01-introducing-neural-networks/) | ✅ | ✅ | ✅ |
| 2 | [Coding Our First Neurons](ch02-coding-our-first-neurons/) | ✅ | ✅ | ✅ |
| 3 | [Adding Layers](ch03-adding-layers/) | ⬜ | ⬜ | ⬜ |
| 4 | [Activation Functions](ch04-activation-functions/) | ⬜ | ⬜ | ⬜ |
| 5 | [Calculating Network Error with Loss](ch05-calculating-network-error-with-loss/) | ⬜ | ⬜ | ⬜ |
| 6 | [Introducing Optimization](ch06-introducing-optimization/) | ⬜ | ⬜ | ⬜ |
| 7 | [Derivatives](ch07-derivatives/) | ⬜ | ⬜ | ⬜ |
| 8 | [Gradients, Partial Derivatives, Chain Rule](ch08-gradients-partial-derivatives-chain-rule/) | ⬜ | ⬜ | ⬜ |
| 9 | [Backpropagation](ch09-backpropagation/) | ⬜ | ⬜ | ⬜ |
| 10 | [Optimizers](ch10-optimizers/) | ⬜ | ⬜ | ⬜ |
| 11 | [Testing with Out-of-Sample Data](ch11-testing-with-out-of-sample-data/) | ⬜ | ⬜ | ⬜ |
| 12 | [Validation Data](ch12-validation-data/) | ⬜ | ⬜ | ⬜ |
| 13 | [Training Dataset](ch13-training-dataset/) | ⬜ | ⬜ | ⬜ |
| 14 | [L1 and L2 Regularization](ch14-l1-and-l2-regularization/) | ⬜ | ⬜ | ⬜ |
| 15 | [Dropout](ch15-dropout/) | ⬜ | ⬜ | ⬜ |
| 16 | [Binary Logistic Regression](ch16-binary-logistic-regression/) | ⬜ | ⬜ | ⬜ |
| 17 | [Regression](ch17-regression/) | ⬜ | ⬜ | ⬜ |
| 18 | [Model Object](ch18-model-object/) | ⬜ | ⬜ | ⬜ |
| 19 | [A Real Dataset](ch19-a-real-dataset/) | ⬜ | ⬜ | ⬜ |
| 20 | [Model Evaluation](ch20-model-evaluation/) | ⬜ | ⬜ | ⬜ |
| 21 | [Saving and Loading Models](ch21-saving-and-loading-models/) | ⬜ | ⬜ | ⬜ |
| 22 | [Prediction / Inference](ch22-prediction-inference/) | ⬜ | ⬜ | ⬜ |

---

## Why I'm doing this (astro context)

I work on galaxy shape measurement with LSST pipelines, and I've got adjacent projects in
spectral classification, light-curve classification, and binary-star detection. Neural nets
keep showing up as the right tool, and I want to wield them with understanding rather than
copy-pasting. Each chapter's notes end with a short note connecting the concept to a concrete
astro use case.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook        # or open any exercises.ipynb in Colab
```
