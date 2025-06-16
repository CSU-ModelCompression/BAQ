# BAQ: Efficient Bit Allocation Quantization for Large Language Models

This repository contains code for the paper [BAQ: Efficient Bit Allocation Quantization for Large Language Models](https://www.arxiv.org/pdf/2506.05664).

The code is built on top of [OPTQ's repository](https://github.com/IST-DASLab/gptq).

---

## Introduction

We propose a novel framework for allocating quantization bitwidths based on sensitivity metrics derived from a Hessian proxy. We make key assumptions, which allow the layer/component-wise loss function to be expressed as an explicit function of the bitwidths. This enables a neat formulation of the bit allocation problem as a convex optimization task, whose closed-form solution adapts precision across weights to minimize the layer-wise quantization loss.

---

# Dependencies

- `torch`: tested on v1.10.1+cu111
- `transformers`: tested on v4.21.2 (the LLaMa integration currently requires a main install from source and sentencepiece)
- `datasets`: tested on v1.17.0

All experiments were run on a single 80GB NVIDIA A100. However, most experiments will work on a GPU with a lot less memory as well.

# Languange generation

## OPT

```
set CUDA_VISIBLE_DEVICES=0
# Aeverage rate = 3 bits
python opt_baq.py facebook/opt-125m c4 --wbits 3
# Aeverage rate = 2 bits
python opt_baq.py facebook/opt-125m c4 --wbits 2
```
To run other OPT models, replace opt-125m with one of: opt-350m, opt-1.3b, opt-2.7b, opt-6.7b, opt-13b.
