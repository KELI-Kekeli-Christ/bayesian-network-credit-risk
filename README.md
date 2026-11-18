<p align="center">
  <a href="README.md"><b>English</b></a>
  &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a>
  &nbsp;·&nbsp;
  <a href="README.es.md">Español</a>
</p>

# Bayesian Network for Credit Risk Assessment

> M1 Informatique — Parcours Intelligence Artificielle · Université d'Avignon · 2024–2025

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![pgmpy](https://img.shields.io/badge/pgmpy-0.1.25-green)](https://pgmpy.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org)
[![Dataset](https://img.shields.io/badge/Dataset-50K%20samples-lightgrey)](./50000-cases.csv)

---

## Overview

This project builds a complete **Bayesian Network** to predict bank loan approval from client financial profiles. It covers manual CPD definition, learning from real data using **Maximum Likelihood Estimation (MLE)**, and both **exact** (Variable Elimination) and **approximate** (Rejection Sampling) probabilistic inference.

---

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Network Architecture](#network-architecture)
- [Experiments & Results](#experiments--results)
  - [1. Manual CPD Definition](#1-manual-cpd-definition)
  - [2. Learning from Data (MLE)](#2-learning-from-data-mle)
  - [3. Exact Inference — Variable Elimination](#3-exact-inference--variable-elimination)
  - [4. Approximate Inference — Rejection Sampling](#4-approximate-inference--rejection-sampling)
- [Key Results](#key-results)
- [Installation](#installation)
- [Usage](#usage)
- [Author](#author)

---

## Background

A **Bayesian Network** is a Directed Acyclic Graph (DAG) where:
- Nodes represent random variables
- Edges encode conditional dependencies
- Each node stores a Conditional Probability Distribution (CPD): $P(X_i \mid \text{Pa}(X_i))$

The joint distribution factorizes as:
$$P(X_1, \ldots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

---

## Project Structure

```
bayesian-network-credit-risk/
├── tp2.ipynb             # Main Jupyter notebook
├── bank.py               # Standalone Python script
├── 50000-cases.csv       # Training dataset (50,000 client records)
├── tp2.png               # Network DAG visualization
├── reseaux-bayesiens.pdf # Course notes
└── README.md
```

---

## Network Architecture

The network models 8 financial variables with 10 dependency edges:

```
DebtIncomeRatio ──► PaymentHistory ──► Age ──► Reliability ──► BankLoan
       │                                              ↑
       └─────────────────────────────────────────────┘
Income ──► Assets ──► FutureIncome ──────────────────► BankLoan
   └──────────────────► FutureIncome
```

| Variable | Type | States |
|----------|------|--------|
| `DebtIncomeRatio` | Root | High, Low |
| `Income` | Root | Low, Medium, High |
| `PaymentHistory` | Internal | Excellent, Acceptable, Unacceptable |
| `Age` | Internal | 16–25, 26–64, Over65 |
| `Assets` | Internal | Low, Medium, High |
| `FutureIncome` | Internal | Promising, Not_promising |
| `Reliability` | Internal | Reliable, Unreliable |
| `BankLoan` | Target | Positive, Negative |

**d-separation analysis** reveals that `Income` is marginally independent of `DebtIncomeRatio`, `PaymentHistory`, `Age`, and `Reliability` — but becomes d-connected to all of them when conditioning on `BankLoan` (explaining away / collider effect).

---

## Experiments & Results

### 1. Manual CPD Definition

CPDs are defined following 7 domain observations (e.g., *"High debt ratio leads to worse payment history"*, *"Better payment history correlates with older age"*).

```
Model valid: True
```

### 2. Learning from Data (MLE)

The network structure is fixed; CPDs are estimated from **50,000 client records** using Maximum Likelihood Estimation:

$$\hat{P}(X_i = x \mid \text{Pa}(X_i) = \mathbf{pa}) = \frac{\#(X_i = x, \text{Pa} = \mathbf{pa})}{\#(\text{Pa} = \mathbf{pa})}$$

Learned marginal distribution of `DebtIncomeRatio`:

| Value | Probability |
|-------|------------|
| High | 0.5014 |
| Low  | 0.4986 |

### 3. Exact Inference — Variable Elimination

| Query | Evidence | P(BankLoan = Positive) |
|-------|----------|------------------------|
| Marginal | — | **0.5426** |
| Low-risk profile | Income=Low, Age=16–25, PayHist=Excellent, Assets=Low | 0.5326 |
| High-income young | Income=High, Age=16–25, PayHist=Excellent, Assets=High | **0.6935** |
| Best profile | Income=High, Age=Over65, PayHist=Excellent, Assets=High | **0.7338** |

### 4. Approximate Inference — Rejection Sampling

Same queries with 10,000 forward samples. Results closely match exact inference:

| Query | Exact | Approx (10K samples) | Exact time | Approx time |
|-------|-------|----------------------|------------|-------------|
| Marginal | 0.5426 | 0.5429 | **0.003s** | 0.16s |
| Low-risk | 0.5326 | 0.5401 | **0.002s** | 13.89s |
| High income | 0.6935 | 0.6966 | **0.002s** | 7.89s |
| Best profile | 0.7338 | 0.7354 | **0.002s** | 1.41s |

**Variable Elimination is up to 7,000× faster** than Rejection Sampling for conditioned queries.

---

## Key Results

| Metric | Value |
|--------|-------|
| Network validity | ✅ Valid DAG |
| Dataset size | 50,000 records |
| P(Loan ≈ granted) marginal | **54.26%** |
| Best profile loan probability | **73.38%** |
| Exact inference speed | **~2ms per query** |
| Speedup over sampling | up to **×7,000** |
| Learned model matches manual CPDs | ✅ Consistent |

---

## Installation

```bash
git clone https://github.com/KELI-Kekeli-Christ/bayesian-network-credit-risk.git
cd bayesian-network-credit-risk
pip install pgmpy pandas jupyter matplotlib networkx
jupyter notebook tp2.ipynb
```

---

## Usage

```python
from pgmpy.inference import VariableElimination

infer = VariableElimination(model_learned)

# Query: P(BankLoan | Income=High, Age=Over65, PaymentHistory=Excellent, Assets=High)
result = infer.query(
    variables=['BankLoan'],
    evidence={'Income': 'High', 'Age': 'Over65',
              'PaymentHistory': 'Excellent', 'Assets': 'High'}
)
print(result)
```

---

## Author

**Christ Kekeli KELI**  
M1 Informatique — Intelligence Artificielle  
Université d'Avignon  
Encadrant : Yezekael HAYEL
