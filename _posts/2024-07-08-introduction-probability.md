---
title: "Introduction to basic probability."
categories:
  - Basic Probability
tags:
  - Probability
---
# Introduction

In this notebook I want to implement some of the ideas from [this lecture](https://www.youtube.com/watch?v=2Xwk6yNq9og) in Python.
The approach of implementation is inspired by [the great notebook of Peter Norvig](https://github.com/norvig/pytudes/blob/main/ipynb/Probability.ipynb).

Let us start with the notion of a *sample space*. A sample space describes the set of all possible outcomes when performing an experiment.

Let us consider a simple coin throw. The sample space is simply $\{H, T\}$. We can get either Heads or Tails as a result.

```python
sample_space_coin = {"H", "T"}
```

Let's consider the experiment that we throw a coin three times.
The state space contains all possible combinations of Heads and Tails at each turn.

```python
from itertools import product

sample_space_coins = set(product(sample_space_coin,repeat=3))
sample_space_coins
```
```
{('H', 'H', 'H'),
 ('H', 'H', 'T'),
 ('H', 'T', 'H'),
 ('H', 'T', 'T'),
 ('T', 'H', 'H'),
 ('T', 'H', 'T'),
 ('T', 'T', 'H'),
 ('T', 'T', 'T')}
```

An *event* is a possible outcome of our experiment.
Let's consider the event that Heads showed at least two times up.

```python
event = {s for s in sample_space_coins if s.count("H") >= 2}
event
```
```
{('H', 'H', 'H'), ('H', 'H', 'T'), ('H', 'T', 'H'), ('T', 'H', 'H')}
```
The *frequentist* approach states the following:
If all possible outcomes are equally likely an event E occurs with frequency #E/#S where #E is the number of elements in the Event and #S is the number of elements in the State Space.
For our particular example we can do it very simply in Python.

```python
from fractions import Fraction

Fraction(len(event)/len(sample_space_coins))
```

Similar we can consider a repeated dice throw.
We identify each side of the dice with a number from 1-6.
That means our sample space for a single dice throw is $\{1,...,6\}$

```python
sample_space_dice = {1,2,3,4,5,6}
```

```python
sample_space_dices = set(product(sample_space_dice, repeat=2))
sample_space_dices
```
```
{(1, 1),
 (1, 2),
 (1, 3),
 (1, 4),
 (1, 5),
 (1, 6),
 (2, 1),
 (2, 2),
 (2, 3),
 (2, 4),
 (2, 5),
 (2, 6),
 (3, 1),
 (3, 2),
 (3, 3),
 (3, 4),
 (3, 5),
 (3, 6),
 (4, 1),
 (4, 2),
 (4, 3),
 (4, 4),
 (4, 5),
 (4, 6),
 (5, 1),
...
 (6, 2),
 (6, 3),
 (6, 4),
 (6, 5),
 (6, 6)}
```
Let's consider the event that the sum of two dice throws is four or less.

```python
event = {s for s in sample_space_dices if sum(s) <= 4}
event
```
```
{(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)}
```

```python
Fraction(len(event), len(sample_space_dices))
```
```
Fraction(1, 6)
```