---
title: "Permutations and Combinations."
categories:
  - Basic Probability
tags:
  - Probability
---
An ordered set of elements in `[n]={1,2,...,n}` is called a *permutation*.

If we consider an ordered subset with r elements in a set of n elements we call this an *r permutation*.

Example(lottery): An urn contains 25 balls. We draw 4 balls from the urn and record their order and number.
We see that the state space is simple the set {4 permutations}.

```python
from itertools import permutations
import random 

sample_space = set(permutations(range(1,26), 4))
random.sample(sorted(sample_space), 3)
```
```
[(6, 23, 8, 10), (1, 9, 25, 12), (16, 1, 21, 20)]
```
How many elements are in the above sample space?
For the first element we have 25 possibilities, for the second 24, for the third 23 and for the last ball 22 possibilities.

```python
len(sample_space) == 25 * 24 * 23 * 22
```
```
True
```
Let's consider the event that the first two balls are 1 and 2.

```python
event = {s for s in sample_space if s[0]==1 and s[1]==2}
random.sample(sorted(event), 3)
```
```
[(1, 2, 9, 13), (1, 2, 5, 13), (1, 2, 7, 3)]
```
How many elements are in the above event? For the first and second element we have 1 possibility. For the remaining two elements we have 23 and 22 possibilities.

```python
len(event) == 1 * 1 * 23 * 22
```
```
True
```
In general there are n! permutations of $n$. We will call this "n factorial".

There are $(n)_r = n \times (n-1) \times ... \times (n-r+1) = n!/(n-r)!$ r permutations of n. We call this "n fall r".

```python
from math import factorial

def get_number_permutations(n): return factorial(n)

def get_number_fall_permutations(n, r): return factorial(n)/factorial(n-r)
```

```python
len(sample_space) == get_number_fall_permutations(25, 4)
```
```
True
```
Let us consider the above experiment but this time we will ignore the order.
That means something like $(1,2,3,4,5)$ will be equivalent to $(2,1,3,4,5)$.

```python
from itertools import combinations

sample_space_wo = set(combinations(range(1,26), 4))
random.sample(sorted(sample_space_wo), 3)
```
```
[(6, 7, 8, 23), (1, 16, 21, 22), (5, 15, 16, 22)]
```
How many elements will this sample space have?

From above we know that the sample space which takes into account order has $(25)_4$ elements.

To get "the order out" we need to simply divide this quantity by the number of 4-permutations.

${25\choose4} = (25)_4/4!$

The generalized formula is
${n\choose k} = (n)_k/k!$
```python
def get_number_k_out_of_n(n,k):
    return get_number_fall_permutations(n,k)/get_number_permutations(k)
```

```python
len(sample_space_wo) == get_number_k_out_of_n(25,4)
```
```
True
```
We can apply these concepts to the game of poker.
Let's consider the starting hand of Holdem Poker.

We have 13 ranks and 4 suits (Club, Heart, Spades & Diamonds). The deck consists of the unique $(rank, suit)$ combinations. This gives the $13 \times 4 = 52$ cards mentioned above.

```python
from collections import namedtuple
from itertools import product

ranks = list(range(2,15))
suits = ["C", "H", "S", "D"]
Card = namedtuple('Card', ['rank', 'suit'])

cards = [Card(rank, suit) for rank, suit in product(ranks, suits)]
hands = set(combinations(cards, 2))
```

```python
len(hands) == get_number_k_out_of_n(52,2)
```
```
True
```
Let's consider the event that we get a pair as a starting hand.
For that we need the rank of the two cards to be  equal.

```python
event = {h for h in hands if h[0].rank == h[1].rank}
```

How many elements to be have in the event?

We choose one length and than can choose two suits so we have
#E $ = {13 \choose 1}{4 \choose 2}$

```python
len(event) == get_number_k_out_of_n(13,1)*get_number_k_out_of_n(4,2)
```
```
True
```
What's the probability of getting a pair?

```python
from fractions import Fraction 

Fraction(len(event), len(hands)), f"{len(event)/len(hands) * 100:.2f} %"
```
```
(Fraction(1, 17), '5.88 %')
```
Another possible hand is called "suited connector".

That means that the ranks are only seperated by one, for example $((9, H), (10, H))$ is a suited connector.

We want to consider the "best" suited connectors where the minimum rank is larger than 9.

```python
event = {h for h in hands if abs(h[0].rank-h[1].rank)==1 and min(h[0].rank,h[1].rank)>=9 and h[0].suit == h[1].suit}
```

We have 
9, 10, 11, 12 and 13 as the possible lowest rank.
We can choose one of the suits.
So in total we have
${5 \choose 1}{4 \choose 1}$ possibilties.

```python
len(event) == get_number_k_out_of_n(5,1)*get_number_k_out_of_n(4,1)
```
```
True
```
The corresponding probability is easily calculated as:

```python
Fraction(len(event), len(hands)), f"{len(event)/len(hands) * 100:.2f} %"
```
```
(Fraction(10, 663), '1.51 %')
```

If you want to see more about the topics discussed here you may find the corresponding lecture [by Harry Crane](https://www.youtube.com/watch?v=BucyamBwmtE) interesting.