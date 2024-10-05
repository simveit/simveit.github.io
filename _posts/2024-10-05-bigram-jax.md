---
title: "Bigram Language Model with JAX"
categories:
  - LLM
tags:
  - JAX, TPU
---
# Introduction

Nowadays, everyone has heard of ChatGPT, Claude, or Gemini.

In this blog post, I want to shed some light on how natural language processing works by inspecting a simple model called the bigram model.

Of course, the bigram model is much simpler and less capable than modern-day transformer architectures, but it can nevertheless be helpful to educate ourselves on some basic concepts that are important in modern NLP systems as well.

For the implementation of our model, we use JAX, as it's high-performing and enables us to write our model in a simple way without getting overwhelmed by additional details.

In our implementation, we focus on simplicity and understanding the core concepts.

# Bigram Model

So what is a bigram language model? A bigram model works as follows: We have a vocabulary of tokens (such as words or characters), and the bigram model essentially tries to give us the probability of the next token occurring, given the current token.

Before using deep learning, let us think about how we could solve this problem with ordinary statistical methods. To do that, it can be helpful to visualize a big matrix with dimensions equal to the number of entries in our vocabulary for both rows and columns.

Our running example will deal with character-level representation of language, which is a simplification we make to boil the problem down to its essentials.

The corresponding vocabulary is then simply the lowercase letters 'a' to 'z', as well as an additional token which we call the EOS token, which will indicate when a word ends or begins.

A very simple approach would be to go over the whole data set, preprocessing it such that we insert an EOS token at the beginning and end of each word. Then, for every word, we extract all bigrams of the form $w_i$, $w_{i+1}$, and these bigrams will make up our training set.

For example, the name "emma" would be transformed into the bigrams "\<eos\> e", "e m", "m m", "m a", "a \<eos\>", and we would add these to our training set.

After extracting all the bigrams, we can fill up the matrix such that the ($i$,$j$) entry in the matrix is simply the number of times we see the bigram composed of the $i$th and $j$th characters in our vocabulary (note that in our example, we are using characters instead of words).

If we then normalize the matrix row-wise, that will give us a probability distribution in every row for the corresponding first character.

Of course, this model of language is very limited, as it doesn't take into account the broader context or what tokens preceded the current token beyond the immediate previous one, etc.

# Bigram Model with JAX

How can we cast the above problem into a deep learning framework? Well, it's very simple. Our training set will consist of all the bigrams we have in our text.

We will model this as follows: our model simply consists of one matrix $W$, and this matrix $W$ will then be trained with our training set.

What would an appropriate loss function be, one could ask? Well, we want our matrix to accurately predict the next token given the current token, and a common loss function for this is the negative log likelihood.

The idea is that at each training step, we will look at our ground truth and our predicted probability for that ground truth. For example, if we consider the token 'b' and know that in our current example the next token is 'a', we would look at the probability of 'a' given 'b', which would be in the row corresponding to 'b' and the column corresponding to 'a' in our matrix.

We want to do that for all examples in our training set and want the product of the corresponding probabilities to be as large as possible.

From high school math, we know that the logarithm of a product is simply the sum of the logarithms. If we average this over all of our examples, we will get the corresponding loss.

Note that we need to multiply the logarithm by minus one, as the logarithm of a number between zero and one is always negative and gets larger in absolute value as the number decreases. So, to ensure that the loss decreases when the product of probabilities increases, we need to multiply the logarithm by minus one.

# Implementation

Let's now proceed to implement the above outlined ideas. 

## Data Loading and Preprocessing
```python
def load_data(path: str) -> Tuple[List[str], List[str]]:

    with open(path, 'r') as f:
        data = f.read()

    words = data.splitlines()
    words = [word.strip() for word in words] # Remove leading/trailing whitespace
    words = [word for word in words if word] # Remove empty strings

    vocab = sorted(list(set(''.join(words))))
    vocab = ['<eos>'] + vocab
    print(f"number of examples in dataset: {len(words)}")
    print(f"max word length: {max([len(word) for word in words])}")
    print(f"min word length: {min([len(word) for word in words])}")
    print(f"unique characters in dataset: {len(vocab)}")
    print("vocabulary:")
    print(' '.join(vocab))
    print('example for a word:')
    print(words[0])
    return words, vocab
```
This function simply loads our data, removes leading and trailing white spaces, empty strings and inserts an EOS token into the vocabulary. We then print out some basic statistics on our dataset and return the words as well as the vocabulary. 

```python
def encode(word: str, vocab: List[str]) -> List[int]:
    """
    Encode a word, add <eos> at the beginning and the end of the word.
    """
    return [vocab.index('<eos>')] + [vocab.index(char) for char in word] + [vocab.index('<eos>')]

def decode(indices: List[int], vocab: List[str]) -> str:
    """
    Decode a list of indices to a word using the vocabulary.
    """
    return ''.join([vocab[index] for index in indices])
```
Encoding and decoding a word is very simple as well. We only need to take care to insert an EOS token at the end at the beginning of every word and then look up the corresponding index in our vocabulary. For decoding we will simply reverse the above operation. 

```python
def get_dataset(encoded_words: List[List[int]]) -> Tuple[jax.Array, jax.Array]:
    """
    Convert a list of encoded words to a list of bigrams.
    """
    X = []
    y = []
    for word in encoded_words:
        for char1, char2 in zip(word[:-1], word[1:]):
            X.append(char1)
            y.append(char2)
    return jax.numpy.array(X), jax.numpy.array(y)

```
This function will simply loop over all encoded words which it will get as an argument and then will loop over the bigrams and append them for our training examples and targets. 

```python
def get_train_val_test(encoded_words: List[List[int]]) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Split the dataset into training, validation and test sets.
    """
    random.shuffle(encoded_words)
    train_words = encoded_words[:int(0.8*len(encoded_words))]
    val_words = encoded_words[int(0.8*len(encoded_words)):int(0.9*len(encoded_words))]
    test_words = encoded_words[int(0.9*len(encoded_words)):]
    X_train, y_train = get_dataset(train_words)
    X_val, y_val = get_dataset(val_words)
    X_test, y_test = get_dataset(test_words)
    return X_train, y_train, X_val, y_val, X_test, y_test
```
Following good practice in machine learning to evaluate our model, we will split the data set into train, validation and test data set. Here taking a very simple approach by just putting 80% of all words in the training set, 10% of words in a validation set. And the remaining 10% in the test set. 

## Modelling, training and sampling

```python
class Weights(NamedTuple):
    W: jax.Array

def init_weights(vocab_size: int) -> Weights:
    return Weights(W=jax.numpy.array(np.random.randn(vocab_size, vocab_size)))

def forward(weights: Weights, X: jax.Array, return_logits: bool = False) -> jax.Array:
    """
    1) index into the weights matrix W using the input indices
    2) apply the softmax function to obtain a probability distribution over the next character.
    """
    logits = weights.W[X]
    if return_logits:
        return logits
    exp_logits = jax.numpy.exp(logits)
    probs = exp_logits / jax.numpy.sum(exp_logits, axis=1, keepdims=True)
    return probs
```
The above function simply creates our structure where we keep the weights for our model. Then the second function will initialize the weights drawing from a normal distribution centered around zero with a standard deviation of one. And the forward pass is actually very simple but let's look a little bit more detailed into it. 
Looking at the forward pass, we see that the only thing we really need to do is to look up the corresponding indices given in our X, which will then give us the predicted probabilities for the corresponding character. We then can optionally turn on the return logits argument to false to return probabilities. 
We do that by first exponentiating the logit

```python
def loss(weights: Weights, X: jax.Array, y: jax.Array) -> jax.Array:
    """
    1) get the probabilities for the next character
    2) index into the probabilities using the true next character
    3) take the negative log of the probability
    4) return the mean loss over all the examples
    """
    probs = forward(weights, X)
    return -jax.numpy.log(probs[jax.numpy.arange(len(y)), y]).mean()

def update(weights: Weights, X: jax.Array, y: jax.Array, learning_rate: float) -> Union[Weights, Any]:
    """
    1) get the probabilities for the next character
    2) compute the gradient of the loss with respect to the weights
    3) update the weights using the gradient
    """
    grads = jax.grad(loss)(weights, X, y)
    return jax.tree.map(lambda w, g: w - learning_rate * g, weights, grads)

@jax.jit
def train_step(weights: Weights, X: jax.Array, y: jax.Array, learning_rate: float) -> Tuple[Weights, Union[Any, jax.Array]]:
    """
    1) compute the loss
    2) compute the gradient of the loss with respect to the weights
    3) update the weights using the gradient
    4) return the updated weights and the loss
    """
    loss_value = loss(weights, X, y)
    weights = update(weights, X, y, learning_rate)
    return weights, loss_value
```
The next step is then to implement the loss. For that we will simply extract the probabilities for the given weights matrix and the training examples x, give that as an argument to the forward function. Then we will extract the probabilities which our model gives us for the ground truth. Take the negative of the logarithm of that and the mean of that over all samples in our training data. 
Our updating step will then be the familiar gradient descent algorithm. We will calculate using JAX's inbuilt grad function the gradient of the loss over the weights and then use that to make a small step into the negative direction. Which will minimize the loss. 
Our training step consists simply of calculating the loss and updating the weights. 

```python
def train(weights: Weights, X_train: jax.Array, y_train: jax.Array, X_val: jax.Array, y_val: jax.Array, learning_rate: float, N_EPOCHS: int) -> Weights:
    """
    1) loop over the number of epochs
    2) for each epoch, loop over the training data and update the weights
    3) compute the validation loss
    4) print the loss and validation loss
    """
    for epoch in range(N_EPOCHS):
        weights, loss_value = train_step(weights, X_train, y_train, learning_rate)
        val_loss = loss(weights, X_val, y_val)
        if epoch % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss_value}, val_loss: {val_loss}")
    return weights
```
We can then wrap that up into a training function which repeats the above described training step for a given number of epochs with a given learning rate. Every tenth epoch we will print out the value of the loss over the training set and over the validation set. 
To familiarize ourselves with the loss function, we can also guess an initial value. What would be an initial value if we don't know anything? So if the model didn't learn anything, well, it can't do much better than predicting uniformly over all characters, which will give us a corresponding loss of $-\log(1/27)$.