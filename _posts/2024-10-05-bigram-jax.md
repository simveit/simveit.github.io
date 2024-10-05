---
title: "Bigram Language Model with JAX"
categories:
  - LLM
tags:
  - JAX, TPU
---
![Picture](/assets/bigram/image.png)

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
Note that we jitted the training step to perform just-in-time compilation which will increase the performance of our model.
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
We can then wrap that up into a training function which repeats the above described training step for a given number of epochs with a given learning rate. Every tenth epoch we will print out the value of the loss over the training set and over the validation set. Note that here we don't need to loop over the training set, as we can perform the training step on the whole dataset at once, which is a direct computation. Normally we would need to batch the data, but here we can do it in one go as the dataset is not that large.

To familiarize ourselves with the loss function, we can also guess an initial value. What would be an initial value if we don't know anything? 
So if the model didn't learn anything, well, it can't do much better than predicting uniformly over all characters, which will give us a corresponding loss of $\ell = -\log(1/27) \approx 3.3$. If the model performs significantly better than that, we learned something. 

```python
def sample(weights: Weights, key: jax.Array, vocab: List[str]) -> str:
    """
    1) Start with <eos>
    2) Index into the weights matrix W for the current character
    3) Sample the next character from the distribution
    4) Append the sampled character to the sampled word
    5) Repeat steps 3-5 until <eos> is sampled
    6) Return the sampled word
    """
    sampled_word = ['<eos>']
    current_char = jax.numpy.array([vocab.index('<eos>')])
    while True:
        key, subkey = jax.random.split(key)
        logits = forward(weights, current_char, return_logits=True)[0]
        next_char = jax.random.categorical(subkey, logits)
        next_char_int = int(next_char)
        sampled_word.append(vocab[next_char_int])
        if next_char_int == vocab.index('<eos>'):
            break
        current_char = jax.numpy.array([next_char_int])
    return ''.join(sampled_word[1:-1])  # Remove start and end <eos> tokens
```
Lastly, after we trained the model, we would like to generate text with it. That we can do as follows. For now, we just start with an \<eos\> token to indicate that a sequence is starting. Then we get the corresponding logits. We can use then the JAX functionality to sample the next token according to the probability distribution we learned during training.

Using all the previously defined functions, let us now come to the point and perform one training loop with 100 epochs and then compare the generations we obtain with untrained weights to those after the training. 

```python
if __name__ == "__main__":
    words, vocab = load_data('names.txt')
    encoded_words = [encode(word, vocab) for word in words]
    print(f"Encoding from {words[0]} -> {encoded_words[0]}")
    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(encoded_words)
    print("Built train, validation and test sets")
    print(f"# training examples: {len(X_train)}")
    print(f"# validation examples: {len(X_val)}")
    print(f"# test examples: {len(X_test)}")
    weights = init_weights(len(vocab))
    trained_weights = train(weights, X_train, y_train, X_val, y_val, 50, 100)
    print("Sanity check: Compare words generated from trained_weights and untrained_weights")
    for i in range(10):
        key = jax.random.PRNGKey(i)
        print(f"word from untrained weights: {sample(weights, key, vocab)}")
        key, subkey = jax.random.split(key)
        print(f"word from trained weights: {sample(trained_weights, key, vocab)}")
        print("#"*30)
```
The output is as follows: 

```
number of examples in dataset: 32033
max word length: 15
min word length: 2
unique characters in dataset: 27
vocabulary:
<eos> a b c d e f g h i j k l m n o p q r s t u v w x y z
example for a word:
emma
Encoding from emma -> [0, 5, 13, 13, 1, 0]
Built train, validation and test sets
# training examples: 182496
# validation examples: 22819
# test examples: 22831
epoch: 0, loss: 3.859163522720337, val_loss: 3.41965913772583
epoch: 10, loss: 2.6667892932891846, val_loss: 2.6528022289276123
epoch: 20, loss: 2.5551438331604004, val_loss: 2.560276508331299
epoch: 30, loss: 2.520414113998413, val_loss: 2.529088020324707
epoch: 40, loss: 2.5032033920288086, val_loss: 2.5129573345184326
epoch: 50, loss: 2.4927709102630615, val_loss: 2.5030009746551514
epoch: 60, loss: 2.4857537746429443, val_loss: 2.4962339401245117
epoch: 70, loss: 2.480703115463257, val_loss: 2.491330146789551
epoch: 80, loss: 2.4768946170806885, val_loss: 2.4876201152801514
epoch: 90, loss: 2.4739294052124023, val_loss: 2.4847280979156494
Sanity check: Compare words generated from trained_weights and untrained_weights
word from untrained weights: 
word from trained weights: krnen
##############################
word from untrained weights: zzphk
word from trained weights: za
##############################
word from untrained weights: pd
word from trained weights: ri
##############################
word from untrained weights: rzuedlukcrkhxkjvbjtfghkfmgmpw
word from trained weights: zueel
##############################
word from untrained weights: rkaffablgibhnxaw
word from trained weights: sa
##############################
word from untrained weights: cx
word from trained weights: col
##############################
word from untrained weights: mlvbhkxodr
word from trained weights: layn
##############################
word from untrained weights: egqcg
word from trained weights: oqca
##############################
word from untrained weights: gvaqbnxjtdcgopvxiwgdfghjtfhyiioblujhxbsmafgswnbibbpstitisqnbsvmjjxlujeyffg
word from trained weights: vamanijadayn
##############################
word from untrained weights: tinxeailuvvxxbksftt
word from trained weights: menesillvelan
##############################
```
We can observe a few things:
1) Initially the loss is worse than what we would obtain when the probability distribution would be uniform for each character.
2) The model is able to learn which we see in decreasing loss on training and validation set. 
3) The generations we obtain from the trained weights are more coherent than the untrained ones but still not valid names, which is due to the fact that the bigram is still after all a limited kind of model. 

# Conclusion & Outlook
In this blog post, we learned how to implement a simple bigram model to generate natural language. We used JAX to build the model and gained some insights into constructing a basic dataset at the character level. 
Given the simplicity of the model, it was clear from the beginning that we would not achieve state-of-the-art results. Nevertheless, it serves as a helpful exercise to understand the fundamentals.
Next to obvious improvements we could make in modeling or tokenization, another point is that JAX is highly capable of serving as a framework for parallel computations which we didn't implement here as well. I leave this as an exercise for the reader. For further information on this you might find older blog posts by me interesting. 

This blogpost is heavily inspired by the excellent [lecture by Andrej Karpathy](https://www.youtube.com/watch?v=PaCmpygFfXo). I highly encourage you to check it out for a more detailed and longer explanation of the bigram model.
The experiments were performed on a TPU provided by [TRC research program](https://sites.research.google/trc/about/).
If you have questions or suggestions, please let me know. If you are interested in the code, you can find it at [my gitub](https://github.com/simveit/bigram_jax/tree/main).