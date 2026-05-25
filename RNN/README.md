# RNN Paper Replication from Scratch

This repository is a beginner-friendly paper replication of the classic RNN paper:

**Paper:** *Finding Structure in Time*  
**Author:** Jeffrey L. Elman  
**Year:** 1990  

The paper studies how neural networks can process information that comes in a sequence over time. Instead of representing time explicitly as fixed input positions, Elman proposed a simple recurrent architecture where the network keeps memory through its previous hidden state.

This project implements a simplified replication of the main experiments from the paper using **PyTorch from scratch**, without using `nn.RNN`, `nn.LSTM`, or `nn.GRU`.

---

## About the Paper

Many real-world tasks happen over time:

- language
- planning
- actions
- speech
- sequential decision-making

Traditional feedforward neural networks process fixed-size inputs all at once. But sequential data arrives step by step.

The paper asks:

> How can a neural network represent time and memory?

Elman's solution was to use a recurrent network where the hidden state from the previous time step is fed back into the network at the next time step.

The main idea is:

```text
current input + previous hidden state → new hidden state
```

Mathematically:

```text
h_t = tanh(Wx * x_t + Wh * h_(t-1) + b)
```

Output:

```text
y_t = Wo * h_t + b_o
```

---

## Repository Goal

The goal of this repository is to understand and implement the Elman Simple Recurrent Network from scratch.

This project focuses on:

- understanding the paper step by step
- implementing the RNN cell manually
- avoiding PyTorch built-in recurrent layers
- reproducing the major experimental ideas
- visualizing error curves and hidden-state representations

---

## What is Implemented?

The following experiments are implemented:

| Experiment | Paper Section | Task |
|---|---|---|
| Experiment 1 | Exclusive-OR | Sequential XOR prediction |
| Experiment 2 | Structure in Letter Sequences | Predict letters from artificial sequences |
| Experiment 3 | Discovering the Notion "Word" | Predict characters from continuous text without spaces |
| Experiment 4 | Discovering Lexical Classes from Word Order | Predict next word and analyze hidden representations |

---

## Main Architecture

The implemented model follows the Elman RNN idea.

```text
                   Input x_t
                       ↓
              Input-to-hidden layer
                       ↓
           Previous hidden state h_(t-1)
                       ↓
              Hidden-to-hidden layer
                       ↓
            input_part + hidden_part
                       ↓
                      tanh
                       ↓
               New hidden state h_t
                       ↓
                 Output layer
                       ↓
                  Prediction y_t
```

For a sequence:

```text
x1 + h0 → h1 → y1
x2 + h1 → h2 → y2
x3 + h2 → h3 → y3
x4 + h3 → h4 → y4
...
```

---

## Important Note

This is a **simplified educational replication**, not an exact historical reproduction.

Differences from the original paper:

- PyTorch autograd is used for backpropagation.
- The recurrent cell is implemented manually.
- `nn.RNN`, `nn.LSTM`, and `nn.GRU` are not used.
- Some datasets are simplified versions of the paper's simulations.
- For some experiments, one-hot vectors are used instead of the exact feature vectors from the paper.

---

## Project Structure

```text
RNN-Paper-Replication/
│
├── Notebooks/
│   ├── 01_sequential_xor.ipynb
│   ├── 02_letter_sequence.ipynb
│   ├── 03_word_discovery.ipynb
│   └── 04_lexical_classes.ipynb
│
├── Images/
│   ├── exp1_xor_loss.png
│   ├── exp1_cycle_error.png
│   ├── exp2_letter_loss.png
│   ├── exp2_rms_error.png
│   ├── exp2_feature_error.png
│   ├── exp3_char_loss.png
│   ├── exp3_letter_word_error.png
│   ├── exp4_word_loss.png
│   ├── exp4_pca_words.png
│   └── exp4_dendrogram.png
│
├── README.md
└── requirements.txt
```

---

# Experiment 1: Sequential XOR

## Problem

Normal XOR receives two inputs together:

```text
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0
```

But in the paper, XOR is converted into a temporal sequence.

Example:

```text
1 0 → 1
0 0 → 0
1 1 → 0
```

The continuous sequence becomes:

```text
1 0 1 0 0 0 1 1 0 ...
```

The task is:

```text
current bit → predict next bit
```

---

## Parameters

| Parameter | Value |
|---|---|
| Input size | 1 |
| Output size | 1 |
| Hidden size | 8 or 20 |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Sequence length | 20 |
| Batch size | 16 |
| Epochs | 20–50 |

---

## Model Output Shape

```text
Input:
[batch_size, seq_len]

Output:
[batch_size, seq_len, 1]
```

---

## Result

The model learns to use previous hidden states to predict the next bit.

Training loss generally decreases over epochs.

![Experiment 1 Training Loss](assets/exp1_xor_loss.png)

---

## Cycle vs Error

The original paper shows an up-down error pattern because some points in the XOR sequence are predictable and some are uncertain.

![Experiment 1 Cycle Error](assets/exp1_cycle_error.png)

---

## Observation

The RNN learns temporal dependency because the current prediction depends not only on the current bit but also on previous bits stored in the hidden state.

---

# Experiment 2: Structure in Letter Sequences

## Problem

The second experiment checks whether the RNN can learn more complex sequential patterns than XOR.

The artificial grammar is:

```text
b → ba
d → dii
g → guuu
```

Example:

```text
d b g
```

becomes:

```text
d i i b a g u u u
```

The task is:

```text
current letter → predict next letter
```

---

## Vocabulary

| Letter | Index |
|---|---|
| b | 0 |
| d | 1 |
| g | 2 |
| a | 3 |
| i | 4 |
| u | 5 |

---

## Parameters

| Parameter | Value |
|---|---|
| Input size | 6 |
| Output size | 6 |
| Hidden size | 20 |
| Loss | `CrossEntropyLoss` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Sequence length | 20 |
| Batch size | 16 |
| Epochs | 100–150 |
| Random consonants | 1000 |

---

## Model Output Shape

```text
Input:
[batch_size, seq_len, 6]

Output:
[batch_size, seq_len, 6]

Target:
[batch_size, seq_len]
```

---

## Result

The model learns patterns like:

```text
b → a
d → i → i
g → u → u → u
```

Training loss decreases as the network learns the letter structure.

![Experiment 2 Training Loss](assets/exp2_letter_loss.png)

---

## RMS Error Over Full Output Vector

This plot corresponds to the paper's error graph for the letter prediction task.

![Experiment 2 RMS Error](assets/exp2_rms_error.png)

---

## Feature-Level Error

The original paper also analyzed error for specific features such as:

- consonantal feature
- high feature

In this implementation, these were approximated using manually defined feature vectors.

![Experiment 2 Feature Error](assets/exp2_feature_error.png)

---

## Observation

The model predicts vowels better than random consonants because vowels follow fixed rules.

For example:

```text
After d, the network expects i
After g, the network expects u
After b, the network expects a
```

But consonants are random, so predicting the next consonant is harder.

---

# Experiment 3: Discovering the Notion "Word"

## Problem

The third experiment checks whether the RNN can find word-like structure in a continuous stream of letters.

The network is given text without spaces.

Example sentence:

```text
boy eats apple
```

The network sees:

```text
boyeatsapple
```

Many such sentences are joined:

```text
boyeatsapplegirlseesdogcatlikesbread...
```

The task is:

```text
current character → predict next character
```

The network is never told where words start or end.

---

## Example Vocabulary

Words used to generate sentences include:

```text
boy, girl, man, woman, dog, cat, child, teacher, student,
bird, lion, mouse, sees, likes, chases, eats, finds,
helps, watches, follows, meets, loves, bread, apple,
cookie, cake, ball, book, toy, fruit, milk, rice
```

---

## Parameters

| Parameter | Value |
|---|---|
| Input size | `vocab_size` |
| Output size | `vocab_size` |
| Hidden size | 20 |
| Loss | `CrossEntropyLoss` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Sequence length | 20 |
| Batch size | 16 |
| Epochs | 10–30 |
| Sentences | 300–1000 |

---

## Model Output Shape

```text
Input:
[batch_size, seq_len, vocab_size]

Output:
[batch_size, seq_len, vocab_size]

Target:
[batch_size, seq_len]
```

---

## Result

The model learns character-level patterns inside words.

Training loss decreases over time.

![Experiment 3 Training Loss](assets/exp3_char_loss.png)

---

## RMS Error in Letter-in-Word Prediction

The paper's important observation is that prediction error tends to be high near the start of a new word and lower inside more predictable parts of words.

![Experiment 3 Letter-in-Word Error](assets/exp3_letter_word_error.png)

---

## Observation

The model is not explicitly given word boundaries.

Still, the prediction error gives clues:

```text
high error → possible new word
lower error → inside a known word pattern
```

This supports the paper's idea that word-like units may be discoverable from sequential structure.

---

# Experiment 4: Discovering Lexical Classes from Word Order

## Problem

The fourth experiment moves from character prediction to word prediction.

The task is:

```text
current word → predict next word
```

The network is not told grammatical categories like:

```text
noun
verb
animal
human
food
object
```

It only learns from word order.

---

## Word Categories

| Category | Examples |
|---|---|
| Human nouns | boy, girl, man, woman, child, teacher, student |
| Animal nouns | dog, cat, bird, lion, mouse |
| Food nouns | bread, apple, cookie, cake, fruit, milk, rice |
| Object nouns | ball, book, toy |
| Perception verbs | sees, watches |
| Action verbs | chases, finds, helps, follows, meets |
| Emotion verbs | likes, loves |
| Eating verbs | eats |
| Determiners | a, the |
| Connectors | and, or |
| Adverbs | today, slowly, quickly |
| Prepositions | near, with |

---

## Sentence Templates

Example templates:

```text
Noun Verb Object

Determiner Noun Verb Determiner Object

Noun and Noun Verb Object

Determiner Noun Verb Determiner Object Adverb

Determiner Noun Verb Determiner Object Preposition Determiner Noun
```

Example generated sentences:

```text
boy eats apple
the girl sees the dog
cat and mouse chase bread
the teacher helps the student today
the boy watches the bird near the tree
```

---

## Parameters

| Parameter | Value |
|---|---|
| Input size | `word_vocab_size` |
| Output size | `word_vocab_size` |
| Hidden size | 20 |
| Loss | `CrossEntropyLoss` |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Sequence length | 20 |
| Batch size | 16 |
| Epochs | 30 |
| Generated sentences | 1000 |

---

## Model Output Shape

```text
Input:
[batch_size, seq_len, word_vocab_size]

Output:
[batch_size, seq_len, word_vocab_size]

Target:
[batch_size, seq_len]
```

---

## Result

The model learns word-order patterns.

Training loss decreases over epochs.

![Experiment 4 Training Loss](assets/exp4_word_loss.png)

---

## PCA Plot of Word Representations

The PCA plot helps check whether similar words are placed near each other in hidden-state space.

![Experiment 4 PCA Word Representations](assets/exp4_pca_words.png)

Expected grouping:

```text
boy, girl, man, woman → close together

dog, cat, lion, mouse → close together

apple, bread, cookie → close together

sees, watches, chases → close together
```

---

## Hierarchical Clustering

The paper used hierarchical clustering to analyze hidden representations.

This implementation also creates a dendrogram.

![Experiment 4 Dendrogram](assets/exp4_dendrogram.png)

---

## Observation

The RNN can learn useful internal representations from word order.

It may group words by how they behave in sentences.

For example:

```text
boy, girl, man, woman
```

often appear in similar positions, so their hidden representations may become similar.

This supports the paper's idea that lexical categories can emerge from sequential structure.

---

## Important Learning from This Replication

This project shows that even a simple RNN can:

- remember previous inputs
- process sequences step by step
- learn temporal structure
- predict sequential patterns
- discover word-like boundaries
- learn category-like word representations
- build meaningful hidden states without explicit symbolic rules

---

## Conclusion

This repository implements a simplified but meaningful replication of Elman's *Finding Structure in Time*.

The main result is that a very simple recurrent network can learn structure from sequences by using its hidden state as memory.

The project demonstrates the central idea of the paper:

```text
Time does not need to be represented as fixed input positions.
Time can be represented implicitly through changing internal states.
```

This is the core intuition behind recurrent neural networks.





BY:- SUBHAM MOHANTY
