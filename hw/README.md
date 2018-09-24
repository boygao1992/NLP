# Lecture 03

## Smoothing - Discounting

given a lexicon,
assign small transitional probability to the unseen words in the **training set**
- evenly
- proportional to the unigram counts

```
lexicon = { the, cat, dog, peacock }

-- | in the training set
-- unigrams
count(the) = 4 (= count(the *))
count(dog) = 6
count(peacock) = 2
count(the) : count(dog) : count(peacock) = 2 : 3 : 1
-- bigrams
count(the cat) = 4 -- "cat" is the only possible word following "the"
count(the the) = 0
count(the dog) = 0
count(the peacock) = 0

-- bigram distribution (given "the") before smoothing
P[cat | the] = 1
P[the | the] = 0
P[dog | the] = 0
P[peacock | the] = 0

-- discounting
beta = 1

count*(the cat) = 4 - 1 = 3
P*[cat | the] = 3 / 4

alpha(the) = 1 - P*[cat | the] = 1 / 4

-- bigram distribution (given "the") after smoothing
P*[the | the] = alpha(the) * (2 / 6) = 1 / 12
P*[dog | the] = alpha(the) * (3 / 6) = 1 / 8
P*[peacock | the] = alpha(the) * (1 / 6) = 1 /24

```
