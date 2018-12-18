### Final Topics

The final exam will take place on Tuesday Dec 18, 7:10pm-9:10pm (120min) in 451 Computer Science (regular classroom).

Please note that the final exam is cumulative, so any topics  from before the midterm are fair game. In particular, pay attention to  topics that were _not_ tested on the midterm.

While only the content in the uploaded lecture notes is expected for  the final, textbook references and links to other background material  are provided for each of the topics.

(*) indicates that you should be familiar with the basic concept, but details will not be tested on the exam.

**General Linguistics Concepts** (Parts of J&M 2nd ed. Ch 1 but mostly split over different chapters in J&M. Ch.1 not yet available in 3rd ed.)

- Levels of linguistic representation: phonetics/phonology, morphology, syntax, semantics, pragmatics
- Ambiguity, know some examples in syntax and semantics, including PP attachment, noun-noun compounds.
  - **important** types of ambiguity
- Garden-path sentences.
- Type/Token distinction.
- Know the following terms: sentence, utterance, word form, stem, lemma, lexeme.
  - 02-11
- Parts of speech:
  - know the 9 traditional POS and some of the Penn Treebank tags
    - 04-7, 04-11
    - noun (NN)
    - pronoun (P)
    - determiner (DT)
    - **adjective (JJ)**
    - verb (VB)
    - **adverb (RB)**
    - **preposition (IN)**
    - conjunction (C)
    - **interjection (UH)**
- Types of Linguistic Theories (Prescriptive, Descriptive, Explanatory)
  - 05-6~7
- Syntax:
  - Constituency and Recursion. Constituency tests.
  - Dependency.
  - Grammatical Relations.
  - Subcategorization / Valency (and relationship to semantic roles).
  - Long-distance dependencies.
  - Syntactic heads (connection between dependency and constituency structure).
  - Center embeddings.
    - 05-43~47
    - cannot be expressed by regular language, need context-free language
  - Dependency syntax:
    - Head, dependent
    - dependency relations and labels
    - projectivity
  - Agreement.

**Text Processing** (Split over different chapters in J&M. Parts of [J&M 3rd ed. Ch. 6 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/6.pdf))

- Tokenization (word segmentation).
- Sentence splitting.
- Lemmatization.
  - 02-14
- Know why these are useful and challenging.

**Probability Background**

- Prior vs. conditional probability.
- Sample space, basic outcomes
- Probability distribution
- Events
- Random variables
- Bayes' rule
- conditional independence
- discriminative vs. generative models
  - latent variables affecting observable variables
- Noisy channel model.
- Calculating with probabilities in log space.

**Text Classification (**[J&M 3rd ed. Ch. 6 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/6.pdf)**)**

- Task definition and applications
- Document representation: Set/Bag-of-words, vector space model
- Naive Bayes' and independence assumptions.

**Language Models (**J&M 2nd ed. Ch 4.1-4.8, [J&M 3rd ed. Ch 4 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/4.pdf)**)**

- Task definition and applications
- Probability of the next word vs. probability of a sentence
- Markov independence assumption.
- n-gram language models.
- Role of the END marker.
- Estimating ngram probabilities from a corpus:
  - Maximum Likelihood Estimates
  - Dealing with Unseen Tokens
  - Smoothing and Back-off:
    - Additive Smoothing
    - Discounting
    - Linear Interpolation
    - Katz' Backoff
- Perplexity

**Sequence Labeling (POS tagging) (**J&M 2nd ed Ch 5.1-5.5, [J&M 3rd ed. Ch 10.1-10.4 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/10.pdf)**)**

- Linguistic tests for part of speech.
- Hidden Markov Model:
  - Observations (sequence of tokens)
  - Hidden states (sequence of part of speech tags)
  - Transition probabilities, emission probabilities
  - Markov chain
  - Three tasks on HMMS: Decoding, Evaluation, Training
    - Decoding: Find the most likely sequence of tags:
      - Viterbi algorithm (dynamic programming, know the algorithm and data structures involved)
    - Evaluation: Find the probability of a sequence of words
      - Spurious ambiguity: multiple hidden sequences lead to the same observation.
      - Forward algorithm (difference to Viterbi).
    - Training: We only discussed maximum likelihood estimates. There are unsupervised techniques as well. (*)
  - Extending HMMs to trigrams.
- Applying HMMs to other sequence labeling tasks, for example Named Entity Recognition
  - B.I.O. tags for NER.
    - 04-42

**Parsing with Context Free Grammars (**J&M 2nd ed Ch. 12 and 13.1-13.4 and 14.1-14.4 and Ch. 16,
​                                                                  [J&M 3rd ed. Ch. 11.1-11.5 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/11.pdf) and [Ch. 12.1-12.2 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/12.pdf) [Earley not covered in 3rd. ed] and [Ch 13.1-13.4 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/13.pdf), [complexity classes not covered in 3rd ed.] )

- Tree representation of constituency structure. Phrase labels.
- CFG definition: terminals, nonterminals, start symbol, productions
- derivations and language of a CFG.
- derivation trees (vs. derived string).
- Regular grammars (and know that these are equivalent to finite state  machines and regular expressions - you don't need to know FSAs and  regular expression in detail for the midterm).
- Complexity classes.
  - "Chomsky hierarchy"
  - Center embeddings as an example of a non-regular phenomenon.
  - Cross-serial dependencies as an example of a non-context-free phenomenon.
- Probabilitistic context free grammars (PCFG):
  - Maximum likelihood estimates.
  - Treebanks.
- Recognition (membership checking) problem vs. parsing
- Top-down vs. bottom-up parsing.
- CKY parser:
  - bottom-up approach.
  - Chomsky normal form.
  - Dynamic programming algorithm. (know the algorithm and required data structure: CKY parse table). Split position.
  - Backpointers.
  - Parsing with PCFGs (compute tree probabilities or sentence probability)
- Earley parser:
  - Top-down approach.
  - Does not require CNF.
  - Parser state definition. Initial state, goal states.
  - Three operations: Scan, Predict, Complete
  - Dynamic programming algorithm. (know the algorithm and required data structure: parse "Chart" organized by end-position).

**Other Grammar Formalisms**

- Unification Grammar  (J&M 2nd. ed Ch. 15.1-15.3, not covered in 3rd ed.) (*)

  - Feature structures (as Attribute Value Matrix or DAG)
  - Reentrancy in feature structures
  - Unification
  - Unification constraints on grammar rules
  - Know how these are used to enforce agreement

- Lexicalized tree adjoining grammars (not in J&M, supplementary material:

  Abeillé & Rambow (2000): "Tree Adjoining Grammar: An Overview", in Abeillé &

  [Rambow, "Tree Adjoining Grammars", CSLI, U Chicago Press](http://www.cs.columbia.edu/~rambow/papers/intro-only.pdf)) (*)

  - Two types of elementary trees: initial trees and auxiliary trees.
  - Substitution nodes.
  - Foot nodes in auxiliary trees.
  - Adjunction.
  - Derived tree vs. derivation tree
  - Know that TAG is more expressive than CFG in the complexity hierarchy.

- Combinatory Categorial Grammar (CCG)

  - Categories with / and \
  - Forward and backward and backward application.
  - Combinators: forward/backward composition, type raising (*)
  - Relationship to lambda calculus.

- Hyperedge Replacement Grammars (for AMR graphs)  (*)

**Dependency parsing (**Not in J&M 2nd ed., [J&M 3rd ed. Ch 14.1-14.5 (Links to an external site.)Links to an external site.](https://web.stanford.edu/~jurafsky/slp3/14.pdf),
​                                     Supplementary material: [Küber, McDonald, and Nivre (2009): Dependency Parsing, Ch.1 to 4.2 [ebook available through CU library\]](https://clio.columbia.edu/catalog/7851052))

- Grammar based vs. data based.
- Data based approaches:
  - Graph algorithms. vs. transition based
- Transition based dependency parsing:
  - States (configuration): Stack, Buffer, Partial dependency tree
  - Transitions (Arc-standard system): Shift, Left-Arc, Right-Arc
  - Predicting the next transition using discriminative classifiers.
    - Feature definition (address + attribute)
  - Training the parser from a treebank:
    - Oracle transitions from annotated dependency tree.
  - Difference between arc-standard and arc-eager.
- Graph based approaches (*) (only need to be familiar with the basic concepts - no details):
  - Edge-factored model.
  - Compute Maximum Spanning Tree on completely connected graph of all words.
  - Can be done using the Chu-Liu-Edmonds algorithm (not covered in detail).

**Machine Learning** (Some textbook references below. Also take a look at Michael Collins' detailed notes on a variety of topics: <http://www.cs.columbia.edu/~mcollins/notes-spring2018.html>)

- Generative vs. discriminative algorithms

- Supervised learning. Classification vs. regression problems.

- Loss functions: Least squares error. Classification error.

- Training vs. testing error. Overfitting and how to prevent it.

- Linear Models.

  - activation function.
  - perceptron learning algorithm (* i will not ask you to do this on the final)
  - linear separability and the XOR problem.

- Feature functions

- Log-linear / maximum entropy models  (J&M 3rd. ed. ch. 7, J&M 2nd ed. ch. 6.6)

  - Log-likelihood of the model on the training data.
  - Simple gradient ascent.
  - Regularization
  - MEMM (Maximum entropy markov models):
    - POS tagging with MEMMs.

- Feed-forward neural nets  (J&M 3rd. ed. ch. 8, also take a look at

  Yoav Goldberg's book "Neural Network Methods for Natural Language Processing" (Links to an external site.)Links to an external site.

   [available as PDF if you are on the Columbia network] )

  - Multilayer neural nets.
  - Different activation functions (sigmoid, ReLU, tanh)
  - Softmax activation.
  - Input representation options:
    - one-hot representation for features, word embeddings, feature function value.
  - Output representation options:
    - Single score.
    - Probability distribution (softmax)
  - Backpropagation (*,  know the idea of propagating back partial  derivatives of the loss with respect to each weight, but you don't have  to understand the details).

**Formal Lexical Semantics** (J&M 3rd ed. ch 17, J&M 2nd ed. ch 19.1-19.3 and 20.1-20.4 )

- Word senses, lexemes, homonymy, polysemy, metonymy, zeugma.
  - **important** 13-7~11, 15~18
  - homonymy
    - `Lexeme`s with the same `WordForm` but `Sense`s are remote (in some space of `Sense` or `Sentence`)
  - polysemy
    - `Lexeme`s with the same `WordForm` but `Sense`s are close
  - metonomy
    - a subtype of polysemy (?)
      - doesn't seem to have the constraint on `WordForm`
    - containment (whole -> part)
      - "the press" ~ reporters
      - "the White House" ~ the U.S. presidential staff
      - "the Pentagon" ~ the military leadership
    - Synecdoche (part -> whole)
      - "Franklin" ~ $100
    - a physical item -> a related concept
      - "the crown" ~ monarch
      - "stomach" ~ appetite, hunger
    - example 1: MEAT <-> ANIMAL
      - "the chicken" was overcooked ~ chicken MEAT
      - "the chicken" eats a worm ~ ANIMAL


  - zeugma
    - heterogeneous `map`
    - example: "Does United serve breakfast and JFK?"
      - `serves := (S\NP)/NP : \x.\y. Serves(y,x)`
        - `serves :: Any -> Any -> Boolean`
      - `breakfast := NP : \x. Breakfast(x)`
        - `Breakfast :: Any -> Boolean`
        - embellish any entity, `x`, that fulfills `Breakfast` predicate/constraint a type `x :: Food`
      - `jfk := NP : \x. JFK(x)`
        - `JFK :: Any -> Boolean`
        - embellish any entity, `x`, that fulfills `JFK` predicate/constraint a type `x :: Airport`
      - `and` is similar to product type `Tuple :: * -> * -> *`
      - `(breakfast and JFK) :: Tuple Breakfast JFK`
      - implicitly map `serve` over the `Tuple` (heterogeneous because two types under `Tuple` are different)
      - `(serve breakfast and JFK) :: S\NP : \y. exists x1 :: Food. exists x2 :: Airport. Serves(y, x1) ^ Serves(y, x2)`
        - first `Serves :: Food -> Any -> Boolean`
        - second `Serves :: Airport -> Any -> Boolean`
  - Synonym
    - close in `Sense` space
    - without the extra constraint on `WordForm` in polysemy
  - Antonyms
    - opposite/remote in one feature/dimension
  - is-a (related to entailment): Hyponymy / Hypernymy
  - part-of: Meronymy / Holonymy

  

```haskell
type Word = String
type Sentence = Array Word
type WordForm = Word

type Sense = Sentence -- we can further identify the Sense of each Word in a Sentence
type Lexeme = (WordForm, Sense)
```

- WordNet, synsets
  - It groups English words into sets of synonyms called synsets, provides short definitions and usage examples, and records a number of relations among these synonym sets or their members.
  - All synsets are connected to other synsets by means of semantic relations. 
    - Nouns
      - is-a: hypernym, hyponym
      - part-of: meronym, holonym 
    - Verbs
      - is-a: hypernym, hyponym
      - ( troponym, entailment )
- lexical relations (in WordNet)
  - synonym, antonym
  - hypernym, hyponym
    - is-a
  - meronym, holonym
    - part-of
- Word-sense disambiguation:
  - Supervised learning approach and useful features.
    - bag-of-word features
      - multi-hot encoding
    - collocational features
      - feature function:
        - Input: a sentence and its corresponding POS tag sequence
        - apply a fixed-length window filter on both
        - zip
    - concatenate/project word-embedding vectors into a phrase-embedding vector
      - projection layer
        - maps the discrete word indices of an n-gram context to a continuous vector space
  - (dictionary-based) Lesk algorithm (J&M 3rd ed. ch. 17.6)
    - simplified algorithm
      - Given: `lemma :: Word`, `context :: Array Word`
      - `lemma` -> `Set Sense`
        - `gloss :: Set Word`, sentence description and examples of the given `lemma`
        - `synset :: Set Word`, a set of synonyms surrounding a `Sense`
      - count word overlap between `synset`/`gloss` of each `Sense` and `context`
    - improvements
      - include example sentences
      - add glosses from related words
        - hypernyms
        - meronyms
      - Word2Vec, word vector distance

  - Bootstrapping approach (*) (don't have to know the details, J&M 3rd ed. ch.  17.8).
- Lexical substitution task
  - need a (probabilistic) language model to evaluate the likelihood of a modified sentence still being "valid"
    - faithfulness
    - fluency

**Distributional (Vector-based) Lexical Semantics** (J&M 3rd ed. ch 15 & 16, not in 2nd ed.)

- Distributional hypothesis
  - 12-09
- Co-occurence matrix
- Distance/Similarity metrics (euclidean distance, cosine similarity)
- Dimensions (parameters)of Distributional Semantic Models
  - Preprocessing, term definition, context definition, feature  weighting, normalization, dimensionality reduction, similarity/distance  measure
- Semantic similarity and relatedness (paradigmatic vs. syntagmatic relatedness)
  - 12-20
  - Effect of context size on type of relatedness.
    - (small -> large) ~ (syntagmatic-related -> paradigmatic-related)
- Term weighting (*) (not discussed in detail):
  - TF*IDF
- Sparse vs. Dense Vectors
- One-hot representation (of words, word-senses, features)
- Word Embeddings
- Word2Vec embeddings using a neural network.
  - Skip-gram model
    - Input: one-hot encoding of a single word
    - Output: context (a fixed-length window around the word)
      - for each position in the context, emit a probability distribution over all words
    - Error:
  - CBOW (*)

**Semantic Role Labeling** (J&M Ch. 22)

- Event
  - Frame in FrameNet
  - Frameset in PropBank

- Semantic Role
  - Frame Element in FrameNet
  - Numbered/Indexed Arguments in PropBank

- Frame Semantics
  - Frame, Frame Elements (specific to each frame)
    - Frame
      - evoke a frame by presence of a word/pharse
        - GIVING(frame) -> { donate, hand over, ... }
    - Frame Element
      - Core vs Non-core
      - can be treated as a predicate
        - \x. Doner(x)
        - \x. Theme(x)
        - \x. Receipient(x)
  - Valence Patterns
    - assigning an additional grammatical role (GF) to each frame element (FE)
      - Frame Element(FE)
      - Grammatical Function(GF)
      - Phrase Type(PT), composition of Part-of-speech (POS) tags on individual words
        - NP -> subj, obj, obj2
        - PPof (of NP) -> `dep-of`
        - PPto (to NP) -> `dep-to`
  - FrameNet:
    - Frame definitions
    - Lexical Units
    - Example annotations to illustrate valence patterns.
      - John(subj/DONOR) gave(V) Mary(obj/RECIPIENT) the book(obj2/THEME)
    - Frame-to-frame relations and frame-element relations (*) (you do not need to remember an exhaustive list of these relations).
      - 14-8
      - inherits
        - frame element to frame element mapping/alignment
      - perspective on
      - subframe of
  - PropBank:
    - Differences to FrameNet.
    - Semantic Roles (ARGx)
      - Numbered/Indexed Arguments
        - each index is bounded by a set of grammatical functions
          - Proto-Agent (`Arg0`)
          - Proto-Patient (`Arg1`)
          - Arg2 ~ Arg5 are not strictly consistent leaving room for ambiguity
            - parsing might not be unique and relies on additional parse rules
    - predicate-argument structure.
    - framesets
      - each corresponds a distinct sense of a lemma
      - allow only a subset of Arguments present
      - each predicate argument consists of an argument and a predicate
        - Arg0 (indexed argument): causer of increase (predicate)
  - Semantic Role Labeling / Frame Semantic Parsing:
    - Steps of the general syntax-based approach
      - Target identification, semantic role/frame element identification and labeling.
      - Input: CFG parse tree of a raw sentence
      - Output: Frame / Frameset
    - Features for semantic role role/FE identification and labeling. (*)  (you do not need to remember an exhaustive list, but have a general  sense of which features are important).
      - Selectional restrictions and preferences.
      - Parse-tree path features

**Semantic Parsing (full-sentence semantic analysis)** (J&M 2nd ed. Ch 17, not in 3rd ed.)

- Goals for meaning representations (unambiguous, canonical form, supports inference, expressiveness).
- First-order logic (aka Predicate Logic):
  - Syntax of FOL: Constants, Functions, Terms, Quantifiers, Variables, Connectives
  - Semantics for FOL: Model-theoretic sematnics.
  - Event logical (Neo-Davidsonian) representation.
- Semantic analysis with first-order logic:
  - Principle of compositionality (and examples for non-compositional phenonema)
  - Lambda expressions.
  - Function application.
  - Higher-order functions.
  - Types.
- Categorical Grammar and Combinatory Catagorical Grammar (CCG)  (*)

**Abstract Meaning Representation** (not in textbook, planned for 3rd edition)

- meaning of vertices (entities) and edges (relations) (* you do not  have to remember specific special relations and concepts in the AMR  annotation guidelines).
  - vertex
    - a variable
    - a predicate
      - root and branch: frameset predicate
      - leaf node: named entity predicate
  - edge (from one vertex to one vertex, no hyperedge)
    - a binary predicate with two variables from both vertices
    - Indexed Arguments in Frameset from PropBank
      - `:ARG0`, `:ARG1-of` (inverse), etc.
    - constants
      - `:quant`, `:polarity`
    - non-core roles
      - `:time`, `:location`, etc.
- reentrancy
  - inverse relations
    - from
      - (e/ eat-01 :ARG0 **(d / dog)** :ARG1 (b / bone))
      - **(f/ find-01 :ARG0 d :ARG1 b)**
    - to
      - ... **(d / dog :ARG1-of (f / find-01 : ARG0 d))** ...
- ARG-of edges (inverse relations)
  - normalize a DAG into a tree
- constants
  - `:quant`, quantity
  - `:polarity`
- relation to event logic (*)
  - 16-13
    - PropBank
      - want-01
        - ARG0: prop-agent description
        - ARG1: prop-patient description
      - eat-01
        - ...
    - AMR
      - (w / want-01 :ARG0 (d / dog) :ARG1(e / eat-01 :ARG0 d :ARG1 (b / bone )))
    - Event Logic
      - exists w, e, d, b.
      - want-01(w)
      - and ARG0(w, d) and ARG1(w, e)
      - and eat-01(e)
      - and ARG0(e, d) and ARG1(e, b)
      - and dog(d)
      - and bone(b)
- AMR parsing (*)
  - JAMR approach (*)
  - Hyperedge Replacement Grammar (HRG) approach (*)

**Machine Translation (MT****)** (J&M 2nd ed. Ch 25)

- Challenges for MT, word order, lexical divergence
- Vauquois triangle.
- Faithfulness vs. Fluency

**Statistical MT** (J&M 2nd ed. Ch, also see Michael Collins' notes on IBM M2 here: <http://www.cs.columbia.edu/~mcollins/ibm12.pdf> )

- Parallel Corpora

- Noisy Channel Model for MT
  - latent variables govern the (conditional) probability distribution of observable variables
- Word alignments
- IBM Model 2:
  - Alignment variables and model definition.
    - two fixed-length sentences
      - source `{e_1 ... e_l}`
      - target `{f_1 ... f_m}`
    - `i`th position in target `f` to `j` th position in source `e`
  - EM training for Model 2
    - **important**
- Phrase-based MT (*)
- MT Evaluation:
  - BLEU Score
    - **important**
    - BP
      - penalty on length if shorter than reference translation
      - 1 if `c > r`
      - `e^{1- r/c}` if `c <= r`
        - `in (0, 1)`
    - `(a1 x a2 x ... x an)^{1/n}`
      - average performance on n-gram

**Recurrent Neural Nets**

- Recurrent neural nets.
  - Neural language model (without RNNs)
  - Basic RNN concept, hidden layer as state representation.
  - Common usage patterns:
    - Acceptor / Encoder
      - the last output (`y5`) is the output of the network
      - only compute loss for the final output and backprop to recompute weights
        - weights for all RNN units are shared
    - Transducer
      - Transducer as generator
        - offset by 1
        - aggregate errors for prediction of each word in the sentence
      - Conditioned transduction (encoder-decoder)
        - append additional context vector to each word (possibly a vector from some word embeddings)

  - Backpropagtion through time (BPTT) (*)
  - LSTMs (*)
- Neural MT
  - Attention Mechanisms (*)
    - 18-20~21
    - assume the context upon a given word (`h_f3`)represent that word (`verde`)
    - attention weight measures the correlation between contexts (`h_t` and `h_s`)
