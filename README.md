a coarse understanding of the problem

- highest-level abstraction: a universal model/language for knowledge/structure representation
  - assumption: human as a species still evolving on earth (except astronauts working in international space station), there are some high-level consensus that holds true for almost all individuals, thus can be learned from all sources of truth indifferently
    - physical laws / laws of the currently observable universe centered around earth
    - predicate logic
      , an oversimplification of evolution of the universe (or our minds as "micro universes") where continuum is non-exist (with nothing to fill in the gap between True and False)
      , and an manifestation of inductively accumulated effect of physical laws
- mid-level abstraction: domain-specific language (vague, need to commit to an ontological understanding of the universe / a tree-structured taxonomy of all domains, mostly works because there are a few already in academia)
  - formal and symbolic systems which are structured closely to the committed domain
  - natural languages: domain-specific vocabulary/terminologies, the way people structure their sentences could vary across domains especially in written literatures
  - might need to separate different media (fiction vs non-fiction, news vs literatures, etc.)
- low-level abstraction: natural language
  - certain degree of consensus across all domains syntactically (the saying "I understand every single word in the sentence but I don't understand what he's talking about", syntactical structures captured, but the higher-order semantics/knowledge not)



Ultimate Goal

- machines being able to extract higher-level abstraction from written sources

# Idea

A algebra functor is a coproduct of a set of commands/events which name the edges in a state machine but this state machine at type level is fully connected (TODO: encode a FSM into the type system so certain sequences will be rejected thus no need to have strict parent-child binding).
The Free construct abstract way the recursive unfolding of that state machine to create a potentially infinite sequence of commands/events.
Unlike the least fixed-point of functor (`Mu`/`Fix`), `Free` construct has a termination branch (`Pure`, termination of the computation).
Then the program is like translating from one DSL to another where the end point is an IO monad that executes side effects to talk to the external world.
Given any state of the machine, the unfolding of the machine is a tree which encodes all the possible evolution pathways of the system.
Interleaving these DSLs in layers of the tree is like internal communication between independent components of the system where the command/event sequence encodes the event cascading pathway.
It's kind of similar to a decision tree in which you make a decision about one dimension of the system/state vector at each layer but the DSL tree may come to the same dimension multiple times depending how the state transition logic is organized around these dimensions (firstly, need to commit to a state space partition).

Vision-related neural networks have a couple of good assumptions about the structures of the problem space baked in, like convolution and pooling, which restrict the number of cascading pathways to a point that the possible structures expressible by the network is close "enough" to the acceptable solutions and navigating the search/hypothesis space is computationally affordable.
Thus, regulating the architecture of programs and figuring out some essential assumptions/primitives around the problem space is the first step towards a supervised formalism of automated event-driven programming.

similarity
- from a sparse data set to a full space representation: in the spec, edge cases are formally described (which can later be used as unit tests), the task of programmers is to close the boundaries of valid state subspace and filling the gap among valid states by state transition functions so that the valid state subspace is fully able to navigate.

difficulty
- the data set will be extremely small which means low noise tolerance thus the size of the hypothesis space has to be strictly restricted

## State Formulation

- deterministic: `State -> State`
- probabilistic: `State -> { State0: 0.7 , State1: 0.2, ... }`

```
current_State (of the entire universe) -> next_State (of the entire universe)

-- event-driven formulation models the interaction between two systems:
-- System1 (internal), System2 (external, or the rest of the world)

current_System1_State x current_System2_State -> next_System1_State x next_System2_State

-- model the communication process between two systems
-- basic case: full state synchronization
-- each system has its own state directly accessible
-- but need explicit synchronization step to get the other system's newest state
-- old_System2_State is System1's latest understanding of the state of System2 before the next synchronization step
-- similar for old_System1_State
(current_System1_State x old_System2_State) x (old_System1_State x current_System2_State)
->
(next_System1_State x current_System2_State) x (current_System1_State, next_System2_State)

-- model the alternating communication process
1. (current_System1_State x old_System2_State) x current_System2_State
-> (next_System1_State x current_System2_State) x next_System1_State
2. (old_System1_State x current_System2_State) x next_System1_State
-> (new_System1_State x next_System2_State) x next_System2_State
3. loop

-- full state sychronization is not practical
-- in reality, both systems only get to observe partial information about the other system's state
-- thus, need a decoding step to reconstruct the full state
-- the encoding step is mandatory in this model
-- in addition to previous model
1. Input(from System2) --decode-> current_System2_State,
   next_System1_State --encode-> Output(to System2)
2. Input(from System1) --decode-> next_System1_State,
   next_System1_State --encode-> Output(to System1)
```

decoder example
- [3D-R2N2: 3D Recurrent Reconstruction Neural Network - Stanford](http://3d-r2n2.stanford.edu/)
reconstruct the 3d model of the object given its 2d observations (image)
- [Neural scene representation and rendering -- Deepmind](https://deepmind.com/blog/neural-scene-representation-and-rendering/)


time-series prediction only has observation of the Output sequences
thus, it's a degenerated scenario like a monologue
(the system still could have model of the rest of the world but trivial because it's not expecting any feedback.
the update of the understanding of the external world is limited to "the external world received all my words up to this point")

to learn the mealy machine behind a event-driven system,
- need to observe the Input sequence as well
- or use generative model with "reasonable" assumptions

# Reference

## Factorial Hidden Markov Model

### 1.[Factorial Hidden Markov Models - Columbia EE](http://www.ee.columbia.edu/~sfchang/course/svia-F03/papers/factorial-HMM-97.pdf)

### 2.[Supertagging with Factorial Hidden Markov Models](http://www.aclweb.org/anthology/Y09-2043)
