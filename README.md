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
