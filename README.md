# MCMC Graph Editor

An interactive **Markov-chain / state-machine editor and simulator** built with [Dear PyGui](https://github.com/hoffstadt/DearPyGui).  
Create states (nodes), connect them with weighted directed edges (probabilities), run single or batched simulations, and inspect how often each state is visited.

---

## Features

- **Drag-and-drop graph editing**
  - Add/delete/rename nodes
  - Add/update/delete directed edges
  - Self-loops supported
  - Automatic arrow **offset** for reverse edges (`arrow_gap`) so parallel arrows don’t overlap
- **Save / Load** projects to a simple JSON format
- **Single run simulation**
  - Choose start node, steps, and an optional seed
  - View per-state visits and visit frequency
- **Monte-Carlo (multi-run) simulation**
  - Run `M` simulations of `N` steps each
  - Pick a **target node** and see:
    - Average visits & frequency
    - Standard deviation (visits & frequency)
    - Probability the node is hit at least once
  - Table of average visits/frequency for **all** nodes
- Clean, single-file implementation; no external model dependencies.