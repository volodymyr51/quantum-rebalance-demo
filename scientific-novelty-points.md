## Short:
To our knowledge, this work is among the first reproducible demonstrations of a transaction-cost-aware, dynamically re-optimized hybrid quantum-classical portfolio policy that operates in a rolling rebalancing setting and is validated under both controlled regime-switching simulations and real market data

## Verbose:
Online hybrid rebalancing architecture
We formulate portfolio allocation as a sequential decision process in which a variational quantum circuit is re-optimized at each rebalance date using rolling market information, rather than solving a single static allocation problem.

Turnover-aware quantum policy learning
We integrate expected return, risk penalty, and transaction-cost (turnover) penalty into one utility objective used during parameter search, so the learned policy explicitly balances performance and trading frictions.

Quantum-probability-to-portfolio mapping with feasibility guarantees
We introduce a practical mapping from qubit-state probabilities to long-only portfolio weights that always remain normalized and implementable in real portfolio constraints.

Unified empirical workflow across synthetic and real data
We evaluate the same algorithmic pipeline in both regime-switching simulated markets and real historical market data, enabling controlled stress testing and practical validation within one framework.

Reproducible research artifact for quantum finance practice
We provide a compact, end-to-end implementation (algorithm, circuit construction, optimization loop, and reporting workflow) designed for transparent replication and extension by other researchers.

##For the field overall:

There are theoretical speed-up results for specific quantum subroutines under strong assumptions.
For end-to-end dynamic portfolio rebalancing with realistic constraints, there is still no widely accepted, practical, reproducible quantum advantage over strong classical baselines.
So this area is not fully explored, but speed-up evidence is still immature.