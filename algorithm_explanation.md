# Algorithm Overview and Comparison

This document summarizes the hybrid quantum–classical algorithm defined in `main.py` and compares it to a more traditional ("default") classical rebalancing strategy. A short example illustrates how the two approaches behave using the same synthetic market data.

---

## 1. Hybrid Quantum–Classical Algorithm (main.py)

The proof‑of‑concept implemented in `main.py` performs **dynamic portfolio rebalancing** by combining a small variational quantum circuit with classical parameter tuning.

### Key components

1. **Market simulation** (`simulate_market`)
   - Generates synthetic returns for `n_assets` over `n_steps` using regime‑switching drifts and a fixed covariance matrix.

2. **Feature construction** (`build_features`)
   - For each rebalance date the most recent `lookback` returns are combined into a z‑scored signal (momentum relative to mean and volatility).
   - The signal is squeezed into the interval \([-\pi,\pi]\) and used as rotation angles for qubits.

3. **Quantum policy** (`quantum_policy`)
   - Builds a Qiskit `QuantumCircuit` with one qubit per asset.
   - Applies `Ry` rotations by the feature angles, a chain of `CNOT` gates for entanglement, and parameterized `Ry`/`Rz` gates controlled by trainable vector `theta` (length `3 * n_assets`).
   - Executes the circuit with a statevector simulator and converts measurement probabilities of the `|1\rangle` state for each qubit into raw scores.
   - Normalizes the scores to obtain a **long‑only weight vector** summing to one.

4. **Classical optimization** (`optimize_quantum_parameters`)
   - Starting from a center parameter vector `theta_center`, perform a local random search using `random_search_samples` candidates.
   - Each candidate is evaluated by `evaluate_candidate` which computes a utility: expected return minus a variance penalty (`risk_aversion`) and a turnover cost (`transaction_cost`) relative to previous weights.
   - The best candidate is retained and used for the next period.

5. **Rebalancing loop** (`run_hybrid_rebalancing`)
   - Iterates through time steps (from `lookback` to `n_steps`), updating features, optimizing parameters, computing turnover and realized return, and logging results.
   - At the end, prints summary statistics (cumulative return, Sharpe, average turnover, etc.) and the last few weight vectors.

### Characteristics

- **Adaptive**: the quantum circuit is re‑optimized each period based on recent data.
- **Non‑convex optimisation**: the variational circuit can express complex decision boundaries; classical random search compensates for the lack of gradients.
- **Portfolio weights** are implicitly derived from quantum probability amplitudes rather than solved analytically.

---

## 2. Default (Classical) Rebalancing Strategy

For comparison we consider a simple classical algorithm that rebalances to the **mean‑variance optimal weights** each period without any quantum component. The default policy implemented below uses the same simulated returns and features but computes weights analytically:

```python
import numpy as np

def classical_policy(mu: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    # maximize mu^T w - (risk_aversion/2) w^T cov w subject to 1^T w = 1, w >= 0
    inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
    raw = inv @ mu
    raw = np.maximum(raw, 0.0)
    return raw / raw.sum()
```

The overall simulation loop is the same as the hybrid version, but weights are updated via `classical_policy` and there is no parameter search.

### Characteristics of the default algorithm

- **Deterministic**: given estimated `mu` and `cov`, the weights are uniquely determined (up to numerical noise).
- **Convex optimization**: can be solved directly via matrix inversion.
- **No learning rate or parameters**: only depends on estimated statistics and a `risk_aversion` multiplier.

---

## 3. Example Comparison

Instead of a runnable code snippet, the following expressions describe how each algorithm computes the weights and what each symbol represents. Both methods operate on the same simulated return matrix

\[R \in \mathbb{R}^{T \times n},\]

where \(T\) is the number of steps and \(n\) the number of assets. At time \(t\), a rolling window of length \(L\) (the "lookback") produces

\[W_t = R_{t-L:t} \in \mathbb{R}^{L\times n}.\]

Define

- \(\mu_t = \frac{1}{L}W_t^{\top} \mathbf{1}\) as the vector of sample means, and
- \(\Sigma_t = \frac{1}{L-1}(W_t - \mathbf{1}\mu_t^{\top})^{\top}(W_t - \mathbf{1}\mu_t^{\top})\) as the covariance estimate.

### Hybrid quantum policy

1. **Features**: compute per‑asset angles
   \[\phi_t = \pi\,\tanh\!igl(3.5\,\frac{m_t - \bar m_t}{\sigma_t}\bigr)\in[-\pi,\pi]^n,
   \] where
   - \(m_t\) is the mean of the last 5 rows of \(W_t\) (short‑term momentum),
   - \(\bar m_t\) is the column mean of \(W_t\),
   - \(\sigma_t\) is the column standard deviation of \(W_t\) plus a small constant.

2. **Circuit evaluation**: given trainable parameters \(\theta_t \in \mathbb{R}^{3n}\), construct a variational circuit yielding statevector probabilities
   \[p^{(t)}(b) = \bigl|\langle b|U(\phi_t,\theta_t)\,|0\rangle\bigr|^2,
   \] for each computational basis string \(b\in\{0,1\}^n\). The probability an individual qubit \(i\) is in state \(|1\rangle\) is
   \[q^{(t)}_i = \sum_{b: b_i=1} p^{(t)}(b).
   \]

3. **Weights**: normalize to long‑only portfolio
   \[w^{(t)}_\text{hybrid} = \frac{q^{(t)} + \varepsilon}{\mathbf{1}^{\top}(q^{(t)} + \varepsilon)},
   \] with \(\varepsilon=10^{-9}\) to avoid division by zero.

4. **Parameter update**: new parameters are found by sampling candidates \(\theta = \theta_t + \delta\) where \(\delta\sim\mathcal{N}(0,0.35^2 I)
   \) and selecting the one maximizing utility
   
   \[U(\theta) = w(\theta)^{\top} \mu_t - \lambda\, w(\theta)^{\top}\Sigma_t w(\theta) - \gamma\,\|w(\theta) - w^{(t-1)}\|_1,
   \]
   with risk aversion \(\lambda\), transaction cost \(\gamma\), and previous weights \(w^{(t-1)}\).

### Classical mean–variance policy

Directly solve the quadratic program (ignoring non‑negativity initially):

\[w^{(t)}_\text{classical} = \underset{w\ge 0,\,\mathbf{1}^{\top}w=1}{\arg\max}\; w^{\top}\mu_t - \frac{\lambda}{2}\,w^{\top}\Sigma_t w.
\]

An analytical approximation used in the code is

\[w^{(t)}_\text{classical} = \frac{\max\{0,\,\Sigma_t^{-1} \mu_t\}}{\mathbf{1}^{\top}\max\{0,\,\Sigma_t^{-1} \mu_t\}}.
\]

### Parameter meanings

| Symbol | Description |
|--------|-------------|
| \(n\) | number of assets |
| \(T\) | total simulation steps |
| \(L\) | lookback window length |
| \(\lambda\) | risk aversion coefficient (higher -> more variance penalty) |
| \(\gamma\) | transaction cost coefficient (penalizes turnover) |
| \(\theta_t\) | quantum circuit parameters at step \(t\) |
| \(w^{(t)}\) | portfolio weights after rebalancing at \(t\) |

The hybrid weights fluctuate based on the non‑linear mapping performed by the quantum circuit, whereas the classical weights follow the smooth, deterministic dependence on \(\mu_t\) and \(\Sigma_t\).

---


---

## 4. Notes and Extensions

- The ``default`` algorithm shown above is intentionally simple for comparison. A real classical baseline might include transaction costs, regularization, or other constraints.
- Since the quantum part is simulated with `Statevector`, the performance difference is purely conceptual; running on real hardware would incur noise and measurement sampling effects.
- Users can experiment by modifying `PoCConfig` or replacing the classical optimizer with gradient‑based routines.

---

*End of document.*
