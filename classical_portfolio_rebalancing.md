# Classical Portfolio Rebalancing (Step-by-Step, No Code)

This note explains how to perform the portfolio rebalancing step in a fully classical way, using the same risk-return-cost logic as the current workflow.

## 1) Inputs at Rebalance Time

At each rebalance date $t$, use the last $L$ observations (lookback window) of asset returns:

$$
R_{t-L:t-1} \in \mathbb{R}^{L \times N}
$$

where:
- $N$ = number of assets
- $L$ = lookback length

Also keep previous portfolio weights:

$$
\mathbf{w}_{t-1} = (w_{t-1,1}, \dots, w_{t-1,N})^\top
$$

with constraints:

$$
\sum_{i=1}^N w_{t-1,i} = 1, \quad w_{t-1,i} \ge 0
$$

## 2) Estimate Return and Risk from the Lookback Window

Expected return vector:

$$
\boldsymbol{\mu}_t = \frac{1}{L}\sum_{\tau=t-L}^{t-1} \mathbf{r}_{\tau}
$$

Covariance matrix (sample covariance computed from the lookback window):

$$
\Sigma_t = \frac{1}{L-1}\sum_{\tau=t-L}^{t-1}
(\mathbf{r}_\tau - \boldsymbol{\mu}_t)
(\mathbf{r}_\tau - \boldsymbol{\mu}_t)^\top
$$

## 3) Classical Optimization Objective

Choose new weights $\mathbf{w}_t$ by maximizing the utility:

$$U(\mathbf{w}) = \boldsymbol{\mu}_t^\top \mathbf{w} - \lambda\,\mathbf{w}^\top \Sigma_t \mathbf{w} - c\,\left\lVert \mathbf{w} - \mathbf{w}_{t-1} \right\rVert_1$$

subject to:

$$
\sum_{i=1}^N w_i = 1, \quad w_i \ge 0
$$

where:
- $\lambda$ = risk aversion
- $c$ = transaction cost coefficient
- $\left\lVert \mathbf{w} - \mathbf{w}_{t-1} \right\rVert_1 = \sum_{i=1}^N \lvert w_i - w_{t-1,i} \rvert$ is turnover

Interpretation:
- First term rewards expected return.
- Second term penalizes variance (risk).
- Third term penalizes trading too much.

## 4) Practical Classical Solving Methods

You can solve this with a classical optimizer:
- Constrained nonlinear optimization (directly with the $L_1$ turnover term).
- Quadratic programming if you replace $L_1$ by $L_2$ turnover penalty.
- Random search over feasible weights (simple baseline): sample many candidate weight vectors, evaluate $U(\mathbf{w})$, keep the best.

## 5) Worked Example (3 Assets)

Assume:

$$
\boldsymbol{\mu}_t =
\begin{bmatrix}
0.0010 \\
0.0006 \\
0.0012
\end{bmatrix},
\quad
\Sigma_t =
\begin{bmatrix}
0.00030 & 0.00009 & 0.00006 \\
0.00009 & 0.00024 & 0.00008 \\
0.00006 & 0.00008 & 0.00028
\end{bmatrix}
$$

Previous weights:

$$
\mathbf{w}_{t-1} =
\begin{bmatrix}
0.3333 \\
0.3333 \\
0.3333
\end{bmatrix}
$$

Parameters:

$$
\lambda = 8, \quad c = 0.002
$$

Evaluate two feasible candidate portfolios.

### Candidate A

$$
\mathbf{w}^{(A)} = [0.40,\ 0.20,\ 0.40]^\top
$$

Expected return:

$$
\boldsymbol{\mu}_t^\top \mathbf{w}^{(A)}
= 0.40(0.0010) + 0.20(0.0006) + 0.40(0.0012)
= 0.00100
$$

Variance:

$$
(\mathbf{w}^{(A)})^\top \Sigma_t \mathbf{w}^{(A)} = 0.0001488
$$

Turnover:

$$
\|\mathbf{w}^{(A)} - \mathbf{w}_{t-1}\|_1
= |0.40-0.3333| + |0.20-0.3333| + |0.40-0.3333|
= 0.267
$$

Utility:

$$
U_A = 0.00100 - 8(0.0001488) - 0.002(0.267)
= -0.0007238
$$

### Candidate B

$$
\mathbf{w}^{(B)} = [0.25,\ 0.45,\ 0.30]^\top
$$

Expected return:

$$
\boldsymbol{\mu}_t^\top \mathbf{w}^{(B)} = 0.00088
$$

Variance:

$$
(\mathbf{w}^{(B)})^\top \Sigma_t \mathbf{w}^{(B)} = 0.0001434
$$

Turnover:

$$
\|\mathbf{w}^{(B)} - \mathbf{w}_{t-1}\|_1 = 0.2333
$$

Utility:

$$
U_B = 0.00088 - 8(0.0001434) - 0.002(0.2333)
= -0.0007338
$$

Since $U_A > U_B$, choose Candidate A:

$$
\mathbf{w}_t = \mathbf{w}^{(A)}
$$

## 6) Realized Net Return and Wealth Update

If next-day realized asset returns are:

$$
\mathbf{r}_t = [0.0020,\ -0.0010,\ 0.0015]^\top
$$

Gross portfolio return:

$$
r_{p,t}^{\text{gross}} = \mathbf{w}_t^\top \mathbf{r}_t
= 0.40(0.0020) + 0.20(-0.0010) + 0.40(0.0015)
= 0.0012
$$

Net return after transaction cost:

$$
r_{p,t}^{\text{net}} = r_{p,t}^{\text{gross}} - c\,\|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1
= 0.0012 - 0.002(0.267)
= 0.0006
$$

Wealth recursion:

$$
W_t = W_{t-1}(1 + r_{p,t}^{\text{net}})
$$

If $W_{t-1}=1.0000$, then:

$$
W_t = 1.0000 \times (1 + 0.0006) = 1.0006
$$

## 7) Repeat Through Time

For each new time step:
1. Roll the lookback window forward.
2. Recompute $\boldsymbol{\mu}_t$ and $\Sigma_t$.
3. Re-optimize $\mathbf{w}_t$ with the same utility.
4. Apply realized return and transaction cost.
5. Update wealth and continue.

This is a complete classical replacement for the rebalancing step, with no quantum circuit dependency.
