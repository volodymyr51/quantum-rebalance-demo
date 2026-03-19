from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class PoCConfig:
	"""Configuration container for the hybrid rebalancing proof-of-concept.

	Attributes define market simulation size, optimization intensity, and
	risk/transaction-cost preferences used during dynamic rebalancing.
	"""
	# Number of assets in the simulated universe (up to 3 for this example).
	n_assets: int = 3
	# Total number of time steps to simulate (e.g. 180 trading days).
	n_steps: int = 180
	# Number of past days used to build features and estimate returns/covariance.
	lookback: int = 20
	# Number of random candidate parameter vectors evaluated at each rebalance.
	random_search_samples: int = 80
	# Higher values lead to more conservative portfolios with lower variance but also lower expected returns.
	risk_aversion: float = 8.0
	# Cost coefficient applied to turnover in the utility function (e.g. 0.002 for 0.2% per unit turnover).
	transaction_cost: float = 0.002
	# Random seed for reproducibility of market simulation and optimization.
	seed: int = 7


def quantum_policy(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
	"""Map market features to portfolio weights with a Qiskit variational circuit.

	Args:
		features: Encoded per-asset signals used as input rotations.
		theta: Trainable circuit parameters with length ``3 * n_assets``.

	Returns:
		Normalized long-only weight vector summing to 1.
	"""

	n_assets = len(features)
	qc = QuantumCircuit(n_assets)

	for qubit, value in enumerate(features):
		qc.ry(float(value), qubit)

	for qubit in range(n_assets - 1):
		qc.cx(qubit, qubit + 1)

	for qubit in range(n_assets):
		qc.ry(float(theta[qubit]), qubit)
		qc.rz(float(theta[n_assets + qubit]), qubit)

	for qubit in range(n_assets - 1, 0, -1):
		qc.cx(qubit, qubit - 1)

	for qubit in range(n_assets):
		qc.ry(float(theta[2 * n_assets + qubit]), qubit)

	probabilities = Statevector.from_instruction(qc).probabilities()
	p_one = np.zeros(n_assets, dtype=float)
	for basis, prob in enumerate(probabilities):
		for asset in range(n_assets):
			if (basis >> asset) & 1:
				p_one[asset] += prob

	raw_scores = p_one + 1e-9
	return raw_scores / raw_scores.sum()


def simulate_market(n_steps: int, n_assets: int, seed: int) -> np.ndarray:
	"""Generate synthetic multi-asset returns with regime-switching drifts.

	Args:
		n_steps: Number of time steps to simulate.
		n_assets: Number of assets in the universe.
		seed: Random seed for reproducible sampling.

	Returns:
		Array of shape ``(n_steps, n_assets)`` with daily returns.
	"""

	rng = np.random.default_rng(seed)
	returns = np.zeros((n_steps, n_assets))

	# Each row corresponds to a different macro‑regime (the code cycles every 45 steps).
	# Each column gives the per‑asset drift for that regime.
	drift_regimes = np.array(
		[
			[0.0012, 0.0008, 0.0010],
			[-0.0003, 0.0014, 0.0006],
			[0.0016, -0.0002, 0.0009],
		]
	)
	drift_regimes = drift_regimes[:, :n_assets]

	# Diagonal entries (0.00030, 0.00024, 0.00028) are the variances of each asset’s return.
	#Off‑diagonal entries (e.g. 0.00009 between asset 0 and asset 1) are their pairwise covariances.
	base_cov = np.array(
		[
			[0.00030, 0.00009, 0.00006],
			[0.00009, 0.00024, 0.00008],
			[0.00006, 0.00008, 0.00028],
		]
	)
	covariance = base_cov[:n_assets, :n_assets]

	# The code models the market as switching every 45 days between a few predefined drift vectors. Each of those vectors is a regime:

	# Regime 0 might represent a benign, rising‑market environment (all assets have small positive drifts).
	# Regime 1 could be a rotation or correction (one or more drifts turn negative).
	# Regime 2 another distinct market state, etc.
	regime = 0
	for t in range(n_steps):
		if t > 0 and t % 45 == 0:
			regime = (regime + 1) % drift_regimes.shape[0]
		daily = rng.multivariate_normal(mean=drift_regimes[regime], cov=covariance)
		returns[t] = daily
	
	# This creates a more realistic simulation where the expected returns of the assets change over time, forcing the rebalancing strategy to adapt to shifting market conditions.
	return returns


def build_features(window_returns: np.ndarray) -> np.ndarray:
	"""Create bounded feature angles from a rolling return window.

	The features combine momentum, mean, and volatility into a z-scored signal,
	then squash it to ``[-pi, pi]`` so it can be used as quantum rotation angles.

	Args:
		window_returns: Rolling returns matrix for recent observations.

	Returns:
		Per-asset feature vector of rotation angles.
	"""

	mean_signal = window_returns.mean(axis=0)
	volatility = window_returns.std(axis=0) + 1e-9
	momentum = window_returns[-5:].mean(axis=0)
	zscore = (momentum - mean_signal) / volatility
	scaled = np.tanh(3.5 * zscore)
	return np.pi * scaled


def evaluate_candidate(
	theta: np.ndarray,
	features: np.ndarray,
	mu: np.ndarray,
	cov: np.ndarray,
	prev_weights: np.ndarray,
	risk_aversion: float,
	transaction_cost: float,
) -> Tuple[float, np.ndarray]:
	"""Score one candidate parameter vector using utility maximization.

	Utility is defined as expected return minus variance penalty and turnover cost.

	Args:
		theta: Candidate circuit parameters.
		features: Current feature angles for the quantum policy.
		mu: Expected returns estimated from the rolling window.
		cov: Covariance matrix estimated from the rolling window.
		prev_weights: Portfolio weights before rebalancing.
		risk_aversion: Multiplier on portfolio variance.
		transaction_cost: Cost coefficient applied to turnover.

	Returns:
		Tuple of ``(utility_score, candidate_weights)``.
	"""
	# Get candidate weights from the quantum policy and compute utility components.
	candidate_weights = quantum_policy(features=features, theta=theta)
	# The expected return is the dot product of the candidate weights and the estimated expected returns (mu).
	expected_return = float(candidate_weights @ mu)
	# The variance is calculated as the quadratic form candidate_weights^T * cov * candidate_weights, which gives the portfolio variance based on the covariance matrix and the candidate weights.
	variance = float(candidate_weights @ cov @ candidate_weights)
	# Turnover is the sum of absolute differences between the candidate weights and the previous weights, representing how much the portfolio would need to be adjusted. This is multiplied by the transaction cost to get the total cost of rebalancing.
	turnover = float(np.abs(candidate_weights - prev_weights).sum())
	# The utility combines the expected return, penalizes variance according to the risk aversion parameter, and subtracts the transaction cost based on turnover. A higher utility indicates a more desirable candidate portfolio.
	utility = expected_return - risk_aversion * variance - transaction_cost * turnover
	return utility, candidate_weights


def optimize_quantum_parameters(
	rng: np.random.Generator,
	features: np.ndarray,
	mu: np.ndarray,
	cov: np.ndarray,
	prev_weights: np.ndarray,
	theta_center: np.ndarray,
	random_search_samples: int,
	risk_aversion: float,
	transaction_cost: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
	"""Perform local random search to improve quantum circuit parameters.

	Args:
		rng: Random generator used for parameter perturbations.
		features: Feature angles passed to the quantum policy.
		mu: Estimated expected returns.
		cov: Estimated return covariance matrix.
		prev_weights: Previous portfolio allocation.
		theta_center: Current parameter vector around which candidates are sampled.
		random_search_samples: Number of sampled candidates to evaluate.
		risk_aversion: Utility penalty factor for variance.
		transaction_cost: Utility penalty factor for turnover.

	Returns:
		Tuple ``(best_theta, best_weights, best_utility)``.
	"""
	# Start with the current parameters as the best candidate and evaluate its utility.
	best_theta = theta_center.copy()
	best_utility, best_weights = evaluate_candidate(
		theta=best_theta,
		features=features,
		mu=mu,
		cov=cov,
		prev_weights=prev_weights,
		risk_aversion=risk_aversion,
		transaction_cost=transaction_cost,
	)
	# Perform a local random search by sampling candidate parameter vectors around the current center. Each candidate is evaluated, and if it has a higher utility than the best found so far, it becomes the new best.
	for _ in range(random_search_samples):
		candidate_theta = theta_center + rng.normal(0.0, 0.35, size=theta_center.shape[0])
		utility, candidate_weights = evaluate_candidate(
			theta=candidate_theta,
			features=features,
			mu=mu,
			cov=cov,
			prev_weights=prev_weights,
			risk_aversion=risk_aversion,
			transaction_cost=transaction_cost,
		)
		if utility > best_utility:
			best_utility = utility
			best_theta = candidate_theta
			best_weights = candidate_weights

	return best_theta, best_weights, best_utility


def run_hybrid_rebalancing(config: PoCConfig) -> None:
	"""Run the full hybrid quantum-classical dynamic rebalancing simulation.

	At each rebalance step, the function builds features from recent returns,
	optimizes quantum parameters classically, applies new weights, and logs
	portfolio statistics before printing a compact summary.

	Args:
		config: Simulation and optimization settings.
	"""
	# Step 1: Simulate market returns with regime-switching drifts.
	rng = np.random.default_rng(config.seed)
	returns = simulate_market(
		n_steps=config.n_steps,
		n_assets=config.n_assets,
		seed=config.seed,
	)

	theta = rng.normal(0.0, 0.5, size=3 * config.n_assets)
	weights = np.full(config.n_assets, 1.0 / config.n_assets)

	portfolio_returns = []
	wealth = [1.0]
	turnover_log = []
	utility_log = []
	weight_log = []

	# Step 2: Iterate through time steps, rebalance portfolio, and log results.
	for t in range(config.lookback, config.n_steps):
		window = returns[t - config.lookback : t]
		features = build_features(window)
		mu = window.mean(axis=0)
		cov = np.cov(window, rowvar=False)

		# Optimize quantum parameters with local random search to find better weights.
		theta, new_weights, utility = optimize_quantum_parameters(
			rng=rng,
			features=features,
			mu=mu,
			cov=cov,
			prev_weights=weights,
			theta_center=theta,
			random_search_samples=config.random_search_samples,
			risk_aversion=config.risk_aversion,
			transaction_cost=config.transaction_cost,
		)
		# Calculate turnover and realized return after applying new weights, accounting for transaction costs.
		turnover = float(np.abs(new_weights - weights).sum())
		realized = float(new_weights @ returns[t] - config.transaction_cost * turnover)
		# Update portfolio weights and log statistics for this step.
		weights = new_weights
		portfolio_returns.append(realized)
		wealth.append(wealth[-1] * (1.0 + realized))
		turnover_log.append(turnover)
		utility_log.append(utility)
		weight_log.append(weights.copy())

	# After the simulation, convert logs to arrays for easier analysis and print summary statistics.
	portfolio_returns_arr = np.array(portfolio_returns)
	weight_log_arr = np.array(weight_log)

	ann_factor = 252
	avg_daily = float(portfolio_returns_arr.mean())
	vol_daily = float(portfolio_returns_arr.std() + 1e-12)
	sharpe = np.sqrt(ann_factor) * avg_daily / vol_daily
	cumulative_return = wealth[-1] - 1.0

	print("Hybrid quantum-classical dynamic rebalancing PoC")
	print("=" * 56)
	print(f"Assets: {config.n_assets}, Steps: {config.n_steps}, Lookback: {config.lookback}")
	print(f"Final wealth: {wealth[-1]:.4f}")
	print(f"Cumulative return: {100 * cumulative_return:.2f}%")
	print(f"Average daily return: {100 * avg_daily:.3f}%")
	print(f"Daily volatility: {100 * vol_daily:.3f}%")
	print(f"Sharpe (annualized): {sharpe:.2f}")
	print(f"Average turnover per rebalance: {np.mean(turnover_log):.3f}")
	print(f"Average utility score: {np.mean(utility_log):.6f}")

	print("\nLast 5 rebalances (weights):")
	for idx, row in enumerate(weight_log_arr[-5:], start=len(weight_log_arr) - 4):
		weight_text = ", ".join(f"{w:.3f}" for w in row)
		print(f"t={idx:3d} -> [{weight_text}]")


if __name__ == "__main__":
	run_hybrid_rebalancing(PoCConfig())
