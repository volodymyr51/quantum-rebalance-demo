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

	n_assets: int = 3
	n_steps: int = 180
	lookback: int = 20
	random_search_samples: int = 80
	risk_aversion: float = 8.0
	transaction_cost: float = 0.002
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

	drift_regimes = np.array(
		[
			[0.0012, 0.0008, 0.0010],
			[-0.0003, 0.0014, 0.0006],
			[0.0016, -0.0002, 0.0009],
		]
	)
	drift_regimes = drift_regimes[:, :n_assets]

	base_cov = np.array(
		[
			[0.00030, 0.00009, 0.00006],
			[0.00009, 0.00024, 0.00008],
			[0.00006, 0.00008, 0.00028],
		]
	)
	covariance = base_cov[:n_assets, :n_assets]

	regime = 0
	for t in range(n_steps):
		if t > 0 and t % 45 == 0:
			regime = (regime + 1) % drift_regimes.shape[0]
		daily = rng.multivariate_normal(mean=drift_regimes[regime], cov=covariance)
		returns[t] = daily

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

	candidate_weights = quantum_policy(features=features, theta=theta)
	expected_return = float(candidate_weights @ mu)
	variance = float(candidate_weights @ cov @ candidate_weights)
	turnover = float(np.abs(candidate_weights - prev_weights).sum())
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

	for t in range(config.lookback, config.n_steps):
		window = returns[t - config.lookback : t]
		features = build_features(window)
		mu = window.mean(axis=0)
		cov = np.cov(window, rowvar=False)

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

		turnover = float(np.abs(new_weights - weights).sum())
		realized = float(new_weights @ returns[t] - config.transaction_cost * turnover)

		weights = new_weights
		portfolio_returns.append(realized)
		wealth.append(wealth[-1] * (1.0 + realized))
		turnover_log.append(turnover)
		utility_log.append(utility)
		weight_log.append(weights.copy())

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
