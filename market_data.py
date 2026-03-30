from __future__ import annotations

from typing import Tuple

import numpy as np


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

	# Each row corresponds to a different macro-regime (the code cycles every 45 steps).
	# Each column gives the per-asset drift for that regime.
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


def fetch_real_market_returns(
	tickers: Tuple[str, ...],
	period: str,
	interval: str,
) -> np.ndarray:
	"""Fetch daily returns from Yahoo Finance for a list of tickers.

	Args:
		tickers: Asset tickers compatible with Yahoo Finance.
		period: Historical period accepted by Yahoo Finance (e.g. ``2y``).
		interval: Data interval accepted by Yahoo Finance (e.g. ``1d``).

	Returns:
		Array of shape ``(n_steps, n_assets)`` with aligned percentage returns.
	"""
	try:
		import yfinance as yf
	except ImportError as exc:
		raise RuntimeError(
			"Missing dependency 'yfinance'. Install it to use real market data mode."
		) from exc

	history = yf.download(
		list(tickers),
		period=period,
		interval=interval,
		auto_adjust=True,
		progress=False,
		threads=False,
	)
	if history.empty:
		raise RuntimeError("No data returned from Yahoo Finance for the selected tickers/period")

	if "Close" in history:
		close_prices = history["Close"]
	elif "Adj Close" in history:
		close_prices = history["Adj Close"]
	else:
		raise RuntimeError("Yahoo Finance response does not contain close prices")

	if getattr(close_prices, "ndim", 1) == 1:
		close_prices = close_prices.to_frame(name=tickers[0])

	close_prices = close_prices.dropna(how="any")
	returns_frame = close_prices.pct_change().dropna(how="any")
	if returns_frame.empty:
		raise RuntimeError("Insufficient price history to compute returns")

	for ticker in tickers:
		if ticker not in returns_frame.columns:
			raise RuntimeError(f"Ticker '{ticker}' missing in Yahoo Finance response")

	ordered = returns_frame.loc[:, list(tickers)]
	return ordered.to_numpy(dtype=float)