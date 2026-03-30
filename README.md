# Hybrid quantum–classical dynamic portfolio rebalancing (PoC)

This project provides a compact proof-of-concept (PoC) for a **hybrid quantum–classical** portfolio rebalancing workflow.

The idea is simple:
1. Use recent market data to build features (mean, volatility, momentum).
2. Feed features into a small variational quantum circuit built with Qiskit.
3. Convert measured qubit probabilities into portfolio weights.
4. Use a classical optimizer (random search) to tune circuit parameters at each rebalance step.
5. Rebalance dynamically with transaction cost penalty.

## Files

- `main.py`: complete runnable PoC script.
- `quantum_finance_business.ipynb`: notebook version with business-friendly execution flow.
- `requirements.txt`: minimal dependency list.

## Quick start

```powershell
py -m uv venv .venv
py -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt
py -m uv run --python .venv\Scripts\python.exe main.py
```

## Data source modes

`main.py` now supports both simulated and real market data.

Use simulated data (default):

```powershell
py -m uv run --python .venv\Scripts\python.exe main.py --data-source simulated --n-assets 3 --n-steps 180
```

Use real market data (Yahoo Finance API via `yfinance`):

```powershell
py -m uv run --python .venv\Scripts\python.exe main.py --data-source real --tickers SPY,TLT,GLD --real-period 2y --real-interval 1d
```

## Run as notebook

```powershell
py -m uv pip install --python .venv\Scripts\python.exe jupyter
py -m uv run --python .venv\Scripts\python.exe jupyter notebook
```

Open `quantum_finance_business.ipynb` and run cells from top to bottom.

## What this PoC demonstrates

- **Quantum part**: a compact Qiskit circuit with parameterized `Ry`/`Rz` gates and `CNOT` entanglement, evaluated via statevector probabilities.
- **Classical part**: utility maximization based on expected return, risk penalty, and turnover cost.
- **Dynamic rebalancing**: repeated optimization and reallocation across time.

## Notes

- This is an educational prototype, not production portfolio advice.
- Real market data mode uses Yahoo Finance (`yfinance`), and simulated mode remains available for controlled experiments.
- You can tweak `PoCConfig` in `main.py` to test different settings.


## Run example
```
========================================================
Assets: 3, Steps: 180, Lookback: 20
Cumulative return: 56.73%
Average daily return: 0.289%
Daily volatility: 1.241%
Sharpe (annualized): 3.70
Average turnover per rebalance: 0.223
Average utility score: 0.002689

Last 5 rebalances (weights):
t=156 -> [0.069, 0.653, 0.278]
t=157 -> [0.044, 0.689, 0.267]
t=158 -> [0.042, 0.706, 0.253]
t=159 -> [0.186, 0.614, 0.200]
t=160 -> [0.238, 0.434, 0.327]
```