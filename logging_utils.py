from __future__ import annotations

import logging


def configure_logging(level: str) -> logging.Logger:
	"""Configure module logger with a level controlled by configuration.

	Args:
		level: Desired log level (e.g. ``DEBUG``, ``INFO``, ``WARNING``).

	Returns:
		Configured logger instance.
	"""
	logger = logging.getLogger("hybrid_rebalancing")
	resolved_level = getattr(logging, level.upper(), logging.INFO)
	logger.setLevel(resolved_level)

	if not logger.handlers:
		handler = logging.StreamHandler()
		handler.setFormatter(
			logging.Formatter(
				"%(asctime)s | %(levelname)s | %(name)s | %(message)s",
				datefmt="%H:%M:%S",
			)
		)
		logger.addHandler(handler)

	logger.propagate = False
	return logger