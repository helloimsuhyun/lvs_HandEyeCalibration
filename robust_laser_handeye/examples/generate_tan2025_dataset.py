#!/usr/bin/env python3
"""Backward-compatible entry point for the generic acquisition generator."""

from generate_calibration_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
