#!/usr/bin/env python3
"""SafeSABR training entry point.

Example:
    python train_sabr.py 1 0 4 100000 \
        --risk-mode cvar_rebuf --risk-alpha 0.95 --risk-lambda 20
"""

from train_sabr_logged import main


if __name__ == "__main__":
    main()

