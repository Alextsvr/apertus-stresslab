"""Allow `python -m stresslab ...`."""

import sys

from stresslab.cli import main

if __name__ == "__main__":
    sys.exit(main())
