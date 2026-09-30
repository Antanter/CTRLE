
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from interface.application import main

if __name__ == "__main__":
    main()