"""`python -m audio_note.cli` entry point; same as `python -m audio_note` and the `audio-note` console script."""
from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
