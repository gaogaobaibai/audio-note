"""Evaluation harness for audio-note (SPEC v1 section 10). See README.md in this folder.

story.py           hand-written deterministic 3-day story + memory ground truth + gold lexicon
build.py           MiMo TTS synthesis, session audio with acoustic conditions, truth.json, bench profile import
audiofx.py         pure numpy acoustics: resampling, levels, synthetic reverb, noise floor, channel levels
textnorm.py        text normalisation for CER (pure)
align.py           character alignment, per-term error rate, utterance <-> truth line matching (pure)
bench_cer.py       CER raw / corrected, proper-noun error rate, helpful / harmful edits
bench_speakers.py  speaker accuracy / FAR / FRR, embedding replay with pluggable scorers, real-data proxy
bench_cost.py      CPU-seconds and wall-seconds per audio-second of the local stages at 1 / 2 / 4 threads
report.py          merge result JSONs into Markdown tables
common.py          code-under-test selection, API key handling, read-only sqlite, stage timer

Every script runs as a file (``python scripts/bench/<name>.py ...``); the modules are also importable as a
package (the tests load them that way). Nothing here is imported by the application.
"""
