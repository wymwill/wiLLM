# tools/

Python helpers for reference work only. Nothing here is imported by the engine.

Planned for M1:

* `download_stories15m.py` fetches `stories15M.bin` and `tokenizer.bin` into `models/` (ignored by git).
* `dump_golden.py` runs the model in numpy and writes per layer activations the C++ golden tests compare against.

Allowed packages are numpy, plus torch only when a tool truly needs it. They
will be listed in `tools/requirements.txt` when the first tool lands.
