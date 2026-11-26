# tools/

Python reference tools. Nothing here is imported by the engine. Only numpy is
needed (`pip install -r tools/requirements.txt`).

| Script | What it does |
| --- | --- |
| `llama2c.py` | Reads and writes llama2.c checkpoints and runs the forward pass in numpy (float64). The reference the C++ is tested against. |
| `download_stories15m.py` | Fetches `stories15M.bin` and `tokenizer.bin` into `models/` (ignored by git). |
| `dump_golden.py` | Runs `llama2c.forward` over a fixed token list and writes every intermediate activation to a `.golden` file. |
| `make_tiny_model.py` | Writes the tiny random checkpoint in `tests/data/`, used by the always on golden test. |

## Golden tests for stories15M

```sh
pip install -r tools/requirements.txt
python3 tools/download_stories15m.py
python3 tools/dump_golden.py models/stories15M.bin models/stories15M.golden
ctest --preset debug -R model
```

The stories15M golden test is skipped until both files exist in `models/`.

## Regenerating the tiny model

Only needed if the golden format or the reference changes:

```sh
python3 tools/make_tiny_model.py tests/data/tiny.bin
python3 tools/dump_golden.py tests/data/tiny.bin tests/data/tiny.golden --tokens 1 5 9 2 17 30
```
