"""Run the numpy reference over a fixed token sequence and save every
intermediate activation for the C++ golden tests.

    python3 tools/dump_golden.py models/stories15M.bin models/stories15M.golden

Entries are named "pos<P>.<activation>", for example "pos2.layer0.q". The
token ids themselves are stored as "tokens" so the C++ test replays the same input.
"""

import argparse

import numpy as np

import llama2c

# BOS followed by a few ordinary ids. The exact text does not matter for the
# golden test, only that C++ and numpy see the same ids.
DEFAULT_TOKENS = [1, 9038, 2501, 263, 931, 29892]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    args = ap.parse_args()

    config, weights = llama2c.read_checkpoint(args.checkpoint)
    tokens = [t % config.vocab_size for t in args.tokens]
    trace = llama2c.forward(config, weights, tokens)

    tensors = {"tokens": np.array(tokens, dtype=np.float32)}
    for pos, step in enumerate(trace):
        for name, value in step.items():
            tensors[f"pos{pos}.{name}"] = value
    llama2c.write_golden(args.output, tensors)
    print(f"wrote {len(tensors)} tensors for {len(tokens)} tokens to {args.output}")


if __name__ == "__main__":
    main()
