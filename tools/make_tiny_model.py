"""Write a tiny random llama2.c checkpoint for fast, offline golden tests.

    python3 tools/make_tiny_model.py tests/data/tiny.bin

The shapes are chosen to exercise every code path stories15M does not:
grouped query attention (4 query heads share 2 kv heads) and a separate
classifier matrix (negative vocab_size in the header). It is about 90 KB, so
it lives in the repo and CI needs no download.
"""

import argparse

import numpy as np

import llama2c

TINY = llama2c.Config(dim=32, hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      vocab_size=32, seq_len=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    weights = {}
    for name, shape in llama2c.weight_shapes(TINY):
        if name.startswith("freq_cis"):
            continue  # unused, written as zeros
        if name.startswith("rms"):
            weights[name] = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            weights[name] = 0.3 * rng.standard_normal(shape)
    weights["wcls"] = 0.3 * rng.standard_normal((TINY.vocab_size, TINY.dim))
    llama2c.write_checkpoint(args.output, TINY, weights, shared=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
