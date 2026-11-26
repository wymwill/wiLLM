# 0005 Golden tests against a float64 numpy reference

Status: accepted
Date: 2026-11-26

## Context
A forward pass that is wrong usually still produces numbers. We need to know
that the C++ is correct, and when it is not, which step is wrong.

## Decision
* `tools/llama2c.py` is a short, readable numpy forward pass computed in
  float64, transcribed from llama2.c's `run.c`. It is the definition of correct.
* `Transformer::forward` takes an optional `ActivationSink` and records about a
  dozen named activations per layer. The golden test compares every one, per
  position, and reports the first mismatch (for example
  `position 0, activation "layer0.attn_out"`).
* Tolerance is `|actual - expected| <= 1e-4 * (1 + |expected|)`, absolute
  near zero and relative for large logits. f32 summation order alone moves
  results by around 1e-6, so a real bug is far outside this band.
* A tiny random model (`tests/data/tiny.bin`, 80 KB) is committed so the test
  runs offline and in CI. It uses grouped query attention and a separate
  classifier, which stories15M does not, so both code paths are covered.
  stories15M is tested too when its files are present in `models/`.
* Golden files use a small custom binary format (see `write_golden`) that C++
  maps and views in place. No npy or JSON parser needed.

## Consequences
* Every future backend (NEON, Metal, int8) reuses the same golden files, only
  the tolerance may change for quantized paths.
* The sink is a virtual call per activation, which only happens when a sink is
  passed. Benchmarks pass nullptr.
