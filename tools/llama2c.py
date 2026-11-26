"""Read, write and run llama2.c checkpoints in numpy.

This is the reference implementation the C++ engine is tested against. It is
written for clarity, not speed, and computes in float64 so that its own
rounding error is far below the 1e-4 tolerance of the golden tests.

Checkpoint layout (legacy llama2.c "version 0" format, which stories15M uses):
    header   7 x int32: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
             vocab_size < 0 means the classifier (wcls) is stored separately
    float32  token_embedding [vocab, dim]
             rms_att         [L, dim]
             wq              [L, dim, dim]        (out, in)
             wk              [L, kv_dim, dim]
             wv              [L, kv_dim, dim]
             wo              [L, dim, dim]
             rms_ffn         [L, dim]
             w1              [L, hidden, dim]
             w2              [L, dim, hidden]
             w3              [L, hidden, dim]
             rms_final       [dim]
             freq_cis_real   [seq_len, head_dim / 2]  (unused, skipped)
             freq_cis_imag   [seq_len, head_dim / 2]  (unused, skipped)
             wcls            [vocab, dim]            (only if vocab_size < 0)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

HEADER = struct.Struct("<7i")
RMS_EPS = 1e-5
ROPE_THETA = 10000.0


@dataclass
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def weight_shapes(c: Config) -> list[tuple[str, tuple[int, ...]]]:
    """Every tensor in file order, with its shape. Shared by reader and writer."""
    L = c.n_layers
    return [
        ("token_embedding", (c.vocab_size, c.dim)),
        ("rms_att", (L, c.dim)),
        ("wq", (L, c.dim, c.dim)),
        ("wk", (L, c.kv_dim, c.dim)),
        ("wv", (L, c.kv_dim, c.dim)),
        ("wo", (L, c.dim, c.dim)),
        ("rms_ffn", (L, c.dim)),
        ("w1", (L, c.hidden_dim, c.dim)),
        ("w2", (L, c.dim, c.hidden_dim)),
        ("w3", (L, c.hidden_dim, c.dim)),
        ("rms_final", (c.dim,)),
        ("freq_cis_real", (c.seq_len, c.head_dim // 2)),
        ("freq_cis_imag", (c.seq_len, c.head_dim // 2)),
    ]


def read_checkpoint(path: str) -> tuple[Config, dict[str, np.ndarray]]:
    raw = np.fromfile(path, dtype=np.uint8)
    fields = HEADER.unpack_from(raw.tobytes()[: HEADER.size])
    shared = fields[5] > 0
    config = Config(*fields[:5], abs(fields[5]), fields[6])

    data = raw[HEADER.size :].view(np.float32)
    weights: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in weight_shapes(config):
        n = int(np.prod(shape))
        weights[name] = data[offset : offset + n].reshape(shape)
        offset += n
    if shared:
        weights["wcls"] = weights["token_embedding"]
    else:
        n = config.vocab_size * config.dim
        weights["wcls"] = data[offset : offset + n].reshape(config.vocab_size, config.dim)
        offset += n
    if offset != data.size:
        raise ValueError(f"{path}: expected {offset} floats after the header, found {data.size}")
    return config, weights


def write_checkpoint(path: str, config: Config, weights: dict[str, np.ndarray], shared: bool) -> None:
    vocab = config.vocab_size if shared else -config.vocab_size
    with open(path, "wb") as f:
        f.write(HEADER.pack(config.dim, config.hidden_dim, config.n_layers, config.n_heads,
                            config.n_kv_heads, vocab, config.seq_len))
        for name, shape in weight_shapes(config):
            w = weights.get(name, np.zeros(shape, dtype=np.float32))
            assert w.shape == shape, (name, w.shape, shape)
            f.write(np.ascontiguousarray(w, dtype=np.float32).tobytes())
        if not shared:
            f.write(np.ascontiguousarray(weights["wcls"], dtype=np.float32).tobytes())


# ---------------------------------------------------------------- forward pass


def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return weight * x / np.sqrt(np.mean(x * x) + RMS_EPS)


def rope(vec: np.ndarray, pos: int, head_dim: int) -> np.ndarray:
    """Rotate consecutive pairs (v[2i], v[2i+1]) inside each head, llama2.c style."""
    out = vec.copy()
    for i in range(0, vec.size, 2):
        freq = 1.0 / ROPE_THETA ** ((i % head_dim) / head_dim)
        angle = pos * freq
        c, s = np.cos(angle), np.sin(angle)
        v0, v1 = vec[i], vec[i + 1]
        out[i] = v0 * c - v1 * s
        out[i + 1] = v0 * s + v1 * c
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def forward(config: Config, weights: dict[str, np.ndarray], tokens: list[int]) -> list[dict[str, np.ndarray]]:
    """Runs the tokens one position at a time and returns, per position, every
    named activation the C++ forward pass is expected to record."""
    c = config
    w = {k: v.astype(np.float64) for k, v in weights.items()}
    hd = c.head_dim
    group = c.n_heads // c.n_kv_heads
    k_cache = np.zeros((c.n_layers, c.seq_len, c.kv_dim))
    v_cache = np.zeros((c.n_layers, c.seq_len, c.kv_dim))

    trace: list[dict[str, np.ndarray]] = []
    for pos, token in enumerate(tokens):
        t: dict[str, np.ndarray] = {}
        x = w["token_embedding"][token].copy()
        t["embedding"] = x.copy()

        for l in range(c.n_layers):
            p = f"layer{l}."
            xb = rmsnorm(x, w["rms_att"][l])
            t[p + "attn_norm"] = xb
            q = rope(w["wq"][l] @ xb, pos, hd)
            k = rope(w["wk"][l] @ xb, pos, hd)
            v = w["wv"][l] @ xb
            t[p + "q"], t[p + "k"], t[p + "v"] = q, k, v
            k_cache[l, pos], v_cache[l, pos] = k, v

            att_out = np.zeros(c.dim)
            for h in range(c.n_heads):
                kvh = h // group
                qh = q[h * hd : (h + 1) * hd]
                keys = k_cache[l, : pos + 1, kvh * hd : (kvh + 1) * hd]
                vals = v_cache[l, : pos + 1, kvh * hd : (kvh + 1) * hd]
                scores = softmax(keys @ qh / np.sqrt(hd))
                att_out[h * hd : (h + 1) * hd] = scores @ vals
            t[p + "attn_out"] = att_out

            x = x + w["wo"][l] @ att_out
            t[p + "attn_residual"] = x.copy()

            xb = rmsnorm(x, w["rms_ffn"][l])
            t[p + "ffn_norm"] = xb
            hb = silu(w["w1"][l] @ xb) * (w["w3"][l] @ xb)
            ffn_out = w["w2"][l] @ hb
            t[p + "ffn_out"] = ffn_out
            x = x + ffn_out
            t[p + "output"] = x.copy()

        x = rmsnorm(x, w["rms_final"])
        t["final_norm"] = x
        t["logits"] = w["wcls"] @ x
        trace.append(t)
    return trace


# ---------------------------------------------------------------- golden files

GOLDEN_MAGIC = b"WGLD"
GOLDEN_VERSION = 1


def write_golden(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Golden file format, read by tests/support/golden.cpp. All little endian.

        "WGLD"  u32 version  u32 count
        count times:
            u32 name_len, name bytes padded with zeros to a multiple of 4
            u32 rank, rank x i64 dims
            f32 data (row major)

    Padding keeps the float data 4 byte aligned so C++ can view it in place.
    """
    with open(path, "wb") as f:
        f.write(GOLDEN_MAGIC + struct.pack("<II", GOLDEN_VERSION, len(tensors)))
        for name, value in tensors.items():
            encoded = name.encode()
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded + b"\0" * (-len(encoded) % 4))
            arr = np.ascontiguousarray(value, dtype=np.float32)
            f.write(struct.pack("<I", arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))
            f.write(arr.tobytes())
