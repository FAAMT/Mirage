"""
marian_fp32.py -- Helsinki-NLP/opus-mt-en-ar in numpy, float32, no torch.

This is the TRUTH the whole verification chain hangs from. It is not the
model the FPGA runs; it is the model the FPGA is trying to approximate. Its
job is to answer, unambiguously, "what is the correct Arabic translation of
'Hello', and what is the correct value of every intermediate tensor on the
way there".

Architecture, from hf/config.json:
    d_model 512, 8 heads, head_dim 64, ffn 2048, 6 encoder + 6 decoder layers
    activation_function  swish (SiLU)
    normalize_before     false   -> POST-norm: x = LN(x + sublayer(x))
    static_position_embeddings   -> sinusoidal, but stored in the checkpoint
    scale_embedding      true    -> embeddings scaled by sqrt(d_model)
    normalize_embedding  false   -> no layernorm on the embedding
    add_final_layer_norm false   -> no layernorm on the encoder output
    decoder_start_token_id 62801 (= pad), eos_token_id 0

Every operation here is written out longhand rather than vectorised into
something clever, because the int8 model next door has to mirror it operation
for operation, and a fused expression is one you cannot point a testbench at.
"""

import json
import os

import numpy as np

import torchbin

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.join(HERE, "hf")

D_MODEL = 512
N_HEADS = 8
HEAD_DIM = 64
FFN_DIM = 2048
N_LAYERS = 6
PAD_ID = 62801
EOS_ID = 0
DECODER_START = 62801
LN_EPS = 1e-5


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def layer_norm(x, weight, bias, eps=LN_EPS):
    """LayerNorm over the last axis. Biased variance, as PyTorch uses."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def silu(x):
    """swish / SiLU: x * sigmoid(x). config.activation_function == 'swish'."""
    return x / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def linear(x, weight, bias):
    """torch.nn.Linear: y = x @ W^T + b, W stored as (out_features, in)."""
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------

def split_heads(x):
    """(T, D) -> (H, T, head_dim)"""
    t = x.shape[0]
    return x.reshape(t, N_HEADS, HEAD_DIM).transpose(1, 0, 2)


def merge_heads(x):
    """(H, T, head_dim) -> (T, D)"""
    t = x.shape[1]
    return x.transpose(1, 0, 2).reshape(t, D_MODEL)


def attention(q_in, kv_in, w, prefix, causal=False, trace=None, tag=""):
    """One multi-head attention block. Returns (T_q, D).

    `trace`, if given, collects every intermediate the RTL also produces, so
    the hardware can be checked stage by stage rather than only end to end.
    """
    scaling = HEAD_DIM ** -0.5

    q = linear(q_in, w[f"{prefix}.q_proj.weight"], w[f"{prefix}.q_proj.bias"]) * scaling
    k = linear(kv_in, w[f"{prefix}.k_proj.weight"], w[f"{prefix}.k_proj.bias"])
    v = linear(kv_in, w[f"{prefix}.v_proj.weight"], w[f"{prefix}.v_proj.bias"])

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)

    scores = np.einsum("htd,hsd->hts", qh, kh)          # (H, Tq, Tk)
    if causal:
        tq, tk = scores.shape[1], scores.shape[2]
        mask = np.triu(np.full((tq, tk), -np.inf, dtype=np.float32), k=1 + tk - tq)
        scores = scores + mask

    probs = softmax(scores, axis=-1)
    ctx = np.einsum("hts,hsd->htd", probs, vh)          # (H, Tq, head_dim)
    merged = merge_heads(ctx)
    out = linear(merged, w[f"{prefix}.out_proj.weight"], w[f"{prefix}.out_proj.bias"])

    if trace is not None:
        trace[f"{tag}q"] = q
        trace[f"{tag}k"] = k
        trace[f"{tag}v"] = v
        trace[f"{tag}scores"] = scores
        trace[f"{tag}probs"] = probs
        trace[f"{tag}ctx"] = merged
        trace[f"{tag}attn_out"] = out
    return out


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

def encoder_layer(x, w, li, trace=None):
    p = f"model.encoder.layers.{li}"
    tag = f"enc{li}." if trace is not None else ""

    residual = x
    x = attention(x, x, w, f"{p}.self_attn", trace=trace, tag=tag)
    x = residual + x
    x = layer_norm(x, w[f"{p}.self_attn_layer_norm.weight"],
                   w[f"{p}.self_attn_layer_norm.bias"])
    if trace is not None:
        trace[f"{tag}post_attn_ln"] = x

    residual = x
    h = silu(linear(x, w[f"{p}.fc1.weight"], w[f"{p}.fc1.bias"]))
    y = linear(h, w[f"{p}.fc2.weight"], w[f"{p}.fc2.bias"])
    x = residual + y
    x = layer_norm(x, w[f"{p}.final_layer_norm.weight"],
                   w[f"{p}.final_layer_norm.bias"])

    if trace is not None:
        trace[f"{tag}ffn_hidden"] = h
        trace[f"{tag}ffn_out"] = y
        trace[f"{tag}out"] = x
    return x


def decoder_layer(x, enc_out, w, li):
    p = f"model.decoder.layers.{li}"

    residual = x
    x = attention(x, x, w, f"{p}.self_attn", causal=True)
    x = residual + x
    x = layer_norm(x, w[f"{p}.self_attn_layer_norm.weight"],
                   w[f"{p}.self_attn_layer_norm.bias"])

    residual = x
    x = attention(x, enc_out, w, f"{p}.encoder_attn")
    x = residual + x
    x = layer_norm(x, w[f"{p}.encoder_attn_layer_norm.weight"],
                   w[f"{p}.encoder_attn_layer_norm.bias"])

    residual = x
    h = silu(linear(x, w[f"{p}.fc1.weight"], w[f"{p}.fc1.bias"]))
    y = linear(h, w[f"{p}.fc2.weight"], w[f"{p}.fc2.bias"])
    x = residual + y
    x = layer_norm(x, w[f"{p}.final_layer_norm.weight"],
                   w[f"{p}.final_layer_norm.bias"])
    return x


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class Marian:
    def __init__(self, weights=None):
        self.w = weights if weights is not None else torchbin.load(
            os.path.join(HF, "pytorch_model.bin"))
        self.embed_scale = np.float32(np.sqrt(D_MODEL))
        self.vocab = json.load(open(os.path.join(HF, "vocab.json"), encoding="utf-8"))
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # -- embeddings ------------------------------------------------------
    def embed(self, ids, which="encoder"):
        tok = self.w["model.shared.weight"][ids] * self.embed_scale
        pos = self.w[f"model.{which}.embed_positions.weight"][:len(ids)]
        return (tok + pos).astype(np.float32)

    # -- encoder ---------------------------------------------------------
    def encode(self, ids, trace=None):
        x = self.embed(ids, "encoder")
        if trace is not None:
            trace["embed"] = x
        for li in range(N_LAYERS):
            x = encoder_layer(x, self.w, li, trace=trace)
        return x

    # -- decoder ---------------------------------------------------------
    def decode_step(self, dec_ids, enc_out):
        x = self.embed(dec_ids, "decoder")
        for li in range(N_LAYERS):
            x = decoder_layer(x, enc_out, self.w, li)
        logits = x @ self.w["model.shared.weight"].T
        logits = logits + self.w["final_logits_bias"]
        return logits

    # -- greedy generation ----------------------------------------------
    def generate(self, ids, max_new=24):
        enc_out = self.encode(ids)
        dec = [DECODER_START]
        for _ in range(max_new):
            logits = self.decode_step(np.array(dec), enc_out)[-1]
            logits[PAD_ID] = -np.inf        # config.bad_words_ids
            nxt = int(logits.argmax())
            dec.append(nxt)
            if nxt == EOS_ID:
                break
        return dec[1:]

    def detokenise(self, ids):
        pieces = [self.inv_vocab[i] for i in ids
                  if i not in (PAD_ID, EOS_ID)]
        return "".join(pieces).replace("▁", " ").strip()


if __name__ == "__main__":
    import sentencepiece as spm

    m = Marian()
    sp = spm.SentencePieceProcessor(model_file=os.path.join(HF, "source.spm"))

    for text in ("Hello", "Hello world", "Good morning"):
        pieces = sp.encode(text, out_type=str)
        ids = [m.vocab[p] for p in pieces] + [EOS_ID]
        out = m.generate(np.array(ids))
        print(f"{text!r}")
        print(f"  source pieces {pieces} -> ids {ids}")
        print(f"  target ids    {out}")
        print(f"  target pieces {[m.inv_vocab[i] for i in out]}")
        print(f"  TRANSLATION   {m.detokenise(out)!r}")
