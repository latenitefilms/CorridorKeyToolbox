#!/usr/bin/env python3
"""
export_mlxfn.py — produce CorridorKey .mlxfn bridges for the FxPlug runtime.

This is the upstream artifact step that was previously externalised to a
private workflow. It builds two kinds of bridge:

1. **Standard fp16** — same architecture and weights as the bundled fp32
   `.mlxfn` files but with all parameters and the input/output tensors in
   `float16`. On Apple Silicon the GPU is bandwidth-bound on these
   inferences, so cutting input/output traffic in half is the single
   biggest improvement available without changing the model.

2. **Hybrid output / 1024-encoder** — accepts inputs at any target
   resolution (e.g. 2048 or 4096) but downsamples internally to 1024
   before the Hiera encoder, then upsamples logits back to the target
   and runs the CNN refiner at full target resolution. This mirrors the
   approach a closed-source competitor uses to get ~3-4× faster 4K
   keying without retraining: the encoder grid drops from 262k tokens
   to 65k tokens, and the small refiner does the high-frequency work.

Run from this directory after a `pip install -e <path-to-corridorkey-mlx>`
(or `PYTHONPATH=…/corridorkey-mlx/src python export_mlxfn.py …`):

    # smoke-test one rung
    python export_mlxfn.py \
        --weights ~/Downloads/corridorkey_mlx.safetensors \
        --rungs 512 --fp16 --output-dir ./out

    # mass-produce all green fp16 bridges + the 4K hybrid
    python export_mlxfn.py \
        --weights ~/Downloads/corridorkey_mlx.safetensors \
        --rungs 512,768,1024,1536,2048 --fp16 \
        --hybrid-output 4096 --hybrid-encoder 1024 \
        --output-dir ./out

The output filenames match `MLXBridgeArtifact.filename(...)` in the
Swift side:

    corridorkey_mlx_bridge_<rung>.mlxfn          (legacy fp32 – not produced here)
    corridorkey_mlx_bridge_<rung>_fp16.mlxfn     (this script)
    corridorkey_mlx_bridge_<output>_<encoder>hybrid.mlxfn

Drop the resulting files into
`Source/CorridorKeyByLateNite/Plugin/Resources/MLX Models/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

try:
    from corridorkey_mlx.model.corridorkey import GreenFormer
except ImportError as exc:  # pragma: no cover — surfaced to the user
    print(
        "error: corridorkey_mlx not importable. Either:\n"
        "  pip install git+https://github.com/nikopueringer/corridorkey-mlx.git@v1.0.0\n"
        "  or set PYTHONPATH=/path/to/corridorkey-mlx/src",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _cast_module_fp16(module: nn.Module) -> None:
    """Walk the module's parameters and cast every fp32 array to fp16. The
    refiner's GroupNorm and the Hiera encoder's LayerNorm both happen to
    accept fp16 weights with stable forwards on Apple Silicon, so a
    bulk cast is fine. Buffers (no-grad arrays) ride along with
    `tree_map`."""
    def cast(arr):
        if isinstance(arr, mx.array) and arr.dtype == mx.float32:
            return arr.astype(mx.float16)
        return arr
    module.update(tree_map(cast, module.parameters()))


def _load_model(weights: Path, img_size: int, fp16: bool) -> GreenFormer:
    model = GreenFormer(img_size=img_size)
    model.load_checkpoint(weights)
    if fp16:
        _cast_module_fp16(model)
    model.eval()
    mx.eval(model.parameters())
    return model


def _resize_nhwc(x: mx.array, height: int, width: int, *, dtype: mx.Dtype) -> mx.array:
    """Bilinear NHWC resize that re-casts the result to `dtype`. MLX's
    `nn.Upsample` upcasts to fp32 internally; we cast back so the
    downstream convs stay fp16. Uses scale_factor (not target shape) so
    MLX picks its bilinear path."""
    src_h = x.shape[1]
    src_w = x.shape[2]
    if src_h == height and src_w == width:
        return x
    scale_h = height / src_h
    scale_w = width / src_w
    upsampler = nn.Upsample(
        scale_factor=(scale_h, scale_w),
        mode="linear",
        align_corners=False,
    )
    y = upsampler(x)
    if y.dtype != dtype:
        y = y.astype(dtype)
    return y


# ---------------------------------------------------------------------------
# Standard bridge (matches existing forward, but with fp16 i/o + weights)
# ---------------------------------------------------------------------------

def export_standard_bridge(
    *,
    weights: Path,
    rung: int,
    fp16: bool,
    output_dir: Path,
    filename_prefix: str = "corridorkey_mlx_bridge",
) -> Path:
    """Export a standard CorridorKey bridge that takes a single
    `(1, rung, rung, 4)` tensor and returns `(alpha_final, fg_final)`
    in NHWC. The dtype is fp16 when `fp16` is True, else fp32 (legacy).
    `filename_prefix` lets the caller distinguish colour packs:
    `corridorkey_mlx_bridge` for green, `corridorkeyblue_mlx_bridge`
    for blue. The Swift side resolves the matching prefix per
    `ScreenColor.bridgeFilenamePrefix`."""
    dtype = mx.float16 if fp16 else mx.float32
    suffix = "_fp16" if fp16 else ""
    filename = f"{filename_prefix}_{rung}{suffix}.mlxfn"
    output_path = output_dir / filename

    print(f"[{rung}] loading model + weights ({'fp16' if fp16 else 'fp32'})…")
    model = _load_model(weights, img_size=rung, fp16=fp16)

    def fwd(x: mx.array):
        out = model(x)
        # MLX's `nn.Upsample` upcasts to fp32 internally, so the final
        # sigmoid's result lands in fp32 even when the input was fp16.
        # Cast back to the request dtype so the Swift side gets the
        # half-precision outputs it asked for; this keeps the writeback
        # buffer at half the bytes per pixel as well.
        alpha = out["alpha_final"].astype(dtype)
        foreground = out["fg_final"].astype(dtype)
        return alpha, foreground

    sample = mx.zeros((1, rung, rung, 4), dtype=dtype)
    print(f"[{rung}] tracing → {output_path.name}…")
    t0 = time.time()
    mx.export_function(str(output_path), fwd, sample)
    print(f"[{rung}] done ({time.time() - t0:.1f}s, {output_path.stat().st_size // (1024 * 1024)} MB)")
    return output_path


# ---------------------------------------------------------------------------
# Hybrid bridge — high output res with low-res encoder
# ---------------------------------------------------------------------------

def _hybrid_forward(model: GreenFormer, x_target: mx.array, *, encoder_size: int, dtype: mx.Dtype):
    """Re-implement GreenFormer.__call__ but with internal downsample to
    `encoder_size` before the encoder/decoders, and upsample logits back
    to target before the refiner. The CNN refiner is conv-only and so
    runs natively at the larger target size."""
    target_h = x_target.shape[1]
    target_w = x_target.shape[2]

    # 1) Downsample 4-channel input to encoder native size.
    x_encoder = _resize_nhwc(x_target, encoder_size, encoder_size, dtype=dtype)

    # 2) Encoder + decoder heads at encoder_size; logits land at /4.
    features = model.backbone(x_encoder)
    alpha_logits = model.alpha_decoder(features)
    fg_logits = model.fg_decoder(features)

    # 3) Upsample logits all the way to target (replaces the standard 4x
    #    upsampler — total scale = target / (encoder_size / 4)).
    alpha_logits_up = _resize_nhwc(alpha_logits, target_h, target_w, dtype=dtype)
    fg_logits_up = _resize_nhwc(fg_logits, target_h, target_w, dtype=dtype)

    alpha_coarse = mx.sigmoid(alpha_logits_up)
    fg_coarse = mx.sigmoid(fg_logits_up)

    # 4) Refiner at target size — RGB pulled from the original input.
    rgb_target = x_target[:, :, :, :3]
    coarse = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)
    delta_logits = model.refiner(rgb_target, coarse)

    alpha_final = mx.sigmoid(alpha_logits_up + delta_logits[:, :, :, 0:1])
    fg_final = mx.sigmoid(fg_logits_up + delta_logits[:, :, :, 1:4])
    return alpha_final, fg_final


def export_hybrid_bridge(
    *,
    weights: Path,
    target_resolution: int,
    encoder_resolution: int,
    fp16: bool,
    output_dir: Path,
    filename_prefix: str = "corridorkey_mlx_bridge",
) -> Path:
    """Export a hybrid bridge: takes inputs at `target_resolution` and
    runs the encoder/decoder at `encoder_resolution`, refiner at
    target. The model is instantiated with `img_size=encoder_resolution`
    so the Hiera positional embedding is sized for the encoder grid."""
    dtype = mx.float16 if fp16 else mx.float32
    suffix = "_fp16" if fp16 else ""
    filename = (
        f"{filename_prefix}_{target_resolution}_{encoder_resolution}hybrid{suffix}.mlxfn"
    )
    output_path = output_dir / filename

    print(
        f"[{target_resolution}/{encoder_resolution}] loading model "
        f"({'fp16' if fp16 else 'fp32'})…"
    )
    model = _load_model(weights, img_size=encoder_resolution, fp16=fp16)

    def fwd(x: mx.array):
        alpha, foreground = _hybrid_forward(
            model, x, encoder_size=encoder_resolution, dtype=dtype
        )
        # See `export_standard_bridge` for why this cast is necessary.
        return alpha.astype(dtype), foreground.astype(dtype)

    sample = mx.zeros((1, target_resolution, target_resolution, 4), dtype=dtype)
    print(f"[{target_resolution}/{encoder_resolution}] tracing → {output_path.name}…")
    t0 = time.time()
    mx.export_function(str(output_path), fwd, sample)
    print(
        f"[{target_resolution}/{encoder_resolution}] done "
        f"({time.time() - t0:.1f}s, {output_path.stat().st_size // (1024 * 1024)} MB)"
    )
    return output_path


# ---------------------------------------------------------------------------
# Self-test: numerical sanity vs a freshly-loaded fp32 reference
# ---------------------------------------------------------------------------

def numerical_smoketest(weights: Path, exported_path: Path, rung: int) -> None:
    """Loads the just-exported bridge and a fresh fp32 reference model,
    feeds both a deterministic random input, and reports the MAE
    between the two alphas. A small MAE (≤ 1e-2) indicates the fp16
    cast didn't break the network in any obvious way."""
    print(f"[smoke] running numerical comparison at {rung}px…")
    fn = mx.import_function(str(exported_path))
    reference = _load_model(weights, img_size=rung, fp16=False)

    rng = mx.random.key(7)
    x_ref = mx.random.uniform(low=-1.0, high=1.0, shape=(1, rung, rung, 4), key=rng)

    # Probe the exported function's expected dtype by introspecting the
    # error message — there's no public API for it. Easier: try fp16
    # first, fall back to fp32.
    for dtype in (mx.float16, mx.float32):
        try:
            x_test = x_ref.astype(dtype)
            out = fn(x_test)
            mx.eval(out)
            alpha_test = out[0].astype(mx.float32)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("exported bridge accepted neither fp16 nor fp32 input")

    out_ref = reference(x_ref)
    mx.eval(out_ref["alpha_final"])
    alpha_ref = out_ref["alpha_final"]

    mae = mx.mean(mx.abs(alpha_test - alpha_ref)).item()
    print(f"[smoke] mean absolute alpha error = {mae:.4f}")
    if mae > 0.05:
        print(
            f"[smoke] WARNING: alpha MAE {mae:.4f} is large — fp16 may be losing "
            "precision somewhere. Inspect with bench_mlx.py / compare_reference.py."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to corridorkey_mlx.safetensors (download from "
             "https://huggingface.co/alexandrealvaro/corridorkey-models)",
    )
    parser.add_argument(
        "--rungs",
        default="",
        help="Comma-separated standard rung sizes to export (e.g. 512,1024). "
             "Empty = skip standard bridges.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Cast weights and i/o to float16. Strongly recommended.",
    )
    parser.add_argument(
        "--hybrid-output",
        type=int,
        default=0,
        help="Target output resolution for the hybrid bridge (e.g. 4096). 0 = skip.",
    )
    parser.add_argument(
        "--hybrid-encoder",
        type=int,
        default=1024,
        help="Encoder native resolution inside the hybrid bridge (default 1024).",
    )
    parser.add_argument(
        "--filename-prefix",
        default="corridorkey_mlx_bridge",
        help="Filename prefix for the output bridges. Use "
             "'corridorkey_mlx_bridge' (default) for the green colour pack "
             "and 'corridorkeyblue_mlx_bridge' for the blue pack.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the resulting .mlxfn files.",
    )
    parser.add_argument(
        "--smoketest-rung",
        type=int,
        default=0,
        help="If >0, after exporting that rung run a numerical comparison "
             "vs an fp32 reference and report the alpha MAE.",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"error: {args.weights} not found", file=sys.stderr)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rungs = [int(r) for r in args.rungs.split(",") if r.strip()]
    exported: list[Path] = []
    for rung in rungs:
        exported.append(export_standard_bridge(
            weights=args.weights,
            rung=rung,
            fp16=args.fp16,
            output_dir=args.output_dir,
            filename_prefix=args.filename_prefix,
        ))

    if args.hybrid_output > 0:
        exported.append(export_hybrid_bridge(
            weights=args.weights,
            target_resolution=args.hybrid_output,
            encoder_resolution=args.hybrid_encoder,
            fp16=args.fp16,
            output_dir=args.output_dir,
            filename_prefix=args.filename_prefix,
        ))

    if args.smoketest_rung > 0:
        match = next(
            (p for p in exported if f"_{args.smoketest_rung}" in p.name and "hybrid" not in p.name),
            None,
        )
        if match is None:
            print(f"error: no exported bridge for rung {args.smoketest_rung}", file=sys.stderr)
            return 1
        numerical_smoketest(args.weights, match, args.smoketest_rung)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
