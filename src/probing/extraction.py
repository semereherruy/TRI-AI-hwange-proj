"""Phase 8: frozen Gemma hidden-state extraction.

The model is loaded in inference mode and never fine-tuned. Hidden states are
pooled on the accelerator, moved to CPU immediately and written per layer, so GPU
memory holds only the current batch.

The sample_id -> layer -> representation mapping is preserved by writing one
array per (layer, pooling) alongside a single index of sample_ids in row order.

Run (on a GPU machine):
    python -m src.probing.extraction --split train
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from src import paths
from src.data.splits import SPLIT_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MODEL_CONFIG_PATH = paths.CONFIGS_DIR / "model.yaml"


def load_model_config(path: Optional[Path] = None) -> dict[str, Any]:
    with (path or MODEL_CONFIG_PATH).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device() -> torch.device:
    """CUDA when present; CPU otherwise. MPS is skipped: unreliable for this workload."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("no CUDA device — extraction will be slow")
    return torch.device("cpu")


def pool_hidden_states(
    hidden: torch.Tensor, attention_mask: torch.Tensor, strategy: str
) -> torch.Tensor:
    """Pool one layer's token states to one vector per sequence.

    `last_token` takes the final non-padding position, which for a causal model is
    the only position that has attended to the whole sequence. `mean` averages over
    valid tokens only, so padding never dilutes the representation.
    """
    if strategy == "last_token":
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
    if strategy == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"unknown pooling strategy: {strategy}")


def extract(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Extract pooled hidden states for every row, writing one file per layer/pooling."""
    from transformers import AutoModel, AutoTokenizer

    model_config = config["model"]
    extraction_config = config["extraction"]
    device = device or select_device()

    tokenizer = AutoTokenizer.from_pretrained(model_config["checkpoint"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, model_config["dtype"]) if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_config["checkpoint"], torch_dtype=dtype, output_hidden_states=True
    )
    model.eval().to(device)
    for parameter in model.parameters():  # frozen: no fine-tuning in this phase
        parameter.requires_grad_(False)

    poolings = list(extraction_config["poolings"])
    batch_size = int(model_config["batch_size"])
    max_length = int(model_config["max_length"])
    buffers: dict[tuple[int, str], list[np.ndarray]] = {}
    n_layers: Optional[int] = None

    texts = frame["text"].astype(str).tolist()
    for start in tqdm(range(0, len(texts), batch_size), desc="extracting"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)

        with torch.inference_mode():
            outputs = model(**encoded)

        hidden_states = outputs.hidden_states
        if n_layers is None:
            n_layers = len(hidden_states)
            layer_indices = (
                list(range(n_layers))
                if extraction_config["layers"] == "all"
                else list(extraction_config["layers"])
            )

        for layer_index in layer_indices:
            layer = hidden_states[layer_index]
            for pooling in poolings:
                pooled = pool_hidden_states(layer, encoded["attention_mask"], pooling)
                # move off the accelerator immediately; nothing accumulates in VRAM
                buffers.setdefault((layer_index, pooling), []).append(
                    pooled.to(torch.float32).cpu().numpy()
                )
        del outputs, hidden_states
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dtype = np.dtype(extraction_config["output_dtype"])
    written = []
    for (layer_index, pooling), chunks in buffers.items():
        matrix = np.concatenate(chunks, axis=0).astype(output_dtype)
        destination = output_dir / f"layer{layer_index:02d}_{pooling}.npy"
        np.save(destination, matrix)
        written.append({"layer": layer_index, "pooling": pooling, "shape": list(matrix.shape)})

    index = frame[["sample_id", "label_binary", "source", "language", "split"]].reset_index(drop=True)
    index.to_parquet(output_dir / "index.parquet", index=False)

    manifest = {
        "checkpoint": model_config["checkpoint"],
        "n_samples": int(len(frame)),
        "n_layers": n_layers,
        "poolings": poolings,
        "max_length": max_length,
        "device": device.type,
        "files": written,
        "note": "row order in every .npy matches index.parquet",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("wrote %d arrays to %s", len(written), output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Gemma hidden states")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test", "probe"])
    parser.add_argument("--limit", type=int, default=None, help="rows per split, for smoke tests")
    parser.add_argument("--checkpoint", default=None, help="override the configured checkpoint")
    parser.add_argument("--output", default=None, help="override the output directory")
    args = parser.parse_args()

    config = load_model_config()
    if args.checkpoint:
        config["model"]["checkpoint"] = args.checkpoint

    frame = pd.read_parquet(SPLIT_PATH)
    frame = frame[frame["split"].isin(args.splits) & frame["label_binary"].notna()]
    if args.limit:
        frame = frame.groupby("split", group_keys=False).head(args.limit)

    output_dir = Path(args.output) if args.output else paths.EMBEDDINGS_DIR
    extract(frame.reset_index(drop=True), config, output_dir)


if __name__ == "__main__":
    main()
