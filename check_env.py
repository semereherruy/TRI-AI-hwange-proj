"""Phase 1 environment verification.

Reports the interpreter, the dependency versions this project actually relies on,
and the available compute device. Runs identically on Colab and on a local machine
so that Phase 1 evidence is reproducible in both places.

Usage:
    python check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Optional

# (import name, human label, required for the project to be usable at all)
DEPENDENCIES: list[tuple[str, str, bool]] = [
    ("torch", "PyTorch", True),
    ("transformers", "Transformers", True),
    ("huggingface_hub", "HuggingFace Hub", True),
    ("datasets", "Datasets", True),
    ("accelerate", "Accelerate", True),
    ("sentencepiece", "SentencePiece", True),
    ("safetensors", "Safetensors", True),
    ("pandas", "pandas", True),
    ("numpy", "NumPy", True),
    ("pyarrow", "PyArrow", True),
    ("sklearn", "scikit-learn", True),
    ("scipy", "SciPy", True),
    ("matplotlib", "Matplotlib", True),
    ("tqdm", "tqdm", True),
    ("ipykernel", "ipykernel", False),
    ("umap", "umap-learn (optional, Phase 13)", False),
]

MIN_PYTHON = (3, 9)


def _version(module_name: str) -> Optional[str]:
    """Return an installed module's version, or None if it is not importable."""
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "unknown")


def check_dependencies() -> list[str]:
    """Print each dependency's status. Returns the names of missing required ones."""
    print("Dependencies")
    missing_required: list[str] = []
    for import_name, label, required in DEPENDENCIES:
        version = _version(import_name)
        if version is None:
            mark = "MISSING" if required else "absent (optional)"
            if required:
                missing_required.append(label)
            print(f"  {label:<34} {mark}")
        else:
            print(f"  {label:<34} {version}")
    return missing_required


def check_numpy_bridge() -> Optional[str]:
    """Verify the torch<->numpy bridge. Returns a problem description, or None.

    PyTorch built against the NumPy 1.x C API imports fine under NumPy 2.x but
    cannot convert tensors to arrays. Phase 8 -> Phase 9 depends on that path,
    so this is checked explicitly rather than discovered mid-extraction.
    """
    try:
        import torch

        torch.zeros(2).numpy()
    except ImportError:
        return "torch not installed"
    except Exception as exc:
        return f"tensor.numpy() failed: {exc}"
    return None


def check_device() -> None:
    """Report the compute device available for hidden-state extraction."""
    print("\nCompute")
    try:
        import torch
    except ImportError:
        print("  torch not installed — cannot determine device")
        return

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  CUDA GPU: {name} ({total_gb:.1f} GB)")
    elif torch.backends.mps.is_available():
        print("  MPS backend reported available (Apple Metal)")
        print("  NOTE: unverified for Gemma hidden-state extraction; treat as CPU-class")
    else:
        print(f"  CPU only ({torch.get_num_threads()} threads)")
        print("  NOTE: Phase 8 (Gemma extraction) needs a CUDA GPU to be practical")


def main() -> int:
    print("Environment")
    print(f"  Python                             {platform.python_version()}")
    print(f"  Platform                           {platform.system()} {platform.machine()}")
    print()

    missing = check_dependencies()
    check_device()

    problems: list[str] = []
    if sys.version_info < MIN_PYTHON:
        problems.append(f"Python {platform.python_version()} is below the {'.'.join(map(str, MIN_PYTHON))} minimum")
    if missing:
        problems.append(f"missing required packages: {', '.join(missing)}")
    bridge_problem = check_numpy_bridge()
    if bridge_problem:
        problems.append(f"torch/numpy bridge broken — {bridge_problem}")

    print()
    if problems:
        print("BLOCKERS")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("OK — environment satisfies Phase 1 requirements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
