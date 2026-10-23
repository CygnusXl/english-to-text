"""Milestone 1: read-only environment and project inspection.

Run from the repository root:

    python3 scripts/check_env.py

This script only READS information. It never installs, upgrades, downloads,
or modifies anything. Paste its full output back into the chat.
"""

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Packages this project is likely to use. We only check whether they are
# installed and which version; we never assume a version.
PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "pillow",
    "datasets",
    "huggingface-hub",
    "jiwer",
    "editdistance",
    "matplotlib",
    "pandas",
    "tqdm",
    "pytest",
    "streamlit",
    "gradio",
]

# Folders we will not walk into when listing the project tree.
SKIP_DIRS = {".git", "venv", ".venv", "env", "__pycache__", "node_modules", ".pytest_cache"}


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def run(cmd: list[str]) -> str:
    """Run a command and return its output, or a short note if it fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (result.stdout or result.stderr).strip()
    except FileNotFoundError:
        return f"(command not found: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return f"(timed out: {' '.join(cmd)})"


def check_python() -> None:
    section("Python and virtual environment")
    print("python executable :", sys.executable)
    print("python version    :", platform.python_version())
    print("platform          :", platform.platform())
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print("inside a venv     :", in_venv)
    print("VIRTUAL_ENV       :", os.environ.get("VIRTUAL_ENV", "none"))


def check_torch() -> None:
    section("PyTorch and CUDA")
    try:
        import torch
    except ImportError:
        print("torch is NOT installed in this Python environment.")
        return
    print("torch version     :", torch.__version__)
    print("built for CUDA    :", torch.version.cuda)
    print("cuda available    :", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("gpu name          :", props.name)
        print("gpu total memory  : %.2f GB" % (props.total_memory / 1024**3))
        free, total = torch.cuda.mem_get_info()
        print("gpu free memory   : %.2f GB of %.2f GB" % (free / 1024**3, total / 1024**3))
        print("cudnn enabled     :", torch.backends.cudnn.enabled)
    else:
        print("No usable GPU. Check the NVIDIA driver and that torch is a CUDA build.")


def check_nvidia_smi() -> None:
    section("nvidia-smi")
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found on PATH.")
        return
    print(
        run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
                "--format=csv",
            ]
        )
    )


def check_packages() -> None:
    section("Installed packages (None = not installed)")
    for name in PACKAGES:
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            version = None
        print(f"{name:26s} {version}")


def check_git(root: Path) -> None:
    section("Git")
    if shutil.which("git") is None:
        print("git not found on PATH.")
        return
    if not (root / ".git").exists():
        print("This folder is not a git repository yet.")
        return
    print("branch/status :\n" + run(["git", "-C", str(root), "status", "-sb"]))
    print("remotes       :\n" + run(["git", "-C", str(root), "remote", "-v"]))


def check_tree(root: Path, max_depth: int = 2) -> None:
    section(f"Project tree (depth {max_depth}) in {root}")
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        depth = len(Path(dirpath).parts) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
        indent = "  " * depth
        print(f"{indent}{Path(dirpath).name}/")
        for filename in sorted(filenames):
            print(f"{indent}  {filename}")


def check_dataset_cache() -> None:
    section("Hugging Face cache (is OpenHand-Synth already downloaded?)")
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if not hub.exists():
        print("No Hugging Face hub cache found.")
        return
    entries = sorted(p.name for p in hub.iterdir())
    print("\n".join(entries) if entries else "(cache is empty)")
    print("OpenHand-Synth cached:", any("OpenHand-Synth" in e for e in entries))


def check_disk(root: Path) -> None:
    section("Disk space")
    total, used, free = shutil.disk_usage(root)
    print("free: %.1f GB of %.1f GB" % (free / 1024**3, total / 1024**3))


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    check_python()
    check_torch()
    check_nvidia_smi()
    check_packages()
    check_git(root)
    check_tree(root)
    check_dataset_cache()
    check_disk(root)
    print("\nDone. Paste this entire output into the chat.")


if __name__ == "__main__":
    main()
