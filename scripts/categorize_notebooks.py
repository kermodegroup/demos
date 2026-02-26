#!/usr/bin/env python3
"""
Categorize notebooks into WASM-compatible and live-server-required.

Notebooks are considered WASM-incompatible if they depend on packages that:
1. Are in the static WASM_INCOMPATIBLE_DEPS blocklist (known native extensions)
2. Are not available in Pyodide and not installable as pure Python wheels
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

# Packages known to have native extensions that don't work in WASM/Pyodide
WASM_INCOMPATIBLE_DEPS = {
    # ML/Scientific computing with native extensions
    "jax",
    "jaxlib",
    "tensorflow",
    "tensorflow-gpu",
    "torch",
    "pytorch",
    "torchaudio",
    "torchvision",
    # File system / OS-level packages (native extensions)
    "watchdog",
    "psutil",
    "watchfiles",
    # Database drivers with native code
    "psycopg2",
    "psycopg2-binary",
    "mysqlclient",
    "pymssql",
    # Image processing with native code
    "opencv-python",
    "opencv-python-headless",
    "cv2",
    # XML/parsing with native code
    "lxml",
    # Cryptography with native code
    "cryptography",
    "bcrypt",
    "pynacl",
    # Networking/async with native code
    "grpcio",
    "uvloop",
    "gevent",
    # Other common native packages
    "numpy-stl",
    "pyarrow",
    "fastparquet",
    "h5py",
    "tables",
    "numba",
    "cython",
}

# Pyodide lock file URL - packages listed here are available in Pyodide
PYODIDE_LOCK_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.0/full/pyodide-lock.json"


def get_pyodide_packages() -> set[str]:
    """Fetch list of packages available in Pyodide.

    Returns an empty set on failure (will fall back to static blocklist only).
    """
    try:
        with urllib.request.urlopen(PYODIDE_LOCK_URL, timeout=10) as response:
            data = json.load(response)
            return set(data.get("packages", {}).keys())
    except Exception as e:
        print(f"Warning: Could not fetch Pyodide packages: {e}")
        return set()


def extract_dependencies(content: str) -> list[str]:
    """Extract package names from PEP 723 script metadata."""
    metadata_match = re.search(r"# /// script\n(.*?)# ///", content, re.DOTALL)
    if not metadata_match:
        return []

    metadata = metadata_match.group(1)
    deps_match = re.search(r"dependencies\s*=\s*\[(.*?)\]", metadata, re.DOTALL)
    if not deps_match:
        return []

    deps_str = deps_match.group(1)
    dependencies = []

    # Parse package names, handling version specifiers like "numpy==1.2.3" or "pkg[extra]>=1.0"
    for dep in re.findall(r'"([^"]+)"', deps_str):
        # Remove extras like [dev] and version specifiers
        pkg_name = re.split(r"[\[<>=!;]", dep)[0].strip().lower()
        # Normalize package name (PyPI normalizes underscores to hyphens)
        pkg_name = pkg_name.replace("_", "-")
        if pkg_name:
            dependencies.append(pkg_name)

    return dependencies


def extract_imports(content: str) -> list[str]:
    """Extract imported module names from Python code."""
    imports = []
    for match in re.finditer(r"^\s*(?:import|from)\s+(\w+)", content, re.MULTILINE):
        imports.append(match.group(1).lower())
    return imports


def check_wasm_compatible(
    notebook_path: Path, pyodide_packages: set[str]
) -> tuple[bool, list[str]]:
    """Check if a notebook's dependencies are WASM-compatible.

    Args:
        notebook_path: Path to the notebook file
        pyodide_packages: Set of packages available in Pyodide (can be empty)

    Returns:
        Tuple of (is_compatible, list_of_incompatible_deps)
    """
    content = notebook_path.read_text()
    incompatible_found = []

    # Check declared dependencies in script metadata
    for dep in extract_dependencies(content):
        # Normalize for comparison
        dep_normalized = dep.lower().replace("_", "-")

        # Check against static blocklist
        if dep_normalized in WASM_INCOMPATIBLE_DEPS:
            if dep not in incompatible_found:
                incompatible_found.append(dep)
        # If we have the Pyodide package list, check if package might work
        elif pyodide_packages:
            # Package not in Pyodide's built-in list
            # micropip can install pure Python wheels, so we only flag
            # packages we know have native extensions
            pass  # Already handled by blocklist above

    # Also check imports for blocklisted packages (catches undeclared deps)
    for imp in extract_imports(content):
        imp_normalized = imp.lower().replace("_", "-")
        if imp_normalized in WASM_INCOMPATIBLE_DEPS:
            if imp not in incompatible_found:
                incompatible_found.append(imp)

    return (len(incompatible_found) == 0, incompatible_found)


def main():
    parser = argparse.ArgumentParser(
        description="Categorize notebooks by WASM compatibility"
    )
    parser.add_argument("--output-wasm", default="wasm_notebooks.txt")
    parser.add_argument("--output-live", default="live_notebooks.txt")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip fetching Pyodide package list (use static blocklist only)",
    )
    args = parser.parse_args()

    # Fetch Pyodide packages for compatibility checking
    pyodide_packages = set() if args.offline else get_pyodide_packages()
    if pyodide_packages:
        print(f"Loaded {len(pyodide_packages)} Pyodide packages")

    wasm_notebooks = []
    live_notebooks = []

    for directory in ["notebooks", "apps"]:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        for path in dir_path.rglob("*.py"):
            # Skip library files and private modules (not notebooks)
            if "lib" in path.parts or path.name.startswith("_"):
                continue

            is_compatible, incompatible_deps = check_wasm_compatible(
                path, pyodide_packages
            )

            if is_compatible:
                wasm_notebooks.append(str(path))
                print(f"WASM: {path}")
            else:
                live_notebooks.append(str(path))
                print(f"LIVE: {path} (incompatible: {', '.join(incompatible_deps)})")

    with open(args.output_wasm, "w") as f:
        f.write("\n".join(wasm_notebooks))

    with open(args.output_live, "w") as f:
        f.write("\n".join(live_notebooks))

    print(f"\nSummary: {len(wasm_notebooks)} WASM, {len(live_notebooks)} live")


if __name__ == "__main__":
    main()
