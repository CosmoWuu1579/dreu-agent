"""Extract a Qiskit API reference straight from the installed package.

Pure Python introspection (`inspect`) -- no LLM, no API calls, no network.
This is the astronaut "no-LLM" path: the signatures and docstrings you see on
docs.quantum.ibm.com are just rendered docstrings, and `inspect` reads them
directly out of the installed qiskit package.

Run:
    python qiskit_reference.py

Output:
    qiskit_reference.json  ->  { "qiskit.circuit.library.ZZFeatureMap": {
                                     "signature": "(feature_dimension, reps=2, ...)",
                                     "docstring": "The second-order Pauli-Z ..."
                                 }, ... }

We deliberately do NOT store each object's full source code: it was ~70% of the
file size and the keyword lookup only needs the signature + docstring.
"""

import importlib
import inspect
import json
import pkgutil

import qiskit

# The public modules that matter for feature-map / DQC1 work. Add more as needed.
MODULES = [
    "qiskit.circuit",
    "qiskit.circuit.library",  # gates, ZZFeatureMap, PauliFeatureMap, NLocal, ...
    "qiskit.quantum_info",  # DensityMatrix, Statevector, Pauli, ...
]

# Skip test / benchmark modules while walking sub-packages.
EXCLUDE = ("test", "conftest", "benchmark")


def parse_module(root_name: str) -> dict[str, dict]:
    """Introspect one top-level module and every public class/function under it."""
    root = importlib.import_module(root_name)
    result: dict[str, dict] = {}

    # Walk sub-packages so we reach classes defined deep inside the package,
    # then keep only the ones re-exported on the top-level module (the public API).
    submodules = [root_name]
    if hasattr(root, "__path__"):
        for _, mod_name, _ in pkgutil.walk_packages(root.__path__, root_name + "."):
            if any(x in mod_name for x in EXCLUDE):
                continue
            submodules.append(mod_name)

    for mod_name in submodules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        for name, obj in inspect.getmembers(
            mod, lambda o: inspect.isclass(o) or inspect.isfunction(o)
        ):
            # Only keep names exposed on the top-level module (e.g. qiskit.circuit.QuantumCircuit).
            # This dedups internal re-imports and keeps us on the documented surface.
            if not hasattr(root, name):
                continue

            key = f"{root_name}.{name}"
            if key in result:
                continue

            try:
                signature = str(inspect.signature(obj))
            except (ValueError, TypeError):
                signature = None  # C/Rust-accelerated objects have no Python signature

            result[key] = {
                "signature": signature,
                "docstring": inspect.getdoc(obj) or "",
            }

    return result


def main() -> None:
    print(f"Qiskit version: {qiskit.__version__}")

    data: dict[str, dict] = {}
    for root_name in MODULES:
        module_data = parse_module(root_name)
        print(f"  parsed {len(module_data):>4} entries from {root_name}")
        data.update(module_data)

    out_path = "qiskit_reference.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Total {len(data)} entries written to {out_path}")


if __name__ == "__main__":
    main()
