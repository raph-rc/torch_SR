"""What the installed EZminc programs answered, recorded once.

Ported from ``torch_n3/tests/reference.py``, scoped to this repository's one
use: the answers of ``noise_estimate``/``mincnlm``/``minc_anlm`` are
deterministic (at one thread), so running them on every test run gains
nothing and costs the whole MINC toolkit plus a subprocess per test. They
live in ``tests/reference/legacy_ezminc.npz``/``.json`` instead.
``python3 -m tests.regenerate_reference`` rebuilds the file; if the programs
still give the same answers, ``git diff`` is empty.

Whole volumes are stored at ``float32``, matching what the tools read/write
(16-bit MINC in, ``float`` out); the small scalar answers are kept at full
precision in the ``.json`` sidecar.
"""

import json
import os

import numpy as np
import torch

DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
ARRAYS = os.path.join(DIRECTORY, "legacy_ezminc.npz")
SCALARS = os.path.join(DIRECTORY, "legacy_ezminc.json")

MISSING = ("%s is not recorded in tests/reference/. Run\n"
           "    python3 -m tests.regenerate_reference\n"
           "with the MINC toolkit's noise_estimate/mincnlm/minc_anlm on PATH "
           "to produce it.")


def _numpy(values):
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


class LegacyOutputs:
    """The recorded answers, keyed by ``program.case``."""

    def __init__(self, arrays, scalars):
        self._arrays = arrays
        self._scalars = scalars

    def __getitem__(self, key):
        if key not in self._arrays:
            raise KeyError(MISSING % key)
        return torch.as_tensor(self._arrays[key], dtype=torch.float64)

    def scalar(self, key):
        if key not in self._scalars:
            raise KeyError(MISSING % key)
        return self._scalars[key]

    def keys(self):
        return sorted(list(self._arrays) + list(self._scalars))


def load():
    if not os.path.exists(ARRAYS):
        raise FileNotFoundError(MISSING % "tests/reference/legacy_ezminc.npz")
    with np.load(ARRAYS) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with open(SCALARS) as handle:
        scalars = json.load(handle)["values"]
    return LegacyOutputs(arrays, scalars)


def as_volume(values):
    """Mark an array as a whole volume, storable at ``float32``."""
    return np.asarray(_numpy(values), dtype=np.float32)


def save(arrays, scalars, produced_by):
    """Write the recorded answers. Used only by the regeneration script."""
    os.makedirs(DIRECTORY, exist_ok=True)
    np.savez_compressed(ARRAYS, **{key: _numpy(value)
                                   for key, value in arrays.items()})
    with open(SCALARS, "w") as handle:
        json.dump({"produced_by": produced_by, "values": scalars}, handle,
                  indent=2, sort_keys=True)
        handle.write("\n")
