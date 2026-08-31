"""Run the installed EZminc programs once and record what they said.

    python3 -m tests.regenerate_reference

Requires ``noise_estimate``, ``mincnlm`` and ``minc_anlm`` on ``PATH``;
nothing else in the suite does. The programs are deterministic (at one
thread), so re-running this when nothing has changed leaves
``git diff tests/reference`` empty, which is the check that the recorded
answers are still their answers.

Adapted from ``torch_n3/tests/regenerate_reference.py``'s ``ezminc_cases()``
function only: this repository has no N3/histogram/spline pipeline, so
everything else in that script (the other ``*_cases`` functions, the
platform-reference volume) does not apply here.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

from sr.minc.io import load_minc_volume_np
from tests import reference

#: Everything this script needs to be able to run.
PROGRAMS = ["noise_estimate", "mincnlm", "minc_anlm"]

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"


def main():
    missing = [p for p in PROGRAMS if shutil.which(p) is None]
    if missing:
        sys.exit("not on PATH: %s" % ", ".join(missing))

    workspace = Workspace(pathlib.Path(
        tempfile.mkdtemp(prefix="ezminc_reference_")))
    arrays, scalars = {}, {}

    ezminc_cases(workspace, arrays, scalars)

    reference.save(arrays, scalars, produced_by=_versions())

    print("wrote %s" % reference.ARRAYS)
    for key in sorted(arrays):
        print("  %-44s %s" % (key, tuple(arrays[key].shape)))
    for key in sorted(scalars):
        print("  %-44s %s" % (key, scalars[key]))


def ezminc_cases(workspace, arrays, scalars):
    """The EZminc NLM programs: ``noise_estimate``, ``mincnlm``, ``minc_anlm``.

    ``mincnlm`` must be recorded with ``-mt 1``: its default of four threads
    is nondeterministic. ``minc_anlm`` defaults to one thread already, and is
    recorded with ``--float`` so the reference is not rounded through the
    input's 16-bit quantisation.
    """
    # Fed to the programs *and* read back by legacy_ezminc's own tests
    # (through the ``chunk``/``chunk_mask`` fixtures) from the same file
    # bytes, unmodified -- a round trip through this repository's own MINC
    # writer would re-quantise the 16-bit storage on a different scaling
    # grid and move a sensitive, nonlinear filter like mincnlm/minc_anlm by
    # more than the arithmetic itself does (see the project CLAUDE.md's
    # quantisation pitfall).
    source = str(DATA / "chunk.mnc")
    masked = str(DATA / "chunk_mask.mnc")

    # The scalar answers of `noise_estimate`: Rician by default, Gaussian
    # with --gauss, and a run constrained by the object mask. SNR is the
    # printed mean/noise ratio, so the test can recover mean_signal from the
    # two.
    scalars["noise_estimate.chunk_noise"] = float(
        workspace.run("noise_estimate", "--noise", source))
    scalars["noise_estimate.chunk_snr"] = float(
        workspace.run("noise_estimate", "--snr", source))
    scalars["noise_estimate.chunk_noise_gauss"] = float(
        workspace.run("noise_estimate", "--noise", "--gauss", source))
    scalars["noise_estimate.chunk_snr_gauss"] = float(
        workspace.run("noise_estimate", "--snr", "--gauss", source))
    scalars["noise_estimate.chunk_mask_noise"] = float(
        workspace.run("noise_estimate", "--noise", "--mask", masked, source))
    scalars["noise_estimate.chunk_mask_snr"] = float(
        workspace.run("noise_estimate", "--snr", "--mask", masked, source))

    # `mincnlm`: one run with an explicit sigma and one with the automatic
    # estimate, both deterministic at one thread.
    workspace.run("mincnlm", "-mt", 1, "-clobber", "-sigma", 20000.0,
                  source, workspace.at("mincnlm_sigma.mnc"))
    arrays["mincnlm.chunk_sigma"] = reference.as_volume(
        workspace.read("mincnlm_sigma.mnc"))

    workspace.run("mincnlm", "-mt", 1, "-clobber",
                  source, workspace.at("mincnlm_auto.mnc"))
    arrays["mincnlm.chunk_auto"] = reference.as_volume(
        workspace.read("mincnlm_auto.mnc"))

    # `minc_anlm` with its defaults (search radius 2, patch radius 1).
    workspace.run("minc_anlm", "--search", 2, "--patch", 1, "--float",
                  source, workspace.at("minc_anlm.mnc"))
    arrays["minc_anlm.chunk"] = reference.as_volume(
        workspace.read("minc_anlm.mnc"))


class Workspace:
    """A temporary directory plus the verbs this script needs there.

    File arguments are always absolute: several legacy programs derive their
    own temporary file names from the output path, and do not consistently
    resolve a relative one against the working directory (see the project
    CLAUDE.md pitfall on this).
    """

    def __init__(self, path):
        self.path = path

    def at(self, name):
        """The absolute path of ``name`` inside the workspace."""
        return str(self.path / name)

    def read(self, name):
        data, _ = load_minc_volume_np(self.at(name))
        return data

    def run(self, *command):
        """Run an installed EZminc/MINC program, raising if it fails."""
        result = subprocess.run([str(c) for c in command], cwd=str(self.path),
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit("%s failed (%d)\n%s\n%s"
                             % (command[0], result.returncode,
                                result.stdout[-2000:], result.stderr[-2000:]))
        return result.stdout


def _versions():
    """What produced these numbers, for the record."""
    version = subprocess.run(["mincinfo", "-version"], capture_output=True,
                             text=True).stdout.strip()
    return version.replace("\n", "; ") or "unknown MINC build"


if __name__ == "__main__":
    main()
