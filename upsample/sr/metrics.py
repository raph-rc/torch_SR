"""Quality metrics for evaluating super-resolution output.

Matches the PSNR formula used in the Matlab reference ``test3D.m``
(Manjon & Coupe 2010)::

    psnr = 20 * log10(R / rmse_masked)

where ``R = reference.max() - reference.min()`` and the RMSE is
computed only over voxels with ``reference > mask_threshold``
(``umbral = 10`` in the Matlab code).
"""

from __future__ import annotations

import numpy as np


def compute_psnr(
    reference: np.ndarray,
    estimate: np.ndarray,
    mask_threshold: float = 10.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio between a reference and an estimate.

    Uses the Manjon 2010 formulation::

        R    = reference.max() - reference.min()
        mask = reference > mask_threshold
        RMSE = sqrt(mean((reference[mask] - estimate[mask])^2))
        PSNR = 20 * log10(R / RMSE)

    Background voxels (``reference <= mask_threshold``) are excluded from
    the RMSE calculation, matching ``ind = find(ima1(:) > umbral)`` in
    ``test3D.m`` with ``umbral = 10``.

    Args:
        reference: Ground-truth volume, shape (D, H, W).
        estimate:  Reconstructed volume, same shape as *reference*.
        mask_threshold: Voxels at or below this value are treated as
            background and excluded from RMSE.  Default 10.0 matches
            the Matlab ``umbral`` parameter.

    Returns:
        PSNR in decibels (float).  Returns ``0.0`` if no foreground voxels
        are found; returns ``inf`` if RMSE is exactly zero.

    Raises:
        ValueError: If *reference* and *estimate* have different shapes.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)

    if reference.shape != estimate.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs "
            f"estimate {estimate.shape}"
        )

    R = float(reference.max()) - float(reference.min())
    if R == 0.0:
        return 0.0

    mask = reference > mask_threshold
    if not mask.any():
        return 0.0

    diff = reference[mask] - estimate[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if rmse == 0.0:
        return float("inf")

    return float(20.0 * np.log10(R / rmse))
