"""Rician (or Gaussian) noise level of a volume, as ``noise_estimate.cpp``.

Ported from ``torch_n3.blocks.noise_estimate``, unmodified apart from the
import of the wavelet transform (now :mod:`sr_pytorch.dwt`).  It has no
dependency on N3/torch_n3 beyond that sibling module.

Re-implements the EZminc estimator behind mincnlm's automatic sigma (Coupe et
al. 2009): a one-level wavelet decomposition, object detection on the LLL
subband (k-means over its histogram, or a supplied mask downsampled by two),
removal of the high-gradient edges, the median-absolute-deviation noise
estimate from |HHH|, and -- for Rician data -- the Koay correction of the bias
that rectified noise would otherwise carry.  Everything runs in float64 on the
device of ``volume``.
"""

import math

import torch

from sr_pytorch.dwt import dwt_forward


def _histogram(values, bins, lo, hi):
    """``build_histogram`` + ``histogram.seed``: a seeded, linear-split count.

    ``values`` is a tensor of voxels; the histogram spans ``(lo, hi)`` exactly
    as ``volume_min_max`` would set it.  Each voxel is split linearly between
    its bin and the next, except that the last bin absorbs the whole count
    rather than splitting out of range.  Normalised by the number of finite
    voxels.
    """
    values = values.reshape(-1)
    finite = torch.isfinite(values)
    if not finite.all():
        values = values[finite]
    k = bins / (hi - lo)
    idx = (k * (values - lo - 0.5 / k)).floor().long().clamp(0, bins - 1)
    counts = torch.zeros(bins, dtype=torch.float64, device=values.device)
    centers = lo + (idx + 0.5) / k
    last = idx == bins - 1
    counts[bins - 1] += float(last.sum())
    mid = ~last
    v = values[mid]
    d1 = (centers[mid] - v).abs()
    d2 = (centers[mid] + 1.0 / k - v).abs()
    w = d2 / (d1 + d2)
    counts.scatter_add_(0, idx[mid], w)
    counts.scatter_add_(0, (idx[mid] + 1).clamp(max=bins - 1), 1.0 - w)
    return counts / values.numel()


def _find_percentile(hist, lo, hi, pc):
    """``histogram.find_percentile``: linear interpolation on the cumulative.

    Walks the (normalised) bins until the accumulated probability reaches
    ``pc``, then interpolates between the last nonzero bin and the bin that
    crossed the mark, with the C++ ``+ 0.5 / k`` offset on the values.
    """
    k = hist.numel() / (hi - lo)
    acc = 0.0
    j = 0
    i = hist.numel() - 1
    for i in range(hist.numel()):
        prob = float(hist[i])
        acc += prob
        if acc >= pc:
            break
        if prob > 0.0:
            j = i

    def value(ii):
        return lo + (ii + 0.5) / k

    if i > 0:
        kk = (acc - pc) / float(hist[i])
        return value(i) * (1.0 - kk) + value(j) * kk + 0.5 / k
    kk = (acc - pc) / float(hist[0])
    return (value(0) + 0.5 / k) * (1.0 - kk) + lo * kk


def _k_means(hist, lo, hi, k_means=2, maxiter=10):
    """``simple_k_means``: k-means on the histogram, percentile-initialised.

    The means start at the 0.001 and 0.999 percentiles of ``hist`` (the C++
    uniform spread for two classes), and each iteration reclassifies every bin
    to its nearest mean and recomputes the weighted means.  Returns the mean
    of the two class means, which ``noise_estimate`` uses as its threshold.
    """
    vol_min = _find_percentile(hist, lo, hi, 0.001)
    vol_max = _find_percentile(hist, lo, hi, 0.999)
    mu = torch.tensor([vol_min, vol_max], dtype=torch.float64,
                      device=hist.device)
    centers = lo + (torch.arange(hist.numel(), dtype=torch.float64,
                                 device=hist.device) + 0.5) / (
        hist.numel() / (hi - lo))
    for _ in range(maxiter):
        cls = ((centers - mu[1]).abs() < (centers - mu[0]).abs()).long()
        for c in range(k_means):
            sel = cls == c
            w = hist[sel]
            if w.sum() > 0:
                mu[c] = (w * centers[sel]).sum() / w.sum()
    return float((mu[0] + mu[1]) / 2)


def _gmag(volume):
    """``calc_gradient_mag`` with unit steps: the gradient magnitude of
    ``volume``, each axis' derivative-or-Gaussian kernel built and applied in
    the frequency domain exactly as ``blur_volume`` does -- the kernel samples
    are FFT'd in place, so the filter applied is the FFT of the kernel."""
    sigma = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    a = 1.0 / (2.0 * sigma * sigma)

    def axes(is_deriv):
        out = volume
        for axis in range(3):
            n = volume.shape[axis]
            n2 = 2 * n
            kk = torch.arange(n2, dtype=torch.float64,
                              device=volume.device)
            kk2 = torch.where(kk < n / 2.0, kk, kk - n2)
            valid = (kk < n / 2.0) | (kk >= 3.0 * n / 2.0)
            g = torch.exp(-kk2 * kk2 * a)
            total = g[valid].sum()
            vals = (-kk2 / (sigma * sigma) * g) if is_deriv[axis] else g
            kern = torch.zeros(n2, dtype=torch.complex128,
                               device=volume.device)
            kern[valid] = (vals[valid] / total).to(torch.complex128)
            kern = kern.reshape([n2 if d == axis else 1 for d in range(3)])
            out = torch.fft.ifft(
                torch.fft.fft(out, n=n2, dim=axis)
                * torch.fft.fft(kern, n=n2, dim=axis), n=n2, dim=axis).real
            out = out.narrow(axis, 0, n)
        return out

    vdx = axes((False, False, True))
    vdy = axes((False, True, False))
    vdz = axes((True, False, False))
    return (vdx.pow(2) + vdy.pow(2) + vdz.pow(2)).sqrt()


def _downsample(mask):
    """``noise_estimate.cpp``'s mask downsampling: floor loops, rest zero."""
    mask = mask.to(torch.bool)
    sz, sy, sx = mask.shape
    out = torch.zeros(((sz + 1) // 2, (sy + 1) // 2, (sx + 1) // 2),
                      dtype=torch.bool, device=mask.device)
    out[:sz // 2, :sy // 2, :sx // 2] = (
        mask[:2 * (sz // 2):2, :2 * (sy // 2):2, :2 * (sx // 2):2])
    return out


def _epsi(snr):
    """``epsi``: the ratio of the observed to the true signal mean.

    Computed through the *scaled* Bessel functions (``i0e``/``i1e``, stable
    for large arguments): with ``x = snr^2/4`` the C++ ``exp(-snr^2/2) *
    ((2 + snr^2) I0(x) + snr^2 I1(x))^2`` collapses to ``((2 + snr^2) i0e(x) +
    snr^2 i1e(x))^2`` because ``exp(-snr^2/2) * exp(x)^2 = 1``.
    """
    if snr > 37.0:
        return 1.0
    s = torch.as_tensor(snr, dtype=torch.float64)
    x = s * s / 4.0
    s2 = s * s
    inner = (2.0 + s2) * torch.special.i0e(x) + s2 * torch.special.i1e(x)
    return float(2.0 + s2 - math.pi / 8.0 * inner * inner)


def _noise_correct(sig, nsig):
    """``noise_correct``: the Koay fixed-point correction of the noise level.

    Iterates the signal-to-noise fixed point to a ``1e-9`` change in the SNR
    (at most 500 iterations), then returns the corrected noise from the
    converged ``epsi``.
    """
    snr1 = sig / nsig
    for _ in range(500):
        snr2 = math.sqrt(_epsi(snr1) * (1.0 + sig * sig / (nsig * nsig))
                         - 2.0)
        if abs(snr1 - snr2) < 1e-9:
            break
        snr1 = snr2
    return math.sqrt(nsig * nsig / _epsi(snr1))


def noise_estimate(volume, mask=None, gaussian=False, hist_bins=2000):
    """Rician (or Gaussian) noise level of ``volume``, as the EZminc tool.

    ``volume`` is a ``(z, y, x)`` tensor.  ``mask``, when given, is the object
    mask used instead of the built-in k-means detection; it is downsampled by
    two inside the routine exactly as the program does.  Returns
    ``(noise, mean_signal)``, the population standard deviation of the noise
    and the mean of the signal it was estimated against.  The estimate is made
    on the lowest-resolution wavelet level of ``volume``, so the histograms
    have ``hist_bins`` bins there, not on the volume itself.
    """
    if volume.ndim != 3:
        raise ValueError("noise_estimate: volume must be 3-D (got %r)"
                         % (volume.shape,))
    volume = volume.to(torch.float64)
    sub = dwt_forward(volume)
    lll, hhh = sub[0], sub[7]

    lo, hi = float(lll.min()), float(lll.max())
    lll_hist = _histogram(lll, hist_bins, lo, hi)

    if mask is None:
        threshold = _k_means(lll_hist, lo, hi)
        lll_mask = lll > threshold
    else:
        lll_mask = _downsample(mask)

    gmag = _gmag(lll)
    gv = gmag[lll_mask]
    glo, ghi = float(gv.min()), float(gv.max())
    gmag_median = _find_percentile(
        _histogram(gv, hist_bins, glo, ghi), glo, ghi, 0.5)
    lll_mask = lll_mask & (gmag < gmag_median)

    hv = hhh.abs()[lll_mask]
    hlo, hhi = float(hv.min()), float(hv.max())
    nsig = _find_percentile(
        _histogram(hv, hist_bins, hlo, hhi), hlo, hhi, 0.5) / 0.6745

    full = (lll_mask.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            .repeat_interleave(2, dim=2))
    full = full[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
    mean_signal = float(volume[full].mean())

    if gaussian:
        return nsig, mean_signal
    return _noise_correct(mean_signal, nsig), mean_signal
