"""The one-level forward DWT, as ``dwt_forward`` computes it.

Ported from ``torch_n3.blocks.dwt`` (the N3/torch_n3 project), where it backs
that project's noise estimator.  Nothing here is specific to that project:
this is a general-purpose 3-D one-level Daubechies-4 wavelet transform with
no MRI/N3-specific assumptions, kept here so :mod:`sr_pytorch.noise_estimate`
can depend on it without pulling in torch_n3.

Re-implements the transform EZminc's ``noise_estimate.cpp`` runs its input
through: ``dwt_forward`` (``dwt_utils.cpp``) pads the volume to the nearest
power-of-two grid, runs ``volume_dwt`` / ``pwt`` (``dwt.cpp``) once in z, y, x,
and cuts the result back into eight subbands.  ``dwt3d_db4`` is the per-step
filter; ``dwt_forward`` is the padding and quadrant extraction around it.

The C++ ``pwt`` reads its four taps at ``(2k + j) mod n``; the PyTorch port
reproduces that with a three-sample circular pad that is then dropped (the
``+3`` shift cancels the pad, so the convolution samples the same cyclically
wrapped taps), and places the split axis with ``movedim(-1, axis + 1)``.
"""

import torch
import torch.nn.functional as F

# The D4 coefficients (``dwt.cpp``'s ``C0..C3``) and the high-pass filter
# ``cr`` (``C3, -C2, C1, -C0``).
_DEC_LO = torch.tensor([-0.1294095225512604, 0.2241438680420134,
                        0.8365163037378079, 0.4829629131445341])
_DEC_HI = torch.tensor([-0.4829629131445341, 0.8365163037378079,
                        -0.2241438680420134, -0.1294095225512604])


def _conv_along(t, axis):
    """One ``pwt`` step: split ``t`` along ``axis`` into low and high halves.

    ``t`` may be any tensor; the leading dimensions are carried through.  The
    returned tensor leads with a 2 (low, high) and carries the halved axis at
    position ``axis + 1``, exactly as the C++ in-place split lays out.
    """
    t = t.movedim(axis, -1)
    shape = t.shape
    t = t.reshape(-1, 1, shape[-1])
    filters = torch.stack([_DEC_LO, _DEC_HI]).to(t.device, t.dtype)
    # Circularly pad by three samples and drop the first three: the taps of
    # ``pwt`` at ``(2k + j) mod n`` for ``j = 0..3`` line up with a stride-2
    # convolution of the cyclically shifted signal, and the shift is undone by
    # taking the pad off the front.
    t = F.pad(t, (3, 3), mode="circular")[..., 3:]
    lo = F.conv1d(t, filters[0].flip(0).view(1, 1, -1), stride=2)
    hi = F.conv1d(t, filters[1].flip(0).view(1, 1, -1), stride=2)
    out = torch.stack([lo, hi], dim=1)
    return (out.reshape(*shape[:-1], 2, -1)
            .movedim(-2, 0).movedim(-1, axis + 1))


def dwt3d_db4(x):
    """One level of the D4 transform on a ``(1, 1, P, P, P)`` tensor.

    Transforms z, then y, then x, each pass splitting the current axis into
    its low and high halves.  The bits of the quadrant index read x, y, z in
    that order (bit 0 is the x half), which is the order ``extract_quadrant``
    uses.  Returns the eight subbands keyed by their quadrant names.
    """
    xd = _conv_along(x, 2)
    xdh = torch.stack([_conv_along(xd[i], 3) for i in range(2)])
    xdhw = torch.stack([_conv_along(xdh[i, j], 4)
                        for i in range(2) for j in range(2)])
    xdhw = xdhw.reshape(2, 2, 2, *xdhw.shape[2:])
    names = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")
    return {n: xdhw[i // 4, (i // 2) % 2, i % 2]
            for i, n in enumerate(names)}


def _find_nearest_square_pow2(n):
    """``find_nearest_square_pow2``: the smallest power of two at or above
    ``n``, halved when ``n`` is exactly that power of two."""
    m = n
    p = 0
    while m != 0:
        m >>= 1
        p += 1
    if n == (1 << (p - 1)):
        p -= 1
    return 1 << p


def dwt_forward(volume):
    """The one-level forward DWT of ``volume``, as ``dwt_forward``.

    ``volume`` is a ``(z, y, x)`` tensor.  It is zero-padded, centred, onto a
    ``(P, P, P)`` grid with ``P`` the nearest power of two at or above the
    largest extent, transformed once, and cut back into the eight subbands,
    returned in quadrant order ``(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)``.
    Each subband is ``ceil(dim / 2)`` along every axis and stays on
    ``volume``'s device and dtype.
    """
    sz, sy, sx = volume.shape
    p = _find_nearest_square_pow2(max(sz, sy, sx))
    p0, p1, p2 = (p - sz) // 2, (p - sy) // 2, (p - sx) // 2
    buf = volume.new_zeros((p, p, p))
    buf[p0:p0 + sz, p1:p1 + sy, p2:p2 + sx] = volume
    subs = dwt3d_db4(buf.view(1, 1, p, p, p))
    q0, q1, q2 = p0 // 2, p1 // 2, p2 // 2
    osz = ((sz + 1) // 2, (sy + 1) // 2, (sx + 1) // 2)
    names = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")
    return [subs[n][0, 0, q0:q0 + osz[0], q1:q1 + osz[1], q2:q2 + osz[2]]
            for n in names]
