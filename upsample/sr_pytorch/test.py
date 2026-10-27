import torch
import torch.nn.functional as F
from core import _local_std


def local_std_no_unsqueeze(vol: torch.Tensor) -> torch.Tensor:
    """Population std in a 3×3×3 neighbourhood (reflect BC)."""
    vol4 = vol.unsqueeze(0)
    padded    = F.pad(vol4, [1] * 6, mode="reflect")
    print(f"padded shape: {padded.shape}")
    print(f"padded : {padded}")
    padded_sq = F.pad(vol4.pow(2), [1] * 6, mode="reflect")
    print(f"padded_sq shape: {padded_sq.shape}")
    print(f"padded_sq : {padded_sq}")
    lmean    = F.avg_pool3d(padded,    kernel_size=3, stride=1, padding=0)
    print(f"lmean shape: {lmean.shape}")
    print(f"lmean : {lmean}")
    lsqmean  = F.avg_pool3d(padded_sq, kernel_size=3, stride=1, padding=0)
    print(f"lsqmean shape: {lsqmean.shape}")
    print(f"lsqmean : {lsqmean}")
    return (lsqmean - lmean.pow(2)).clamp(min=0.0).sqrt().squeeze()

test_nb = torch.rand(3, 3, 3)
vibe = _local_std(test_nb)
no_unsqueeze = local_std_no_unsqueeze(test_nb)
if torch.equal(vibe, no_unsqueeze):
    print("equal computation")
else:
    print("messed up")
    print(f"vibe: {vibe}")
    print(f"no_unsqueeze: {no_unsqueeze}")
