"""
PyTorch re-implementation of Non-local MRI upsampling (Manjon 2010).

Reference
---------
Manjon JV, Coupe P, Buades A, Collins DL, Robles M.
MRI superresolution using self-similarity and image priors.
*Int J Biomed Imaging*, 2010.
"""

from sr.core import nlm_upsample

__all__ = ["nlm_upsample"]
