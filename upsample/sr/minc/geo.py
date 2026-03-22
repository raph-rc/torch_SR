import numpy as np
import math as m


def decompose(aff):
    """Decompose affine matrix into start, step and direction cosines."""
    (u, s, vh) = np.linalg.svd(aff[0:3, 0:3])
    dir_cos = u @ vh
    step  = np.diag(np.linalg.inv(dir_cos) @ aff[0:3, 0:3])
    start = np.squeeze(np.asarray(np.linalg.inv(dir_cos) @ aff[0:3, 3]))
    return start, step, dir_cos


def compose(start, step, dir_cos):
    """Create affine matrix from start, step and direction cosines."""
    aff = np.eye(4)
    aff[0:3, 0:3] = dir_cos @ np.diag(step)
    aff[0:3, 3]   = dir_cos @ start
    return aff


def create_v2p_matrix(shape):
    """Create voxel-to-PyTorch normalised coordinate matrix."""
    v2p = np.diag([2/shape[2], 2/shape[1], 2/shape[0], 1])
    v2p[0:3, 3] = (1/shape[2]-1, 1/shape[1]-1, 1/shape[0]-1)
    return v2p


def create_rotation_matrix(rot):
    """Create 4×4 rotation matrix from (rx, ry, rz) Euler angles."""
    affine_x = np.eye(4)
    sin_, cos_ = m.sin(rot[0]), m.cos(rot[0])
    affine_x[1, 1], affine_x[1, 2] = cos_, -sin_
    affine_x[2, 1], affine_x[2, 2] = sin_,  cos_
    affine_y = np.eye(4)
    sin_, cos_ = m.sin(rot[1]), m.cos(rot[1])
    affine_y[0, 0], affine_y[0, 2] =  cos_, sin_
    affine_y[2, 0], affine_y[2, 2] = -sin_, cos_
    affine_z = np.eye(4)
    sin_, cos_ = m.sin(rot[2]), m.cos(rot[2])
    affine_z[0, 0], affine_z[0, 1] = cos_, -sin_
    affine_z[1, 0], affine_z[1, 1] = sin_,  cos_
    return affine_x @ affine_y @ affine_z


def create_scale_matrix(scale):
    """Create 4×4 diagonal scaling matrix."""
    return np.diag([*scale, 1.0])


def create_translation_matrix(shift):
    """Create 4×4 translation matrix."""
    affine = np.eye(4)
    affine[0:3, 3] = shift
    return affine


def create_shear_matrix(shear):
    """Create 4×4 shear matrix from 6 shear components."""
    affine = np.eye(4)
    affine[0, 1], affine[0, 2] = shear[0], shear[1]
    affine[1, 0], affine[1, 2] = shear[2], shear[3]
    affine[2, 0], affine[2, 1] = shear[4], shear[5]
    return affine


def create_transform(rot, scale, shift, shear):
    """Compose full transform: T @ Sh @ S @ R."""
    return (create_translation_matrix(shift)
            @ create_shear_matrix(shear)
            @ create_scale_matrix(scale)
            @ create_rotation_matrix(rot))
