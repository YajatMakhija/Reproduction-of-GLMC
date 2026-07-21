from enum import Enum


class MatrixType(Enum):
    PERM = "permutation"   # projected via Hungarian assignment (straight-through)
    ORTHO = "orthogonal"   # projected via SVD (U @ Vt)
