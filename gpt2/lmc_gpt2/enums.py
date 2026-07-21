from enum import Enum

class MatrixType(Enum):
    PERM = "permutation"
    SOFT_PERM = "soft_permutation"
    ORTHO = "orthogonal"

class SamplerType(Enum):
    GAUSSIAN = "gaussian"
    UNI = "uniform"
    NARROW_UNI = "narrow_uniform"
    NARROW_UNI_BIASED = "narrow_uniform_biased"