import warnings

from numba.core.errors import NumbaDeprecationWarning

"""Filter out annoying redundant warning messages"""
# Pretty much all of the UserWarnings are redundant and annoying
warnings.filterwarnings("ignore", category=UserWarning)
# Useless warning related to the handling of the SemanticKITTI dataset in mmdet3d
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
