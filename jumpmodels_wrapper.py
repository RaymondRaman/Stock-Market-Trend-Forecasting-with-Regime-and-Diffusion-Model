# jumpmodels_wrapper.py
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
from jumpmodels.sparse_jump import SparseJumpModel
import sys
import pandas as pd
import numpy as np


def patch_jumpmodels():
    """Monkey-patch problematic type checks in jumpmodels"""
    try:
        # Import validation module
        from jumpmodels.utils import validation

        # Replace all Union-based type aliases with tuples
        validation.PD_TYPE = (pd.Series, pd.DataFrame)
        validation.NUMERICAL_OBJ_TYPE = (np.ndarray,) + validation.PD_TYPE
        validation.SER_ARR_TYPE = (np.ndarray, pd.Series)
        validation.DF_ARR_TYPE = (np.ndarray, pd.DataFrame)

        # Replace type-checking functions
        validation.is_ser_df = lambda obj: isinstance(obj, validation.PD_TYPE)
        validation.is_numerical = lambda obj: isinstance(
            obj, validation.NUMERICAL_OBJ_TYPE)

        print("Successfully patched jumpmodels type checks")

    except Exception as e:
        print(f"Patching failed: {e}")
        raise


# Apply patches before importing anything else
patch_jumpmodels()

# Now import the actual components

__all__ = ['SparseJumpModel', 'StandardScalerPD', 'DataClipperStd']
