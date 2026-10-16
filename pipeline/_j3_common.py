"""
Path anchor shared by the pipeline scripts, inherited from an earlier version of this code and reduced to what
this study uses: every script imports it to find the repository root and then changes into it.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(THIS_DIR)

# the repository root for the package imports (models.*, utils.*, pooling.*, journal2_pipeline.*) and this
# directory for the bare-name imports of the pipeline's own modules
for _p in (PROJ_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
