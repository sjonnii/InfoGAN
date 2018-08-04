"""Data handling module.

Downloads files from azure, provides OME-TIFF file processing.
"""

from .azure import may_download_file, upload_file
from .ome_tiff import read_ome_tiff, CellImage, MAX_DIM_RATIO, MIN_DIM_SIZE, MAX_DIM_SIZE
