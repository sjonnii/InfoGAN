"""Module containing helper methods for dealing with OME-TIFF files"""

import os
import tarfile
from io import BytesIO
from tifffile import TiffFile
import numpy as np
from .azure import may_download_file


# MAX_DIM_RATIO = 1.1
# MAX_DIM_SIZE = 70
# MIN_DIM_SIZE = 50

MAX_DIM_RATIO = 2
MAX_DIM_SIZE = 300
MIN_DIM_SIZE = 10


class CellImage(object):
    """Encapsulates cell image information.

    Args:
        channel_shape - the shape of a channel image
    """
    def __init__(self, channel_shape):
        height, width = channel_shape
        self._data = np.zeros((height, width, 2), 'uint16')
        self._metadata = [None, None]

    def set_channel(self, channel, data):
        """Sets the data for a channel.

        Args:
            channel - the channel index
            data - the channel data as (data, metadata)
        """
        data, metadata = data
        self._data[:, :, channel] = data
        self._metadata[channel] = metadata

    @property
    def data(self):
        """The pixel data"""
        return self._data

    @property
    def metadata(self):
        """The image metadata"""
        return self._metadata

    @property
    def is_ready(self):
        """Whether all of the channels have been set"""
        return self._metadata[0] and self._metadata[1]

    def _compute_padding(self, pad_rows, pad_cols):
        """Compute a array of padding values.

        Description:
            Compute the moments of the edge pixels of the array
            in each dimension, and then use these to generate an
            array of noise that can be used as padding.

        Args:
            pad_rows - the number of rows in the padding image
            pad_cols - the number of columns in the padding image

        Returns:
            padding array
        """
        rows, cols, _ = self._data.shape #the row and cols of the image we're processing
        padded = np.zeros((pad_rows, pad_cols, 2), 'uint16') #create an array for the padded image, size (pad_rows, pad_cols, 2)
        for i in range(self._data.shape[2]):
            if i == 0:
                array = self._data[:, :, i]
                values = [array[0, :].reshape(cols),
                        array[-1, :].reshape(cols),
                        array[1:-1, 0].reshape(rows-2),
                        array[1:-1, -1].reshape(rows-2)]
                values = np.concatenate(values)
                mean = values.mean()
                var = values.var()
                values = np.ma.masked_array(values, np.where((values - mean)**2 > 4*var, True, False))
                mean = values.mean()
                std = values.std()
                padded[:, :, i] = np.random.normal(mean, std,
                                                size=(pad_rows, pad_cols)).astype('uint16')
            if i == 1:
                padded[:,:, i] = np.ones(shape=(pad_rows, pad_cols)).astype('uint16') * 20
        return padded

    def pad(self, pad_rows, pad_cols):
        """Pads the data using a model of the edge pixels.

        Description:
            Compute the statistics of the edge pixels and then sample from a normal distribution
            to fill in the padding needed to standardize the shape.

        Args:
            pad_rows - the rows to pad to
            pad_cols - the columns to pad to
        """
        rows, cols, _ = self._data.shape
        padded = self._compute_padding(pad_rows, pad_cols)
        array = self._data
        if rows > pad_rows:
            top = (rows - pad_rows) // 2
            bottom = top + pad_rows
            array = array[top:bottom, :]
            rows = pad_rows

        if cols > pad_cols:
            left = (cols - pad_cols) // 2
            right = left + pad_cols
            array = array[:, left:right]
            cols = pad_cols

        top = (pad_rows - rows) // 2
        bottom = top + rows
        left = (pad_cols - cols) // 2
        right = left + cols
        padded[top:bottom, left:right] = array

        self._data = padded


def _read_channel(file, max_dim_ratio, min_dim_size, max_dim_size):
    """Read a channel OME-TIFF file.

    Args:
        file - the input file handle
        max_dim_ratio - the minimum dimension ratio
        min_dim_size - the minimum dimension size
        max_dim_size - the maximum dimension size

    Returns:
        (data, metadata) for the channel
    """
    try:
        with TiffFile(file) as tif:
            data = tif.pages[0].asarray()

            if _is_valid(data, max_dim_ratio, min_dim_size, max_dim_size):
                metadata = {}
                for tag in tif.pages[0].tags.values():
                    if tag.name == 'ImageDescription':
                        continue
                    else:
                        metadata[tag.name] = str(tag.value)

                return data, metadata

            return None
    except:
        return None
    
    print (counter)


def read_ome_tiff(name, max_dim_ratio=MAX_DIM_RATIO,
                  min_dim_size=MIN_DIM_SIZE, max_dim_size=MAX_DIM_SIZE):
    """Reads an OME-TIFF file from the disk as a numpy array and metadata dictionary.

    Description:
        This method parses the TIFF file and both grabs the image data as a numpy array and
        creates a metadata dictionary from TIFF header and OME data.

    Args:
        path - the path to the OME-TIFF file

    Keyword Args:
        max_dim_ratio - the maximum ratio between the larger and smaller dimensions [1.1]
        min_dim_size - the minimum dimension size [50]
        max_dim_size - the maximum dimension size [70]

    Returns:
        enumerable of (data, metadata)
    """
    skipped = 0
    path = may_download_file(name + ".tgz") #Get the path of the tgz file
    with tarfile.open(path, 'r:gz') as tar: #open the tgz file
        cell_images = {}
        for tarinfo in tar:
            if tarinfo.isdir():
                continue

            name = os.path.basename(tarinfo.name)
            if name.startswith('raw_'):
                name = name[4:]

            channel = 0 if 'Ch1' in name else 1
            name = name[:name.index('_')]

            with BytesIO(tar.extractfile(tarinfo).read()) as file:
                data = _read_channel(file, max_dim_ratio, min_dim_size, max_dim_size)
                if data:
                    if name not in cell_images:
                        cell_images[name] = CellImage(data[0].shape)

                    cell_image = cell_images[name]
                    cell_image.set_channel(channel, data)
                    if cell_image.is_ready:
                        del cell_images[name]
                        print(name)
                        yield cell_image

                else:
                    skipped += 1



    print("Skipped", skipped // 2, "images due to filter parameters")


def _is_valid(data, max_dim_ratio, min_dim_size, max_dim_size):
    """Performs simple tests for validity on the cell image.

    Args:
        data - a HW numpy array
        max_dim_ratio - the ratio between the larger and smaller dimensions
        min_dim_size - the minimum dimension size
        max_dim_size - the maximum dimension size

    Returns:
        whether the array is valid based upon simple tests
    """
    rows, cols = data.shape

    if max(rows, cols) / min(rows, cols) > max_dim_ratio:
        return False

    if min(rows, cols) < min_dim_size:
        return False

    if max(rows, cols) > max_dim_size:
        return False

    return True
