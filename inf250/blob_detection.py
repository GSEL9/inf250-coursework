# -*- coding: utf-8 -*-
#
# blob_detection.py
#

"""Framework for image BLOB detection."""


import sys

import numpy as np

from skimage import measure


def blob_detection(
        image, th=None, black_blobs=True, min_area=None, max_area=None
    ):
    """Perform detection of BLOBs in binary image.

    Args:
        image (array-like): Input image.
        th (int): Threshoding value for image binarization. Applies Otsu by
            default.
        black_blobs (bool): Specify label of BLOBs.
        min_area (float): Removes BLOBs if smaller than min area.
        max_area (float): Removes BLOBs if larger than max area.

    Returns:
        (tuple):

    """
    shape = np.shape(image)
    labeled = region_labeling(image, th=th,  black_blobs=black_blobs)
    no_blobs = labeled.max()

    min_area = -np.float('inf') if min_area is None else min_area
    max_area = np.float('inf') if max_area is None else max_area


    # Find areas:
    areas = histogram(labeled)[1:]
    areas = np.trim_zeros(areas, 'b') # Remove trailing zeros
    #-----WRITE A SHORT DESRIPTION OF WHY THIS GIVES THE AREA OF EACH BLOB-----#
    # Considering a binary image displaying one blob. The histogram of such an
    # image would contain the number of background and foreground pixels in
    # total. Removing the background pixels would result in only the
    # blob region pixels to remain. Then the area of the blob would be
    # equivalent to the number of pixels left in the histogram as given by
    # the zeroth order moment of a region.
    #------------------------------END DESCRIPTION-----------------------------#

    # Filter blobs:
    # Array with the blob-labels to consider
    wanted_blobs = [True]*no_blobs

    for i in range(no_blobs):
        # The i-th value in the `wanted_blobs list should be True if the
        # corresponding blob has an area within the range we consider.

        if not min_area <= areas[i] <= max_area:
            wanted_blobs[i] = False

    # Remove unwanted blobs and recompute areas
    labeled = _remove_and_relabel_blobs(labeled, wanted_blobs)
    areas = histogram(labeled)[1:]
    areas = np.trim_zeros(areas, 'b') # Remove trailing zeros

    blobs = [{'area' : None} for _ in range(no_blobs)]

    no_blobs = labeled.max()
    for blob in range(no_blobs):
        # Blob labels start at 1
        blob_id = blob + 1

        single_blob = _select_single_blob(labeled, blob_id)
        blobs[blob]['area'] = areas[blob]

    return blobs, labeled


def _remove_and_relabel_blobs(labeled, wanted_blobs):
    # Blob detection auxillary function to remove unwanted BLOBs.
    labeled = labeled.copy()
    wanted_blobs = np.array(wanted_blobs)
    no_blobs = len(wanted_blobs)
    unwanted_blobs = np.arange(1, no_blobs+1)[np.logical_not(wanted_blobs)]
    wanted_blobs = np.arange(1, no_blobs+1)[wanted_blobs]

    for unwanted_blob in unwanted_blobs:
        labeled[labeled == unwanted_blob] = 0

    for new_label, wanted_blob in enumerate(wanted_blobs):
        new_label += 1
        labeled[labeled == wanted_blob] = -new_label

    return -labeled


def _select_single_blob(image, color=None):
    # Blobl detection auxillary function for selecting specific BLOB.

    blob = np.copy(image)
    blob[image[:, :] != color] = 0

    return blob
