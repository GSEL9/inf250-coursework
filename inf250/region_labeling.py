import sys

import numpy as np

from skimage import measure


def region_labeling(image, th=None, black_blobs=True, recursion_limit=10000):

    shape = np.shape(image)
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(recursion_limit)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')

    # Threshold image
    labeled = threshold(image, th=th).astype(int)
    labeled = 255-labeled if black_blobs else labeled
    labeled[labeled == 255] = -1

    # Label blobs
    blobs = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if labeled[i, j] == -1:
                blobs += 1
                flood_fill(labeled, y=i, x=j, colour=blobs)

    # Cleanup
    sys.setrecursionlimit(old_recursion_limit)

    return labeled


def flood_fill(image, y, x, colour):
    """Performs depth-first-search flood fill on the image at given location.
    """
    curr_colour = image[y, x]
    image[y, x] = colour

    if y > 0:
        if image[y-1, x] == curr_colour:
            flood_fill(image, y=y-1, x=x, colour=colour)
    if x > 0:
        if image[y, x-1] == curr_colour:
            flood_fill(image, y=y, x=x-1, colour=colour)
    if y+1 < image.shape[0]:
        if image[y+1, x] == curr_colour:
            flood_fill(image, y=y+1, x=x, colour=colour)
    if x+1 < image.shape[1]:
        if image[y, x+1] == curr_colour:
            flood_fill(image, y=y, x=x+1, colour=colour)


def _select_regions(image, wanted_blobs, foreground=255, background=0):

    filtered = np.zeros_like(image, dtype=np.uint8)

    num_blobs = len(wanted_blobs)

    unwanted = np.arange(1, num_blobs + 1)[np.logical_not(wanted_blobs)]
    wanted = np.arange(1, num_blobs + 1)[wanted_blobs]

    # Remove unwanted blobs:
    for unwanted_blob in unwanted:
        filtered[image == unwanted_blob] = background

    # Relabel wanted blobs:
    for new_label, wanted_blob in enumerate(wanted):
        filtered[image == wanted_blob] = foreground

    return filtered


def blob_filter(image, min_area=370, max_area=1200):

    blobs, labeled = blob_detection(image, min_area=370)

    num_regions = labeled.max()
    wanted_blobs = [True] * num_regions

    for num, blob in enumerate(blobs):

        if blob['area'] is not None:
            if blob['area'] < min_area or blob['area'] > max_area:
                wanted_blobs[num] = False

    filtered = _select_regions(labeled, wanted_blobs)

    return filtered
