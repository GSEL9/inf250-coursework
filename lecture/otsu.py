import sys

import numpy as np

from skimage import measure


def histogram(image):

    if np.ndim(image) > 2:
        _image = image.mean(axis=2)
    else:
        _image = np.copy(image)

    hist = np.zeros(256, dtype=int)
    for pixel in _image.ravel():
        hist[int(pixel)] += 1

    return hist


def otsu(image):

    hist = histogram(image)

    K = np.size(hist)
    mu0, mu1, N = _make_mean_tables(hist)

    sigma_max, q_max, n0  = 0, -1, 0
    for q in range(K-1):
        n0 += hist[q]
        n1 = N - n0

        if n0 > 0 and n1 > 0:

            sigma = (1 / (N ** 2)) * n0 * n1 * ((mu0[q] - mu1[q]) ** 2)
            if sigma > sigma_max:
                sigma_max = sigma
                q_max = q

    return q_max


def _make_mean_tables(hist):

    K = np.size(hist)
    mu0, mu1 = np.zeros(K, dtype=int), np.zeros(K, dtype=int)

    n0, s0 = 0, 0
    for q in range(K):
        n0 += hist[q]
        s0 += q * hist[q]

        if n0 > 0:
            mu0[q] = s0 / n0
        else:
            mu0[q] = -1

    N, mu1[K-1] = n0, 0

    n1, s1 = 0, 0
    for q in range(K-2, -1, -1):
        n1 += hist[q + 1]
        s1 += (q + 1) * hist[q + 1]

        if n1 > 0:
            mu1[q] = s1 / n1
        else:
            mu1[q] = -1

    return mu0, mu1, N


def threshold(image, th=None):

    if np.ndim(image) > 2:
        _image = image.mean(axis=2)
    else:
        _image = np.copy(image)

    binary = np.zeros(np.shape(_image), dtype=int)
    if th is None:
        th = otsu(_image)

    binary[_image > th] = 1

    return binary


def _select_single_blob(image, color=None):

    blob = np.copy(image)
    blob[image[:, :] != color] = 0

    return blob


def _remove_and_relabel_blobs(labeled, wanted_blobs):
    """This function removes unwanted blobs.
    """
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


def blob_detection(image, th=None, black_blobs=True, min_area=None, max_area=None):

    # Setup
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


if __name__ == '__main__':

    from skimage.io import imread
    import matplotlib.pyplot as plt

    img = imread('./bie_threshold.jpeg')

    filtered = blob_filter(img, min_area=500, max_area=1000)

    plt.imshow(filtered, cmap=plt.cm.gray)
    plt.show()
