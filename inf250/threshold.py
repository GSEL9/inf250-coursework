# -*- coding: utf-8 -*-
#
# threshold.py
#

"""A naive implementation of Otsu's thresholding algorithm."""


import sys

import numpy as np

from skimage import measure


def threshold(image, th=None):
    """Perform Utso thresholding for image binarization."""

    if np.ndim(image) > 2:
        _image = image.mean(axis=2)
    else:
        _image = np.copy(image)

    binary = np.zeros(np.shape(_image), dtype=int)
    if th is None:
        th = otsu(_image)

    binary[_image > th] = 1

    return binary


def otsu(image):
    """Otsu's thresholding algorithm."""

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


def histogram(image):
    """Compute image histogram."""

    if np.ndim(image) > 2:
        _image = image.mean(axis=2)
    else:
        _image = np.copy(image)

    hist = np.zeros(256, dtype=int)
    for pixel in _image.ravel():
        hist[int(pixel)] += 1

    return hist


def _make_mean_tables(hist):
    # Otsu thresholding auxillary function.

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
