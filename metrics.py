"""Implements different image metrics."""
import torch
from skimage.metrics import structural_similarity


def ncc(img1, img2):
    """Takes two image and compute the negative normalized cross correlation.
       Lower the value, better the alignment.
    """
    # subtract mean and compute norm
    img1_mean = img1 - torch.mean(img1)
    img2_mean = img2 - torch.mean(img2)
    img1_norm = torch.norm(img1_mean)
    img2_norm = torch.norm(img2_mean)

    # Normalize the images to have unit norm
    img1_normalized = img1_mean / img1_norm
    img2_normalized = img2_mean / img2_norm

    # Compute the dot product
    ncc_value = torch.sum(img1_normalized * img2_normalized)

    return -ncc_value


def mse(img1, img2):
    """Takes two image and compute the mean squared error.
       Lower the value, better the alignment.
    """
    # Compute squared difference and compute mean
    diff_sq = (img1 - img2) ** 2
    mse_value = torch.mean(diff_sq)

    return mse_value


def ssim(img1, img2):
    """Takes two image and compute the negative structural similarity.

    This function is given to you, nothing to do here.

    Please refer to the classic paper by Wang et al. of Image quality
    assessment: from error visibility to structural similarity.
    """
    img1 = img1.numpy()
    img2 = img2.numpy()
    return -structural_similarity(img1, img2, data_range=img1.max() - img2.min())