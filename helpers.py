"""Implements helper functions."""
import torch


def custom_shifts(input, shifts, dims=None, padding='circular'):
    """Shifts the input tensor by the specified shifts along
       the specified dimensions. Supports circular and zero padding.

       Input: Tensor
       Returns: Shifted Tensor along the specified dimension
         padded following the padding scheme.
    """
    ret = None
    ## Your CODE HERE ##
    if padding == 'circular':
        shifted = torch.roll(input, shifts, dims=dims)

    elif padding == 'zero':

        shifted = torch.roll(input, shifts, dims=dims)

        u, v = shifts[0], shifts[1]

        if u > 0:
            shifted[:u] = 0
        elif u < 0:
            shifted[u:] = 0

        if v > 0:
            shifted[:, :v] = 0
        elif v < 0:
            shifted[:, v:] = 0

    return shifted