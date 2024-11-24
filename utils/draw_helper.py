import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def draw_all_circles(
    image: np.ndarray,
    # cx: list[int] | np.ndarray,
    # cy: list[int] | np.ndarray,
    # rad: list[int] | np.ndarray,
    blobs,
    filename: str,
    color: str = 'r'
) -> None:
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    filename: output filename
    color: circle color, default to 'r' (red)
    """
    if image.shape[0] == 1:
        image = np.concatenate([image, image, image], 0)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image.transpose(1, 2, 0), cmap='gray')

    blobs = [(b[0], b[1], b[2] * np.sqrt(2)) for b in blobs]

    for x, y, r in blobs:
        circ = Circle((y, x), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(blobs))
    plt.savefig(filename)
