"""Implements the alignment algorithm."""
import torch
import torchvision
import torch.nn.functional as F
from metrics import ncc, mse, ssim
from helpers import custom_shifts
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from net import STN


class AlignmentModel:
    def __init__(self, image_name, metric, lr, steps):
        # Image name
        self.image_name = image_name
        # Metric to use for alignment
        self.metric = metric

        self.lr = lr
        self.steps = steps

    def save(self, output_name):
        torchvision.utils.save_image(self.rgb, output_name)

    def align(self):
        """Aligns the image using the metric specified in the constructor.
           Experiment with the ordering of the alignment.

           Finally, outputs the rgb image in self.rgb.
        """
        self.img = self._load_image()

        b, g, r = self._crop_and_divide_image()

        r_shifted = self._align_pairs(g, r)

        b_shifted = self._align_pairs(g, b)

        self.rgb = torch.cat((r_shifted.unsqueeze(0), g.unsqueeze(0), b_shifted.unsqueeze(0)), dim=0)  # rgb

    def save(self, output_name):
        torchvision.utils.save_image(self.rgb, output_name)

    def _load_image(self):
        """Load the image from the image_name path,
           typecast it to float, and normalize it.

           Returns: torch.Tensor of shape (H, W)
        """
        ret = torchvision.io.read_image(self.image_name).float() / 255

        return ret.squeeze(0)

    def _crop_and_divide_image(self):
        """Crop the image boundary and divide the image into three parts, padded to the same size.

           Feel free to be creative about this.
           You can eyeball the boundary values, or write code to find approximate cut-offs.
           Hint: Plot out the average values per row / column and visualize it!

           Returns: B, G, R torch.Tensor of shape (roughly H//3, W)
        """

        # cut out white boundary
        row_mean = self.img.mean(axis=1)
        col_mean = self.img.mean(axis=0)

        # make values extreme to detect the boundary better
        row_boundary_indices = [i for i, x in enumerate(row_mean**3) if x < 0.1]
        col_boundary_indices = [i for i, x in enumerate(col_mean**3) if x < 0.1]

        top, bottom = row_boundary_indices[0], row_boundary_indices[-1]
        left, right = col_boundary_indices[0], col_boundary_indices[-1]

        # make height divisible by 3
        while (bottom - top) % 3 != 0:
            bottom -= 1

        height = (bottom - top) // 3

        b_channel = self.img[top: top + height, left:right]
        g_channel = self.img[top + height: top + 2 * height, left:right]
        r_channel = self.img[top + 2 * height: top + 3 * height, left:right]

        return b_channel, g_channel, r_channel

    def _align_pairs(self, img1, img2):
        """
        Aligns two images using the metric specified in the constructor.
        Returns: Tuple of (u, v) shifts that minimizes the metric.
        """

        h, w = img2.shape[0], img2.shape[1]
        loss = 0.0

        img2 = img2.view((1, 1, img2.shape[0], img2.shape[1]))

        model = STN()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for step in range(1, 1+self.steps):
            model.train()
            data, target = img2, img1
            optimizer.zero_grad()
            shifted_img, dx, dy = model(data)

            if self.metric == 'mse':
                loss = F.mse_loss(target, shifted_img)

            if self.metric == 'ncc':
                loss = ncc(target, shifted_img)

            loss.backward()
            optimizer.step()

        dx = -int(torch.round(dx * w / 2))
        dy = -int(torch.round(dy * h / 2))
        print(f"Step {step}, Loss: {loss.item()}, dy:{dy}, dx:{dx}")

        return shifted_img
