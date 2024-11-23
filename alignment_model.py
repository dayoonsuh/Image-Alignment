"""Implements the alignment algorithm."""
import torch
import torchvision
from metrics import ncc, mse, ssim
from helpers import custom_shifts
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class AlignmentModel:
    def __init__(self, image_name, metric, color='green', window = 20, padding='circular'):
        # Image name
        self.image_name = image_name
        # Metric to use for alignment
        self.metric = metric
        # Reference color for alignment
        self.color = color
        # Padding mode for custom_shifts
        self.padding = padding


    def save(self, output_name):
        torchvision.utils.save_image(self.rgb, output_name)

    def align(self):
        """Aligns the image using the metric specified in the constructor.
           Experiment with the ordering of the alignment.

           Finally, outputs the rgb image in self.rgb.
        """
        self.img = self._load_image()

        b, g, r = self._crop_and_divide_image()

        ####### align to green ###########
        if self.color == 'green':
            
            ru, rv = self._align_pairs(g, r, delta=15)
            r_shifted = custom_shifts(r, (ru, rv), dims=(0, 1), padding=self.padding)

            bu, bv = self._align_pairs(g, b, delta=15)
            b_shifted = custom_shifts(b, (bu, bv), dims=(0, 1), padding=self.padding)

            with open("green.txt", "a") as f:
               f.write(f"{self.image_name}, ({ru, rv}), ({bu, bv}), {self.metric}\n")

            self.rgb = torch.cat((r_shifted.unsqueeze(0), g.unsqueeze(0), b_shifted.unsqueeze(0)), dim=0) #rgb
            # self.rgb = torch.cat((r_shifted.unsqueeze(0), b_shifted.unsqueeze(0), g.unsqueeze(0)), dim=0) #rbg
            # self.rgb = torch.cat((g.unsqueeze(0), r_shifted.unsqueeze(0), b_shifted.unsqueeze(0)), dim=0) #grb
            # self.rgb = torch.cat((g.unsqueeze(0), b_shifted.unsqueeze(0), r_shifted.unsqueeze(0)), dim=0) #gbr
            # self.rgb = torch.cat((b_shifted.unsqueeze(0), g.unsqueeze(0), r_shifted.unsqueeze(0)), dim=0) #bgr
            # self.rgb = torch.cat((b_shifted.unsqueeze(0), r_shifted.unsqueeze(0), g.unsqueeze(0)), dim=0) #brg

        ######### align to blue ############
        elif self.color == 'blue':
            
            ru, rv = self._align_pairs(b, r, delta = 15)
            r_shifted = custom_shifts(r, (ru, rv), dims = (0,1), padding=self.padding)

            gu, gv = self._align_pairs(b, g, delta=15)
            g_shifted = custom_shifts(g, (gu, gv), dims=(0,1), padding=self.padding)

            with open("blue.txt", "a") as f:
              f.write(f"{self.image_name}, ({ru, rv}), ({gu, gv}), {self.metric}\n")

            self.rgb = torch.cat((r_shifted.unsqueeze(0), g_shifted.unsqueeze(0), b.unsqueeze(0)), dim=0)

        ####### align to red ############
        elif self.color == 'red':
            
            bu, bv = self._align_pairs(r, b, delta = 15)
            b_shifted = custom_shifts(b, (bu, bv), dims = (0,1), padding=self.padding)

            gu, gv = self._align_pairs(r, g, delta=15)
            g_shifted = custom_shifts(g, (gu, gv), dims=(0,1), padding=self.padding)
            self.rgb = torch.cat((r.unsqueeze(0), g_shifted.unsqueeze(0), b_shifted.unsqueeze(0)), dim=0)

            with open("red.txt", "a") as f:
                f.write(f"{self.image_name}, ({gu, gv}), ({bu, bv}), {self.metric}\n")
            
            self.rgb = torch.cat((r.unsqueeze(0), g_shifted.unsqueeze(0), b_shifted.unsqueeze(0)), dim=0)

    def save(self, output_name):
        print(self.rgb.shape)
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

    def _align_pairs(self, img1, img2, delta):
        """
        Aligns two images using the metric specified in the constructor.
        Returns: Tuple of (u, v) shifts that minimizes the metric.
        """
        align_idx = (0, 0)

        mid_h = img1.shape[0] // 2
        mid_w = img1.shape[1] // 2
        img1 = img1[mid_h - 100:mid_h + 100, mid_w - 100:mid_w + 100]

        best_metric = float('inf')

        for u in range(-delta, delta + 1):
            for v in range(-delta, delta + 1):
                # Apply shift to img2 then take the center part
                shifted_img = custom_shifts(img2, (u, v), dims=(0, 1))[mid_h - 100:mid_h + 100, mid_w - 100:mid_w + 100]

                # Calculate metric
                if self.metric == 'mse':
                    current_metric = mse(img1, shifted_img)
                    if current_metric < best_metric:
                        best_metric = current_metric
                        align_idx = (u, v)

                elif self.metric == 'ncc':
                    current_metric = ncc(img1, shifted_img)
                    if current_metric < best_metric:
                        best_metric = current_metric
                        align_idx = (u, v)

                elif self.metric == 'ssim':
                    current_metric = ssim(img1, shifted_img)
                    if current_metric < best_metric:
                        best_metric = current_metric
                        align_idx = (u, v)

        # print(self.image_name, best_metric, align_idx, self.metric)
        return align_idx