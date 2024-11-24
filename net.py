import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        self.shifts = nn.Parameter(torch.zeros(2))

    def forward(self, img):

        theta = torch.zeros(1, 2, 3, device=img.device)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, 0, 2] = self.shifts[0]  # Set the translation in x (tx)
        theta[:, 1, 2] = self.shifts[1]  # Set the translation in y (ty)

        grid = F.affine_grid(theta, img.size(), align_corners=False)
        shifted_img = F.grid_sample(img, grid, mode='bilinear').squeeze()
        return shifted_img, self.shifts[0], self.shifts[1]
