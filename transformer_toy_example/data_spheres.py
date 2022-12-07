import numpy as np
from torch.utils.data import Dataset
import pyvista as pv
import torch
import sys
sys.path.append('../')
import visualization as vis

class SphereGenerator(Dataset):
    def __init__(self, n_points, radius=0.5, n_samples=1):
        # # Generate random spherical coordinates
        # u = np.random.uniform(0, 2 * np.pi, 500)
        # v = np.random.uniform(0, np.pi, 500)
        #
        # # Convert spherical coordinates to Cartesian coordinates
        # x = radius * np.cos(u) * np.sin(v)
        # y = radius * np.sin(u) * np.sin(v)
        # z = radius * np.cos(v)
        #
        #
        # points = np.array([x, y, z]).T.astype(np.float32)
        self.radius = radius
        self.points = []
        for i in range(n_samples):
            # np.random.shuffle(points)
            # self.points.append(points[:n_points])
            self.points.append(self.sample_spherical(n_points).T.astype(np.float32))

    def sample_spherical(self, npoints, ndim=3):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return self.radius * vec
    def __len__(self):
        return len(self.points)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.points[idx]

if __name__ == "__main__":
    sphere_dataset = SphereGenerator(128, 0.5, 2)
    dataloader = torch.utils.data.DataLoader(sphere_dataset, batch_size=1, num_workers=0,
                                                   pin_memory=True, shuffle=True, drop_last=True)

    for batch_idx, data in enumerate(dataloader):
        points = data
        vis.plot_pc_ov(points.detach().cpu().numpy())
