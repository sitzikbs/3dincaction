import numpy as np
from torch.utils.data import Dataset
import torch
import sys
sys.path.append('../')
import visualization as vis

class NoiseGenerator(Dataset):
    def __init__(self, n_points, radius=0.5, n_samples=1, sigma=0.3):
        # # Generate random spherical coordinates

        self.radius = radius
        self.n_points = n_points
        self.points = []
        for i in range(n_samples):
            self.points.append(self.get_noisy_points(sigma))


    def get_noisy_points(self, sigma):
        #TODO add local distortion (use kdtree and fixed motion vec)
        return np.clip(np.random.randn(self.n_points, 3)*sigma, -1, 1).astype(np.float32)

    def __len__(self):
        return len(self.points)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return {'points': self.points[idx]}


if __name__ == "__main__":
    sphere_dataset = NoiseGenerator(512, 0.5, 2)
    dataloader = torch.utils.data.DataLoader(sphere_dataset, batch_size=1, num_workers=0,
                                                   pin_memory=True, shuffle=True, drop_last=True)

    for batch_idx, data in enumerate(dataloader):
        points = data
        vis.plot_pc_pv(points.detach().cpu().numpy())
