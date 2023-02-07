from pathlib import Path
from glob import glob
import numpy as np
import os
import plyfile
import time
from multiprocessing import Pool, TimeoutError
from joblib import Parallel, delayed
import multiprocessing
import glob
import sys
sys.path.append('../')
from models.pointnet2_utils import farthest_point_sample, index_points
import torch

def SampleAndSave(src_file_path, target_file_path, use_fps, num_points):
    '''
    loads a point cloud from ply file, samples it and saves it to a target folder
    :param src_file_path: str, path to source ply file to convert
    :param target_file_path: str, path to location where the sampled ply file will be saved
    :param use_fps: bool, toggle if to use fps sampling or not
    :param num_points: int, number of points to samlpe
    :return:
    '''

    target_dir = os.path.dirname(os.path.abspath(target_file_path))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    plydata = plyfile.PlyData.read(src_file_path)
    d = np.asarray(plydata['vertex'].data)
    pc = np.column_stack([d[p.name] for p in plydata['vertex'].properties])
    if use_fps:
        pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).cuda()
        idxs = farthest_point_sample(pc_tensor[:, :, :3].contiguous(), num_points)
        sampled_points = index_points(pc_tensor, idxs.to(torch.int64)).squeeze().cpu().numpy()
    else:
        idxs = np.random.shuffle(np.arange(len(pc)))
        sampled_points = pc[idxs[:len(pc)]]

    pts = list(zip(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], sampled_points[:, 3],
                   sampled_points[:, 4], sampled_points[:, 5], sampled_points[:, 6], sampled_points[:, 7],
                   sampled_points[:, 8]))
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                  ('red', 'B'), ('green', 'B'), ('blue', 'B')])
    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text="").write(target_file_path)

def createSmallDataset(src_dataset, target_dataset, num_points, use_fps, parallelize=False):
    src_file_list = glob.glob(src_dataset + '*/*_recDir*/norm/Depth Long Throw/*.ply')
    target_file_list = [file_path.replace(src_dataset, target_dataset) for file_path in src_file_list]
    print(src_file_list)
    print(target_file_list)
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(SampleAndSave)(src_file_path, target_file_path, num_points, use_fps)
                                   for src_file_path, target_file_path in zip(src_file_list, target_file_list))
    else:
        for src_file_path, target_file_path in zip(src_file_list, target_file_list):
            print("Converting {}".format(src_file_path))
            start = time.process_time()
            SampleAndSave(src_file_path, target_file_path, num_points, use_fps)
            end = time.process_time()
            print("Done. conversion took {}".format(end - start))



if __name__ == '__main__':

    # linux:
    src_dataset = r'/data1/datasets/HoloLens/'
    target_dataset = r'/data1/datasets/ikeaego_small/'
    # src_dataset = r'/home/sitzikbs/Datasets/temp_Hololens/'
    # target_dataset = r'/home/sitzikbs/Datasets/temp_Hololens_smaller/'
    use_fps = True
    num_points = 4096
    parallelize = True
    createSmallDataset(src_dataset, target_dataset, use_fps, num_points, parallelize)