from glob import glob
import os
import plyfile
import time
from joblib import Parallel, delayed
import glob
import numpy as np
import numexpr as ne
from ikeaego_utils import createAllRecordingDirList, createTrainTestFiles, getListFromFile, writeListToFile, getAllJsonAnnotations

def fps_ne(points, npoint):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    xyz = points[:, :3]
    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    idxs = np.array(farthest)[None]

    for i in range(npoint-1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = ne.evaluate('sum((xyz - centroid) ** 2, 1)')
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.array(np.argmax(distance, -1))[None]
        idxs = np.concatenate([idxs, farthest])


    return points[idxs, :]

def SampleAndSave(src_file_path, target_file_path, use_fps, num_points):
    '''
    loads a point cloud from ply file, samples it and saves it to a target folder
    :param src_file_path: str, path to source ply file to convert
    :param target_file_path: str, path to location where the sampled ply file will be saved
    :param use_fps: bool, toggle if to use fps sampling or not
    :param num_points: int, number of points to samlpe
    :return:
    '''
    print("Converting {}".format(src_file_path))
    start = time.process_time()
    target_dir = os.path.dirname(os.path.abspath(target_file_path))
    if not os.path.exists(target_file_path):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        plydata = plyfile.PlyData.read(src_file_path)
        d = np.asarray(plydata['vertex'].data)
        pc = np.column_stack([d[p.name] for p in plydata['vertex'].properties])
        if use_fps:
            sampled_points = fps_ne(pc, num_points)
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
        end = time.process_time()
        print("Done. conversion took {}".format(end - start))


def createSmallDataset(src_dataset, target_dataset, num_points, use_fps, parallelize=False, workers=32):
    assert os.path.exists(src_dataset)

    src_file_list = glob.glob(src_dataset + '*/*_recDir*/norm/Depth Long Throw/*.ply')
    target_file_list = [file_path.replace(src_dataset, target_dataset) for file_path in src_file_list]
    print(src_file_list)
    print(target_file_list)
    if parallelize:
        num_cores = workers
        Parallel(n_jobs=num_cores)(delayed(SampleAndSave)(src_file_path, target_file_path, num_points, use_fps)
                                   for src_file_path, target_file_path in zip(src_file_list, target_file_list))
    else:
        for src_file_path, target_file_path in zip(src_file_list, target_file_list):
            SampleAndSave(src_file_path, target_file_path, num_points, use_fps)


def createAnnotationJson(dataset_dir, out_dir):
    getAllJsonAnnotations(dataset_dir=dataset_dir, out_dir=out_dir, merged_json={})


def copyActionList(dataset_dir, out_dir, action_list_txt_file=""):

    if action_list_txt_file == "":
        action_list_txt_file = os.path.join(dataset_dir, "action_list.txt")
    action_list = getListFromFile(action_list_txt_file)
    print(action_list)
    writeListToFile(filename=os.path.join(out_dir, "indexing_files", "action_list.txt"), line_list=action_list)


def createSeperateFurnitureRecLists(dataset_dir, out_dir):
    indexing_files_path = os.path.join(out_dir, "indexing_files")

    [createAllRecordingDirList(dataset_dir=os.path.join(dataset_dir, furniture_name),
                               target_file=os.path.join(indexing_files_path, "{}_recording_dir_list.txt".format(furniture_name)),
                               original_dataset_path=dataset_dir)
     for furniture_name in os.listdir(dataset_dir)
     if os.path.isdir(os.path.join(dataset_dir, furniture_name)) and not furniture_name == "indexing_files"]


def createAllIndexingFiles(dataset_dir, target_dataset):
    # w_path = Path(dataset_dir)
    indexing_files_path = os.path.join(target_dataset, "indexing_files")

    if not os.path.exists(indexing_files_path): os.mkdir(indexing_files_path)

    recording_dir_list_path = os.path.join(indexing_files_path, "all_recording_dir_list.txt")
    createAllRecordingDirList(dataset_dir=dataset_dir, target_file=recording_dir_list_path,
                              original_dataset_path=dataset_dir)
    createSeperateFurnitureRecLists(dataset_dir, target_dataset)
    createTrainTestFiles(dataset_dir=dataset_dir, out_dir=target_dataset)
    copyActionList(dataset_dir=dataset_dir, out_dir=target_dataset)
    createAnnotationJson(dataset_dir=dataset_dir, out_dir=target_dataset)


if __name__ == '__main__':
    src_dataset = r'/data1/datasets/Hololens/'
    target_dataset = r'/data1/datasets/ikeaego_small/'
    # src_dataset = r'/home/sitzikbs/Datasets/ikeaego_small/'
    # target_dataset = r'/home/sitzikbs/Datasets/ikeaego_small/'
    use_fps = True
    num_points = 4096
    parallelize = True
    # createSmallDataset(src_dataset, target_dataset, use_fps, num_points, parallelize)
    createAllIndexingFiles(src_dataset, target_dataset)
    # run conver_json_actions.py