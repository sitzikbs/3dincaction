from pathlib import Path
from glob import glob
import numpy as np
import os
import csv
from shutil import copyfile
# from project_hand_eye_to_pv import *
import cv2
import numpy as np
# import cv2
import random
import json
from PIL import Image, ImageDraw, ImageFont
import struct
# import open3d as o3d
import numexpr as ne

import torchvision.transforms as transforms

# from hand_defs import HandJointIndex


def save_ply(output_path, points, rgb=None, cam2world_transform=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals()
    if cam2world_transform is not None:
        # Camera center
        camera_center = (cam2world_transform) @ np.array([0, 0, 0, 1])
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_center[:3])

    o3d.io.write_point_cloud(output_path, pcd)


def imread_pgm(pgmdir):
    depth_image = cv2.imread(pgmdir, -1)

    # print(depth_image[100])

    return depth_image

def fps_np(points, npoint):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    # xyz = points[0] #use the first frame for sampling
    xyz = points

    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    idxs = np.array(farthest)[None]

    for i in range(npoint-1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.array(np.argmax(distance, -1))[None]
        idxs = np.concatenate([idxs, farthest])


    return points[idxs]


def stochastic_vec_sample_numeric(vec, num_samples = 4096):
    np.random.shuffle(vec)
    return vec[:num_samples]

def stochastic_vec_sample_proportional(vec, stochastic_sample_ratio_inv = 1):
    target_length = int(len(vec)/stochastic_sample_ratio_inv)
    return stochastic_vec_sample_numeric(vec, target_length)


def fps_ne(points, npoint, stochastic_sample:bool, stochastic_sample_ratio_inv = 1):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    # xyz = points[0] #use the first frame for sampling
    xyz = points
    if stochastic_sample:
        assert len(points)/stochastic_sample_ratio_inv >= npoint
        xyz = stochastic_vec_sample_proportional(xyz, stochastic_sample_ratio_inv)
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


    return points[idxs]


def fps(n_points, points):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    # xyz = points[0] #use the first frame for sampling
    xyz = points
    print(xyz.shape)
    N, C = xyz.shape
    centroids = np.zeros(n_points)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    idxs = np.array(farthest)[None]

    for i in range(n_points-1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.array(np.argmax(distance, -1))[None]
        idxs = np.concatenate([idxs, farthest])

    return points[idxs]


def read16BitPGM(pgm_dir):
    """Return a raster of integers from a PGM as a list of lists."""
    with open(pgm_dir, 'rb') as pgmf:
        header = pgmf.readline()
        assert header[:2] == b'P5'
        size = pgmf.readline()
        (width, height) = [int(i) for i in size.split()]
        depth = int(pgmf.readline())

        assert depth <= 65535

        raster = []
        for y in range(height):
           row = []
           for y in range(width):
               cv2.imread('a.pgm', -1)
               ###############
               # low_bits =int.from_bytes(pgmf.read(2), byteorder='big')
               # row.append(low_bits)
               ###############
               # low_bits = ord(pgmf.read(1))
               # row.append(low_bits+255*ord(pgmf.read(1)))
               ###############
           raster.append(row)
        # print(raster)
        return raster


def addTextToImg(img_path, txt="null"):
    # Call draw Method to add 2D graphics in an image
    image = Image.open(img_path)
    I1 = ImageDraw.Draw(image)
    # font = ImageFont.load_default()
    # Add Text to an image
    font = ImageFont.truetype(font="arial.ttf", size = 42)

    I1.text((28, 36), txt, fill=(255, 0, 0), font=font)

    # Define a transform to convert PIL
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)
    # Display edited image
    return img_tensor


def getNumFrames(_dir_):
    pv_dir = os.path.join(_dir_, "pv")
    assert os.path.exists(pv_dir)
    return len(os.listdir(pv_dir))

def searchForAnnotationJson(_dir_):
    for sub_dir in os.listdir(_dir_):
        # print("lets look at {}".format(sub_dir))
        if "annotations.json" in sub_dir[-16:]:
            # print("found {}".format(sub_dir))
            return os.path.join(_dir_, sub_dir)

    return None


def translateSecToFrame(start_sec, end_sec,  fps_error_correction_mapping):
    last_frame_vid = list(fps_error_correction_mapping.keys())[-1]
    if end_sec > float(last_frame_vid)/15:
        end_sec = float(last_frame_vid)/15
    if  start_sec > float(last_frame_vid)/15:
        print(start_sec, end_sec, float(last_frame_vid) / 15)
        return
    seg_start = fps_error_correction_mapping[str(int(np.round(15 * start_sec)))]
    seg_end = fps_error_correction_mapping[str(int(np.round(15 * end_sec)))]

    return [seg_start,seg_end ]


def applyMappingToAnnotations(action_annotation_list, id_to_name, fps_error_correction_mapping):
    action_labels = []
    if fps_error_correction_mapping:
        print("starting to decode using json!!")
    for encoded_annotation in action_annotation_list:
        if fps_error_correction_mapping:

            segment = translateSecToFrame(encoded_annotation["start"], encoded_annotation["end"], fps_error_correction_mapping)
            if not segment:
                print("=============", encoded_annotation["start"], encoded_annotation["end"], "=============")
                continue
            # print()
            # segment = [int(fps_error_correction_mapping[encoded_annotation["start"]]), int(fps_error_correction_mapping[encoded_annotation["end"]])]
            # print (segment)
        else:
            segment = [int(np.round(15 * encoded_annotation["start"])), int(np.round(15 * encoded_annotation["end"]))]
        label = id_to_name[encoded_annotation["action"]]
        decoded_label = {"segment": segment, "label": label}
        action_labels.append(decoded_label)
    return action_labels

def removeBackslashT(action_labels):
    for annotation_num in range(len(action_labels)):
        action_labels[annotation_num]["label"] = action_labels[annotation_num]["label"].replace("\t", " ")
        print(action_labels[annotation_num]["label"])


def getIdToNameMapping(action_label_data: list):
    id_to_name = {}
    for single_id_map in action_label_data:
        # print(single_id_map)
        _id_ = single_id_map["id"]
        name = single_id_map["name"]
        id_to_name[_id_] = name
    print(id_to_name)
    return id_to_name

def saveVideoClip(clip_name, clip_frames):
    video = cv2.VideoWriter(clip_name, 0, 15, (960, 540))
    for frame in clip_frames:
        transposed_frame = np.transpose(frame, (1, 2, 0))
        video.write(cv2.cvtColor(np.array(transposed_frame), cv2.COLOR_RGB2BGR))
    print("saved video: ",clip_name )
def decodeJsonAnnotations(current_json_annotation: dict, fps_error_correction_mapping):
    print(f"current_json_annotation: {current_json_annotation.keys()}")
    id_to_name = getIdToNameMapping(current_json_annotation["config"]["actionLabelData"])
    action_labels = applyMappingToAnnotations(current_json_annotation["annotation"]["actionAnnotationList"], id_to_name,
                                              fps_error_correction_mapping=fps_error_correction_mapping)
    removeBackslashT(action_labels)
    print(action_labels)
    return action_labels


def getAllJsonsInDirList(dir_list, merged_json, subset):
    assert subset == "training" or subset == "testing"
    for _dir_ in dir_list:
        json_file = searchForAnnotationJson(_dir_)
        if json_file:
            print("found json file {}".format(json_file))
            with open(json_file) as json_file_obj:
                current_json = json.load(json_file_obj)
                # add error check if json file already exists in database
                fps_error_correction_mapping = None
                fps_error_correction_json_path = os.path.join(_dir_, "frame_rate_translation.json")
                if os.path.exists(fps_error_correction_json_path):
                    with open(fps_error_correction_json_path) as fps_error_correction_obj:
                        fps_error_correction_mapping = json.load(fps_error_correction_obj)

                print(f"@@@@  directory {_dir_} has a translation json @@@@")
                merged_json["database"][_dir_] = {"subset": subset,
                                                      "annotation": decodeJsonAnnotations(current_json,
                                                                                          fps_error_correction_mapping)}
        else:
            print(f"path {_dir_} does not have json yet")


def getAllJsonAnnotations(dataset_dir, merged_json=None):
    if merged_json is None or merged_json == {}:
        merged_json = {"version": "2.0.0", "database": {}}
    print(merged_json)

    # use this code for one test directory:
    # getAllJsonsInDirList([dataset_dir], merged_json, "testing")
    # use this code for real database:
    test_dir_list_file = os.path.join(dataset_dir, "indexing_files", "all_test_dir_list.txt")
    train_dir_list_file = os.path.join(dataset_dir, "indexing_files", "all_train_dir_list.txt")
    assert os.path.exists(test_dir_list_file) and os.path.exists(train_dir_list_file)
    test_dir_list = getListFromFile(test_dir_list_file)
    train_dir_list = getListFromFile(train_dir_list_file)
    getAllJsonsInDirList(test_dir_list, merged_json, "testing")
    getAllJsonsInDirList(train_dir_list, merged_json, "training")
    print(merged_json)

    with open(os.path.join(dataset_dir, "indexing_files", "db_gt_annotations.json"), "w") as new_json_file_obj:
        print(merged_json)
        merged_json = json.dumps(merged_json)
        new_json_file_obj.write(merged_json)
        return


def writeListToFile(filename, line_list):
    with open(filename, "w") as f:
        for line in line_list:
            # FIXME: maybe remove this \n
            f.writelines(line + "\n")
    return


def getListFromFile(filename):
    """
    retrieve a list of lines from a .txt file
    :param :
    :return: list of atomic actions
    """
    with open(filename) as f:
        line_list = f.read().splitlines()
    # line_list.sort()
    return line_list


def getSepereateFurnitureRecDirLists(dataset_dir):
    furniture_sep_rec_dir_list = [
        (
        furniture_name, os.path.join(dataset_dir, "indexing_files", "{}_recording_dir_list.txt".format(furniture_name)))
        for furniture_name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, furniture_name)) and furniture_name != "indexing_files"]
    return furniture_sep_rec_dir_list


def createTrainTestFiles(dataset_dir, train_ratio=0.7):
    furniture_sep_rec_dir_list = getSepereateFurnitureRecDirLists(dataset_dir)
    print(furniture_sep_rec_dir_list)
    all_train_recordings = []
    all_test_recordings = []

    for furniture_name, idx_file_path in furniture_sep_rec_dir_list:
        # check that all furniture indexing files are created.
        print("rec path:", idx_file_path)
        assert (os.path.exists(idx_file_path))
        recording_dir_list = getListFromFile(idx_file_path)
        random.shuffle(recording_dir_list)
        num_furniture_recordings = len(recording_dir_list)
        num_train_recordings = int(train_ratio * num_furniture_recordings)
        num_test_recordings = num_furniture_recordings - num_train_recordings
        train_rec_list = recording_dir_list[:num_train_recordings]
        test_rec_list = recording_dir_list[num_train_recordings:]
        all_train_recordings += train_rec_list
        all_test_recordings += test_rec_list
        writeListToFile(os.path.join(dataset_dir, "indexing_files", "{}_train_dir_list.txt".format(furniture_name)),
                        train_rec_list)
        writeListToFile(os.path.join(dataset_dir, "indexing_files", "{}_test_dir_list.txt".format(furniture_name)),
                        test_rec_list)

    writeListToFile(os.path.join(dataset_dir, "indexing_files", "all_train_dir_list.txt"),
                    all_train_recordings)
    writeListToFile(os.path.join(dataset_dir, "indexing_files", "all_test_dir_list.txt"),
                    all_test_recordings)


def aux_createAllRecordingDirList(dataset_dir, target_file):
    if "_recDir" in dataset_dir[-8:]:
        with open(target_file, "a") as all_rec_idx_file:
            all_rec_idx_file.write(dataset_dir + "\n")
        return

    for sub_dir in glob(rf"{dataset_dir}\*\\"):
        print(f"calling aux_createAllRecordingDirList for path: {sub_dir}, continuing search for recording dir")
        aux_createAllRecordingDirList(sub_dir, target_file)


def createAllRecordingDirList(dataset_dir, target_file):
    if os.path.exists(target_file):
        os.remove(target_file)

    print(f"calling aux_createAllRecordingDirList for top path: {dataset_dir}, continuing search for recording dir")
    aux_createAllRecordingDirList(dataset_dir, target_file)


def getNumRecordings(w_path):
    """
    recursively iterate over directory to get number of sub dirs containing recordings
    """
    return len([f for f in Path(w_path).iterdir() if f.is_dir() and "_recDir" in str(w_path)[-8:]])


def removeOriginalPvImages(w_path):
    orig_pv_path = w_path / "pv"
    assert (w_path / "norm" / "pv").exists()
    orig_pv_images = [f.path for f in os.scandir(orig_pv_path) if os.path.splitext(f)[-1] == '.png']
    for orig_pv_img in orig_pv_images:
        os.remove(orig_pv_img)


def load_head_hand_eye_data(csv_path):
    joint_count = HandJointIndex.Count.value

    data = np.loadtxt(csv_path, delimiter=',')

    n_frames = len(data)
    timestamps = np.zeros(n_frames)
    head_transs = np.zeros((n_frames, 3))

    left_hand_transs = np.zeros((n_frames, joint_count, 3))
    left_hand_transs_available = np.ones(n_frames, dtype=bool)
    right_hand_transs = np.zeros((n_frames, joint_count, 3))
    right_hand_transs_available = np.ones(n_frames, dtype=bool)

    # origin (vector, homog) + direction (vector, homog) + distance (scalar)
    gaze_data = np.zeros((n_frames, 9))
    gaze_available = np.ones(n_frames, dtype=bool)

    for i_frame, frame in enumerate(data):
        timestamps[i_frame] = frame[0]
        # head
        head_transs[i_frame, :] = np.array(frame[1:17].reshape((4, 4)))[:3, 3]
        # left hand
        left_hand_transs_available[i_frame] = (frame[17] == 1)
        left_start_id = 18
        for i_j in range(joint_count):
            j_start_id = left_start_id + 16 * i_j
            j_trans = np.array(frame[j_start_id:j_start_id + 16].reshape((4, 4))).T[:3, 3]
            left_hand_transs[i_frame, i_j, :] = j_trans

        # right hand
        right_hand_transs_available[i_frame] = (
                frame[left_start_id + joint_count * 4 * 4] == 1)
        right_start_id = left_start_id + joint_count * 4 * 4 + 1
        for i_j in range(joint_count):
            j_start_id = right_start_id + 16 * i_j
            j_trans = np.array(frame[j_start_id:j_start_id + 16].reshape((4, 4))).T[:3, 3]
            right_hand_transs[i_frame, i_j, :] = j_trans

        assert (j_start_id + 16 == 851)
        gaze_available[i_frame] = (frame[851] == 1)
        gaze_data[i_frame, :4] = frame[852:856]
        gaze_data[i_frame, 4:8] = frame[856:860]
        gaze_data[i_frame, 8] = frame[860]

    return (timestamps, head_transs, left_hand_transs, left_hand_transs_available,
            right_hand_transs, right_hand_transs_available, gaze_data, gaze_available)


def removeOriginalPlyFiles(w_path, sensor_name="Depth Long Throw"):
    orig_depth_path = w_path / "{}".format(sensor_name)

    assert (w_path / "norm" / "{}".format(sensor_name)).exists()
    orig_depth_ply_files = [f.path for f in os.scandir(orig_depth_path) if os.path.splitext(f)[-1] == '.ply']
    for orig_depth_ply in orig_depth_ply_files:
        os.remove(orig_depth_ply)
    # print(f"going to remove the following files:{orig_depth_ply_files}")


def build_normalized_data_dir(w_path, sensor_name="Depth Long Throw"):
    norm_dir = Path(w_path / "norm")
    norm_pv_dir = Path(w_path / "norm" / "pv")
    norm_depth_dir = Path(w_path / "norm" / sensor_name)

    if not norm_dir.exists():
        os.mkdir(norm_dir)

    if not norm_pv_dir.exists():
        os.mkdir(norm_pv_dir)

    if not norm_depth_dir.exists():
        os.mkdir(norm_depth_dir)


def copyRenamePvImage(w_path, pv_timestamp, frame_number):
    original_pv_path = Path(w_path / "pv" / f"{pv_timestamp}.png")
    norm_pv_path = Path(w_path / "norm" / "pv" / f"{frame_number}.png")
    if norm_pv_path.exists():
        # print(f"{frame_number}.png pv file exists")
        return
    copyfile(original_pv_path, norm_pv_path)


def copyRenameDepthImage(w_path, depth_timestamp, frame_number, sensor_name="Depth Long Throw"):
    for file_format in ["pgm", "ply"]:
        original_depth_path = Path(w_path / sensor_name / f"{depth_timestamp}.{file_format}")
        norm_depth_path = Path(w_path / "norm" / sensor_name / f"{frame_number}.{file_format}")
        if norm_depth_path.exists():
            # print(f"{frame_number}.{file_format} depth file exists")
            return
        copyfile(original_depth_path, norm_depth_path)


def copyRenameHandEyeImage(w_path, pv_to_depth_hand_eye_mapping):
    hand_eye_path = Path(w_path / "head_hand_eye.csv")
    norm_hand_eye_path = Path(w_path / "norm" / "head_hand_eye.csv")

    with open(hand_eye_path, 'r') as f, open(norm_hand_eye_path, 'w') as norm_f:
        csvreader = csv.reader(f)
        norm_csvreader = csv.writer(norm_f)
        hand_eye_dict = {}
        for row in csvreader:
            hand_eye_dict[int(row[0])] = row
        for frame_number, pv_timestamp in enumerate(pv_to_depth_hand_eye_mapping.keys()):
            depth_ts, hand_eye_ts = pv_to_depth_hand_eye_mapping[pv_timestamp]
            norm_csvreader.writerow([frame_number] + hand_eye_dict[hand_eye_ts][1:])


def matchTimestamp(target, all_timestamps):
    return all_timestamps[np.argmin([abs(x - target) for x in all_timestamps])]


def extract_timestamp(path, depth_path_suffix):
    path = path.name.replace(depth_path_suffix, '')
    return int(path.split('.')[0])


def getHandEyeTimestamps(w_path):
    hand_eye_path = Path(w_path / "head_hand_eye.csv")
    # print(f"opening pv file {hand_eye_path}")
    hand_eye_timestamps = []
    with open(hand_eye_path, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            hand_eye_timestamps.append(int(row[0]))
    return hand_eye_timestamps


def getDepthTimestamps(w_path, sensor_name, depth_path_suffix):
    depth_path = Path(w_path / sensor_name)
    depth_paths = sorted(depth_path.glob('*[0-9]{}.pgm'.format(depth_path_suffix)))
    n_depth_frames = len(depth_paths)
    depth_timestamps = np.zeros(n_depth_frames, dtype=np.longlong)
    for i_path, path in enumerate(depth_paths):
        depth_timestamp = extract_timestamp(path, depth_path_suffix)
        depth_timestamps[i_path] = depth_timestamp
    return depth_timestamps


def getPvTimestamps(w_path):
    pv_csv_path = list(w_path.glob('*pv.txt'))[0]
    # print(f"opening pv file {pv_csv_path}")
    with open(pv_csv_path) as f:
        lines = f.readlines()
    if len(lines) <= 0:
        print(f"fount empty pv header file in: {pv_csv_path}")
        return
    n_frames = len(lines) - 1
    frame_timestamps = []
    for i_frame, frame in enumerate(lines[1:]):
        if 'nan' in frame:
            print(frame, "invalid pv header data")
            continue
        if len(frame) > 3:
            frame = frame.split(',')
            frame_timestamps.append(int(frame[0]))
    return frame_timestamps


# def processHandEyeData(folder):
#     print("processing hand eye")
#     norm_proc_eye_path = Path(folder / "norm" / "norm_proc_eye_data.csv")
#     norm_proc_hand_path = Path(folder / "norm" / "norm_proc_hand_data.csv")
#     with open(norm_proc_eye_path, 'w') as norm_proc_eye_f, open(norm_proc_hand_path, 'w') as norm_proc_hand_f:
#         eye_csvwriter = csv.writer(norm_proc_eye_f)
#         hand_csvwriter = csv.writer(norm_proc_hand_f)
#
#         head_hat_stream_path = list(folder.glob('*_eye.csv'))[0]
#         print(f"opening eye data file {head_hat_stream_path}")
#
#         pv_info_path = list(folder.glob('*pv.txt'))[0]
#         pv_paths = sorted(list((folder / 'PV').glob('*png')))
#         if len(pv_paths) == 0:
#             print(f"this is an empty recording: {folder}")
#             return -1
#
#         # load head, hand, eye data
#         (timestamps, _,
#          left_hand_transs, left_hand_transs_available,
#          right_hand_transs, right_hand_transs_available,
#          gaze_data, gaze_available) = load_head_hand_eye_data(head_hat_stream_path)
#
#         eye_str = " and eye gaze" if np.any(gaze_available) else ""
#
#         # load pv info
#         (frame_timestamps, focal_lengths, pv2world_transforms,
#          ox, oy, width, height) = load_pv_data(pv_info_path)
#         principal_point = np.array([ox, oy])
#         n_frames = len(pv_paths)
#         output_folder = folder / 'eye_hands'
#         output_folder.mkdir(exist_ok=True)
#         for pv_id in range(min(n_frames, len(focal_lengths))):
#             eye_data_row = [pv_id]
#             hand_data_row = [pv_id]
#             pv_path = pv_paths[pv_id]
#             sample_timestamp = int(str(pv_path.name).replace('.png', ''))
#
#             hand_ts = match_timestamp(sample_timestamp, timestamps)
#             # print('Frame-hand delta: {:.3f}ms'.format((sample_timestamp - timestamps[hand_ts]) * 1e-4))
#
#             # img = cv2.imread(str(pv_path))
#             # pinhole
#             K = np.array([[focal_lengths[pv_id][0], 0, principal_point[0]],
#                           [0, focal_lengths[pv_id][1], principal_point[1]],
#                           [0, 0, 1]])
#             try:
#                 Rt = np.linalg.inv(pv2world_transforms[pv_id])
#
#             except np.linalg.LinAlgError:
#                 print('No pv2world transform')
#                 continue
#
#             rvec, _ = cv2.Rodrigues(Rt[:3, :3])
#             tvec = Rt[:3, 3]
#
#             # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
#             hands = [(left_hand_transs, left_hand_transs_available),
#                      (right_hand_transs, right_hand_transs_available)]
#             for hand_id, hand in enumerate(hands):
#                 transs, avail = hand
#                 if avail[hand_ts]:
#                     for joint_num, joint in enumerate(transs[hand_ts]):
#                         hand_tr = joint.reshape((1, 3))
#                         hand_data_row += list(hand_tr.reshape(3))  # adding 3d joint point to row.
#                         # print(data_row)
#                         xy, _ = cv2.projectPoints(hand_tr, rvec, tvec, K, None)
#                         ixy = (int(xy[0][0][0]), int(xy[0][0][1]))
#                         ixy = (width - ixy[0], ixy[1])
#                         # print(ixy)
#                         hand_data_row += [ixy[0], ixy[1]]  # adding joint x,y projection to row.
#                 else:
#                     hand_data_row += list(np.zeros(HandJointIndex.Count.value * 5))
#
#             if gaze_available[hand_ts]:
#                 point = get_eye_gaze_point(gaze_data[hand_ts])
#                 eye_data_row += list(gaze_data[hand_ts][:3])  # add origin_homog
#                 eye_data_row += list(point)  # adding 3d pupil point to row.
#                 xy, _ = cv2.projectPoints(point.reshape((1, 3)), rvec, tvec, K, None)
#                 ixy = (int(xy[0][0][0]), int(xy[0][0][1]))
#                 ixy = (width - ixy[0], ixy[1])
#                 eye_data_row += [ixy[0], ixy[1]]  # adding pupil x,y projection to row.
#                 if pv_id % 500 == 0:
#                     print(width, ixy[0], ixy[1])
#                     print(f"saving hand_eye processed data number {pv_id}")
#             else:
#                 eye_data_row += [0, 0, 0, 0, 0, 0, 0, 0]
#
#             eye_csvwriter.writerow(eye_data_row)
#             hand_csvwriter.writerow(hand_data_row)
#             # cv2.imwrite(str(output_folder / 'hands') + 'proj{}.png'.format(str(sample_timestamp).zfill(4)), img)
#             # cv2.imwrite(f"{output_folder}/{str(sample_timestamp)}.png", img)