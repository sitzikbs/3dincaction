import torch
import numpy as np
import json
import os
import plyfile
import torchvision
from IKEAActionDataset import IKEAActionDataset
import cv2


class MultiModalDataset(IKEAActionDataset):
    def __init__(self, dataset_path, db_filename='ikea_annotation_db_full', action_list_filename='atomic_action_list.txt',
                 action_object_relation_filename='action_object_relation_list.txt', train_filename='train_cross_env.txt',
                 test_filename='test_cross_env.txt', transform=None, set='test', camera='dev3', frame_skip=1,
                 frames_per_clip=64, pose_path='predictions/pose2d/keypoint_rcnn',
                 modalities=['rgb', 'depth', 'pose', 'pc'], n_points=2048):

        super().__init__(dataset_path=dataset_path, db_filename=db_filename, action_list_filename=action_list_filename,
                 action_object_relation_filename=action_object_relation_filename, train_filename=train_filename,
                         test_filename=test_filename)
        self.pose_path = pose_path
        self.transform = transform
        self.set = set
        self.camera = camera
        self.frame_skip = frame_skip
        self.frames_per_clip = frames_per_clip
        self.modalities = modalities
        self.n_points = n_points

        if self.set == 'train':
            self.video_list = self.trainset_video_list
        elif self.set == 'test':
            self.video_list = self.testset_video_list
        else:
            raise ValueError("Invalid set name")

        self.video_set = self.get_video_frame_labels()
        self.clip_set, self.clip_label_count = self.get_clips()

    def get_video_frame_labels(self):
        # Extract the label data from the database
        # outputs a dataset structure of (video_path, multi-label per-frame, number of frames in the video)
        video_table = self.get_annotated_videos_table(device=self.camera)
        vid_list = []
        for row in video_table:
            n_frames = int(row["nframes"])
            video_path = row['video_path']
            video_name = os.path.join(video_path.split('/')[0], video_path.split('/')[1])
            video_full_path = os.path.join(self.dataset_path, video_path)
            if not video_name in self.video_list:
                continue
            if n_frames < 66 * self.frame_skip:  # check video length
                continue
            if not os.path.exists(video_full_path):  # check if frame folder exists
                continue

            label = np.zeros((self.num_classes, n_frames), np.float32) # allow multi-class representation
            label[0, :] = np.ones((1, n_frames), np.float32)   # initialize all frames as background|transition
            video_id = row['id']
            annotation_table = self.get_video_annotations_table(video_id)
            for ann_row in annotation_table:
                atomic_action_id = ann_row["atomic_action_id"]  # map the labels
                object_id = ann_row["object_id"]
                action_id = self.get_action_id(atomic_action_id, object_id)
                end_frame = ann_row['ending_frame'] if ann_row['ending_frame'] < n_frames else n_frames
                if action_id is not None:
                    label[:, ann_row['starting_frame']:end_frame] = 0  # remove the background label
                    label[action_id, ann_row['starting_frame']:end_frame] = 1

            vid_list.append(
                (video_full_path, label, n_frames))  # 0 = duration - irrelevant for initial tests, used for start
        return vid_list

    def get_clips(self):
        # extract equal length video clip segments from the full video dataset
        clip_dataset = []
        label_count = np.zeros(self.num_classes)
        for i, data in enumerate(self.video_set):
            n_frames = data[2]
            n_clips = int(n_frames / (self.frames_per_clip * self.frame_skip))
            remaining_frames = n_frames % (self.frames_per_clip * self.frame_skip)
            for j in range(0, n_clips):
                for k in range(0, self.frame_skip):
                    start = j * self.frames_per_clip * self.frame_skip + k
                    end = (j + 1) * self.frames_per_clip * self.frame_skip
                    label = data[1][:, start:end:self.frame_skip]
                    label_count = label_count + np.sum(label, axis=1)
                    frame_ind = np.arange(start, end, self.frame_skip).tolist()
                    clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, 0))
            if not remaining_frames == 0:
                frame_pad =  self.frames_per_clip - remaining_frames
                start = n_clips * self.frames_per_clip * self.frame_skip + k
                end = start + remaining_frames
                label = data[1][:, start:end:self.frame_skip]
                label_count = label_count + np.sum(label, axis=1)
                label = data[1][:, start-frame_pad:end:self.frame_skip]
                frame_ind = np.arange(start-frame_pad, end, self.frame_skip).tolist()
                clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, frame_pad))
        return clip_dataset, label_count

    def load_rgb_frames(self, video_full_path, frame_ind, input_type='rgb'):
        # load video file and extract the frames
        frames = []
        for i in frame_ind:
            if input_type == 'rgb':
                img_filename = os.path.join(video_full_path, str(i).zfill(6) + '.jpg')
                img = torchvision.io.read_image(img_filename).permute(1, 2, 0)
            else:
                # img_filename = os.path.join(video_full_path, str(i).zfill(6) + '.png') #doesnt support 16 bit frames
                img = cv2.imread(os.path.join(video_full_path, str(i).zfill(6) + '.png'),
                                       cv2.IMREAD_ANYDEPTH).astype(np.float32)
                img = cv2.flip(img, 1)  # images were saved flipped
                img = torchvision.transforms.functional.to_tensor(img)


            img = img / 255.
            if input_type == 'depth':
                img = img.repeat([1, 1, 3])
            frames.append(img)
        return torch.stack(frames, 0)

    def load_pc(self, video_full_path, frame_ind, n_points=None):
        # load point cloud fiels

        frames = []
        for counter, i in enumerate(frame_ind):
            pc_filename = os.path.join(video_full_path, str(i).zfill(6) + '.ply')
            plydata = plyfile.PlyData.read(pc_filename)
            d = np.asarray(plydata['vertex'].data)
            pc = np.column_stack([d[p.name] for p in plydata['vertex'].properties])
            pc = torch.from_numpy(pc)
            if counter == 0: # set the translation and scale consistently throughout the sequence
                t = torch.mean(pc[:, 0:3], axis=0)
                s = torch.linalg.norm(torch.max(torch.abs(pc[:, 0:3] - t), axis=0)[0])
            pc[:, 0:3] = (pc[:, 0:3] - t) / s

            if n_points is not None:
                indices = torch.randperm(len(pc))[:n_points]
                pc = pc[indices, :]
            frames.append(pc.t())
        return torch.stack(frames, 0)

    def load_poses(self, video_full_path, frame_ind):
        pose_seq = []
        video_full_path = video_full_path.replace('images', self.pose_path)
        for i in frame_ind:
            pose_json_filename = os.path.join(video_full_path,
                                              'scan_video_' + str(i).zfill(12) + '_keypoints' + '.json')
            # data = utils.read_pose_json(pose_json_filename)
            with open(pose_json_filename) as json_file:
                data = json.load(json_file)
            data = data['people']
            if len(data) > 1:
                pose = self.get_active_person(data, center=(960, 540), min_bbox_area=20000)
            else:
                pose = np.array(data[0]['pose_keypoints_2d'])  # x,y,confidence

            pose = pose.reshape(-1, 3)
            pose_seq.append(pose)

        pose_seq = torch.tensor(pose_seq, dtype=torch.float32)
        pose_seq = pose_seq[:, :, 0:2].unsqueeze(-1)  # format: frames, joints, coordinates, N_people
        pose_seq = pose_seq.permute(2, 0, 1, 3)  # format: coordinates, frames, joints, N_people

        return pose_seq

    def get_active_person(self, people, center=(960, 540), min_bbox_area=20000):
        """
           Select the active skeleton in the scene by applying a heuristic of findng the closest one to the center of the frame
           then take it only if its bounding box is large enough - eliminates small bbox like kids
           Assumes 100 * 200 minimum size of bounding box to consider
           Parameters
           ----------
           data : pose data extracted from json file
           center: center of image (x, y)
           min_bbox_area: minimal bounding box area threshold

           Returns
           -------
           pose: skeleton of the active person in the scene (flattened)
           """

        pose = None
        min_dtc = float('inf')  # dtc = distance to center
        for person in people:
            current_pose = np.array(person['pose_keypoints_2d'])
            joints_2d = np.reshape(current_pose, (-1, 3))[:, :2]
            if 'boxes' in person.keys():
                # maskrcnn
                bbox = person['boxes']
            else:
                # openpose
                idx = np.where(joints_2d.any(axis=1))[0]
                bbox = [np.min(joints_2d[idx, 0]),
                        np.min(joints_2d[idx, 1]),
                        np.max(joints_2d[idx, 0]),
                        np.max(joints_2d[idx, 1])]

            A = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # bbox area
            bbox_center = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)  # bbox center

            dtc = np.sqrt(np.sum((np.array(bbox_center) - np.array(center)) ** 2))
            if dtc < min_dtc:
                closest_pose = current_pose
                if A > min_bbox_area:
                    pose = closest_pose
                    min_dtc = dtc
        # if all bboxes are smaller than threshold, take the closest
        if pose is None:
            pose = closest_pose
        return pose

    def compute_skeleton_distance_to_center(self, skeleton, center=(960, 540)):
        """
        Compute the average distance between a given skeleton and the cetner of the image
        Parameters
        ----------
        skeleton : 2d skeleton joint poistiions
        center : image center point

        Returns
        -------
            distance: the average distance of all non-zero joints to the center
        """
        idx = np.where(skeleton.any(axis=1))[0]
        diff = skeleton - np.tile(center, [len(skeleton[idx]), 1])
        distances = np.sqrt(np.sum(diff ** 2, 1))
        mean_distance = np.mean(distances)

        return mean_distance

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.clip_set)

    def __getitem__(self, index):
        video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]
        out_dict = {'pose': [], 'rgb': [], 'depth': [], 'pc': [],
                    'labels': torch.from_numpy(labels), 'vid_idx': vid_idx, 'frame_pad':frame_pad}
        if 'pose' in self.modalities:
            pose_full_path = video_full_path.replace('images', 'predictions/pose2d/openpose')
            poses = self.load_poses(pose_full_path, frame_ind)
            out_dict['pose'] = poses
        if 'rgb' in self.modalities:
            rgb = self.load_rgb_frames(video_full_path, frame_ind, input_type='rgb')
            rgb = torch.permute(rgb, [3, 0, 1, 2])  # permute for transform of shape [..., H, W]
            if self.transform is not None:
                rgb = self.transform(rgb)
            out_dict['rgb'] = rgb
        if 'depth' in self.modalities:
            depth_full_path = video_full_path.replace('images', 'depth')
            depth = self.load_rgb_frames(depth_full_path, frame_ind, input_type='depth')
            depth = torch.permute(depth, [3, 0, 1, 2])  # permute for transform of shape [..., H, W]
            if self.transform is not None:
                depth = self.transform(depth)
            out_dict['depth'] = depth
        if 'pc' in self.modalities:
            pc_full_path = video_full_path.replace('images', 'point_clouds')
            pc = self.load_pc(pc_full_path, frame_ind, self.n_points)
            out_dict['pc'] = pc
        return out_dict
