import argparse
import torch
from DfaustDataset import DfaustActionDataset as Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='knn', help='knn  | sinkhorn | correformer ')
parser.add_argument('--frames_per_clip', type=int, default=1, help='number of frames in a clip sequence')
parser.add_argument('--batch_size', type=int, default=1, help='number of clips per batch')
parser.add_argument('--n_points', type=int, default=1024, help='number of points in a point cloud')
parser.add_argument('--model_path', type=str, default='./log/pn1_f1_p1024_shuffle_once_sampler_weighted/',
                    help='path to model save dir')
parser.add_argument('--model', type=str, default='000030.pt', help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/dfaust/', help='path to dataset')
parser.add_argument('--visualize_results', type=int, default=False, help='visualzies the first subsequence in each batch')
parser.add_argument('--gender', type=str,
                    default='female', help='female | male | all indicating which subset of the dataset to use')
args = parser.parse_args()

test_dataset = Dataset(args.dataset_path,  set='test', n_points=args.n_points, shuffle_points='each_frame',
                       gender=args.gender)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                              pin_memory=True, drop_last=True)



