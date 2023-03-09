import os
import numpy as np
import torchvision
import pyvista as pv
import torch
import shutil
from PIL import Image, ImageOps
from models.patchlets import PatchletsExtractor
import visualization
import models.pointnet2_utils as utils


point_cloud_dataset_path = '/mnt/IronWolf/Datasets/ANU_ikea_dataset_processed'
img_dataset_path = '/mnt/IronWolf/Datasets/ANU_ikea_dataset'
scan_rel_path = 'Lack_TV_Bench/0005_white_table_10_04_2019_08_28_14_43/dev3'

outdir = './log/ikea_asm_selected_examples/flip_table/'
outdir_img = os.path.join(outdir, 'img')
outdir_pc = os.path.join(outdir, 'pc')
for dirname in [outdir, outdir_img,outdir_pc ]:
    os.makedirs(dirname, exist_ok=True)
point_size = 3
# frames_to_export = [87, 135, 1934, 2133]  # Chrisitan: align leg, spin leg, flip table, attach shelf
# frames_to_export = [90, 135, 720, 730, 740, 750, 760, 770, 780, 785, 790, 795, 800, 810, 820, 830, 840, 850, 1530, 1540, 1550,
#                     1560, 1570, 1580, 1590, 1600, 1610, 1620, 1890, 1900, 1910, 1920, 1930, 1950, 1960]
frames_to_export = [90, 135, 720, 730, 740, 750, 760, 770, 780, 785, 790, 795, 800, 810, 820, 830, 840, 850, 1530, 1540, 1550,
                    1560, 1570, 1580, 1590, 1600, 1610, 1620, 1860, 1865, 1870, 1875, 1880, 1885, 1890, 1895, 1900,1905, 1910,
                    1915, 1920, 1925, 1930, 1935, 1940, 1945, 1950, 1955, 1960]

img_path = os.path.join(img_dataset_path, scan_rel_path, 'images')
pc_path = os.path.join(point_cloud_dataset_path, scan_rel_path, 'point_clouds')


# load image and point cloud and plot
img_file_list = os.listdir(img_path)
n_frames = len(img_file_list)
pc_seq = []
for i in range(n_frames):
    if i in frames_to_export:
        full_img_filename = os.path.join(img_path, str(i).zfill(6) + '.jpg')
        full_pc_filename = os.path.join(pc_path, str(i).zfill(6) + '.ply')
        # shutil.copyfile(full_img_filename, os.path.join(outdir, str(i).zfill(6) + '_img.jpg'))
        # img = torchvision.io.read_image(full_img_filename)
        img_PIL = ImageOps.mirror(Image.open(full_img_filename))
        img_PIL.save(os.path.join(outdir_img, str(i).zfill(6) + '_img.png'))
        pc = pv.read(full_pc_filename)
        pc_seq.append(pc)

        pc.points = pc.points - pc.points.mean(0)
        if i == frames_to_export[0]: # get the scale from the frist frame to avoid flickering
            s = (np.linalg.norm(pc.points.max(0)))
        pc.points = pc.points / s

        pl = pv.Plotter(off_screen=True)
        mesh = pl.add_mesh(pc, render_points_as_spheres=True, point_size=point_size, rgb=True)
        pl.set_background('white', top='white')
        pl.camera.position = [0.1, 0.1, -1.]
        pl.camera.focal_point = [0.1, 0., 0.0]
        pl.camera.up = [0., -1., 0.]
        pl.camera.zoom(0.6)
        pc_out_filename = os.path.join(outdir_pc, str(i).zfill(6) + '_pc.png')
        pl.show(screenshot=pc_out_filename)



# Export point clouds with tpatches
def remove_patchlet_points_from_pc(point_seq, patchlet_point_list, features):
    '''
    get a point sequence and a patchlet point list sequence and remove the patchlet points from the sequence,
    used for visualization
    :param point_seq:
    :param patchlet_point_list:
    :return:
    '''
    t, n, _ = point_seq.shape
    out_points_seq = []
    out_feat_seq = []
    all_patchlet_points = np.concatenate(patchlet_point_list, axis=1)
    for i in range(t):
        rows_to_delete = []
        for j in range(n):
            # if np.any(np.linalg.norm(point_seq[i][j] - all_patchlet_points[i], axis=1) < 1e-2):
            if point_seq[i][j] in all_patchlet_points[i]:
                rows_to_delete.append(j)
        out_points_seq.append(np.delete(point_seq[i], rows_to_delete, 0))
        if features is not None:
            out_feat_seq.append(np.delete(features[i], rows_to_delete, 0))
    return out_points_seq, out_feat_seq

k = 64
n_points = 20000
ndownsampled_points = 20000

point_size = 8
extract_pachlets = PatchletsExtractor(k=k, npoints=n_points, sample_mode='nn',
                                      add_centroid_jitter=0.0, downsample_method='mean_var_t')
downsampled_pc = []
downsampled_colors = []
for pc in pc_seq:
    with torch.no_grad():
        point_color = np.array(pc.active_scalars)
        points = torch.tensor(pc.points).cuda()
        idxs = utils.farthest_point_sample(points.unsqueeze(0).contiguous(), ndownsampled_points).to(torch.int64).squeeze()
        downsampled_pc.append(points[idxs, :])
        downsampled_colors.append(point_color[idxs.detach().cpu().numpy(), :])

patchlet_ids = [1750, 1200, 15000, 19000]
point_seq = torch.stack(downsampled_pc).unsqueeze(0).cuda()
with torch.no_grad():
    patchlet_dict = extract_pachlets(point_seq)
patchlet_points = [patchlet_dict['patchlet_points'].squeeze()[:, id].cpu().numpy() for id in patchlet_ids]
point_seq, point_color = remove_patchlet_points_from_pc(point_seq[0].cpu().numpy(), patchlet_points,
                                                        np.stack(downsampled_colors))
visualization.export_pc_seq(point_seq, patchlet_points, text=None,
                            color=point_color, cmap=None,
                            point_size=point_size, output_path=os.path.join(outdir, 'patchlets'),
                            show_patchlets=True, show_full_pc=True,
                            reduce_opacity=False, view='ikea_front', light=False)