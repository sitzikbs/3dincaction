import os
import numpy as np
import torchvision
import pyvista as pv
import torch
import shutil
from PIL import Image, ImageOps


point_cloud_dataset_path = '/mnt/IronWolf/Datasets/ANU_ikea_dataset_processed'
img_dataset_path = '/mnt/IronWolf/Datasets/ANU_ikea_dataset'
scan_rel_path = 'Lack_TV_Bench/0005_white_table_10_04_2019_08_28_14_43/dev3'

outdir = './log/ikea_asm_selected_examples/'
point_size = 3
frames_to_export = [87, 135, 1934, 2133]  # Chrisitan: align leg, spin leg, flip table, attach shelf

os.makedirs(outdir, exist_ok=True)
img_path = os.path.join(img_dataset_path, scan_rel_path, 'images')
pc_path = os.path.join(point_cloud_dataset_path, scan_rel_path, 'point_clouds')


# load image and point cloud and plot
img_file_list = os.listdir(img_path)
n_frames = len(img_file_list)
for i in range(n_frames):
    if i in frames_to_export:
        full_img_filename = os.path.join(img_path, str(i).zfill(6) + '.jpg')
        full_pc_filename = os.path.join(pc_path, str(i).zfill(6) + '.ply')
        # shutil.copyfile(full_img_filename, os.path.join(outdir, str(i).zfill(6) + '_img.jpg'))
        # img = torchvision.io.read_image(full_img_filename)
        img_PIL = ImageOps.mirror(Image.open(full_img_filename))
        img_PIL.save(os.path.join(outdir, str(i).zfill(6) + '_img.png'))
        pc = pv.read(full_pc_filename)


        pc.points = pc.points - pc.points.mean(0)
        pc.points = pc.points / (np.linalg.norm(pc.points.max(0)))

        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(pc, render_points_as_spheres=True, point_size=point_size, rgb=True)
        pl.set_background('white', top='white')
        pl.camera.position = [0.1, 0.1, -1.]
        pl.camera.focal_point = [0.1, 0., 0.0]
        pl.camera.up = [0., -1., 0.]
        pl.camera.zoom(0.6)
        pc_out_filename = os.path.join(outdir, str(i).zfill(6) + '_pc.png')
        pl.show(screenshot=pc_out_filename)
