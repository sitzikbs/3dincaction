import os
import numpy as np
import csv
import torchvision
import plyfile
import pyvista as pv
from PIL import Image, ImageOps

scan_path = '/home/sitzikbs/Datasets/ego_full_scans/Stool'
outdir = './log/ikeaego_vie_pc_eye_aligned/Stool/'
point_size = 4
# frames_to_export = [233, 562, 1012, 1209]  # Drawer
frames_to_export = [373, 1315, 2353, 2719]  # Stool:  use drill, attach stool side, flip stool, align top stool step

os.makedirs(outdir, exist_ok=True)
img_path = os.path.join(scan_path, 'pv')
pc_path = os.path.join(scan_path, 'Depth Long Throw')

# load eye gaze data
eye_csv_data = {'frame_id': [], 'origin': [], 'target': [], 'projection': []}
with open(os.path.join(scan_path, 'norm_proc_eye_data.csv'), 'r') as read_obj:
    csv_reader = csv.reader(read_obj, delimiter=',')

    for row in csv_reader:
        if row:
            eye_csv_data['frame_id'].append(int(row[0]))
            # eye_csv_data['origin'].append([float(x) for x in row[1:4]])
            # eye_csv_data['target'].append([float(x) for x in row[4:7]])
            eye_csv_data['origin'].append([float(row[1]), float(row[2]), float(row[3])])
            eye_csv_data['target'].append([float(row[4]), float(row[5]), float(row[6])])
            eye_csv_data['projection'].append([int(x) for x in row[7:9]])


# load image and point cloud and plot
img_file_list = os.listdir(img_path)
n_frames = len(img_file_list)
for i in range(n_frames):
    if i in frames_to_export:
        full_img_filename = os.path.join(img_path, str(i) + '.png')
        full_pc_filename = os.path.join(pc_path, str(i) + '.ply')
        # img = torchvision.io.read_image(full_img_filename)
        img_PIL = Image.open(full_img_filename)
        img_PIL.save(os.path.join(outdir, str(i).zfill(6) + '_img.png'))
        pc = pv.read(full_pc_filename)

        pl = pv.Plotter()
        pl.add_mesh(pc, render_points_as_spheres=True, point_size=point_size, rgb=True)
        pl.set_background('white', top='white')
        pl.camera.position = eye_csv_data['origin'][i]
        pl.camera.focal_point = eye_csv_data['target'][i]
        # side = np.cross(np.array([0., 1., 0.]),
        #                 np.array(eye_csv_data['target'][i]) - np.array(eye_csv_data['origin'][i]))
        # up = np.cross(np.array(eye_csv_data['target'][i]) - np.array(eye_csv_data['origin'][i]), side)
        pl.camera.up = [0., 1., 0.]
        pl.camera.zoom(0.5)
        pl.show()