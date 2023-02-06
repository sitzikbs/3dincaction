import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import os

COLOR_PALLET = {'darkred': [0.7,0.1,0.1], 'darkgreen': [0.1, 0.7, 0.1], 'dblue': [0.2,0.2,0.8],
                'maroon': [0.76,.13,.28],
                'burntorange': [0.81,.33,0], 'cyan': [0.0,0.7,0.94], 'salmon': [0.99,0.51,0.46],
                'green':[0.03,0.91,0.43] }
CAMLOC = {'iso': (1, 1, 1), 'front': (0, 0, 2), 'side': (2, 0, 0), 'top': (0, 2, 0)}

class MeshCustomRoutine:
    def __init__(self, vert_seq, faces, mesh_obj, text, pl, color):

        self.seq = vert_seq
        # default parameters
        self.kwargs = { 'value': 0, }
        self.text = text
        self.pl = pl
        self.output = mesh_obj
        self.faces = faces
        self.color = color

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update(value)

    def update(self, value):
        # This is where you call your simulation
        mesh = pv.PolyData(self.seq[int(value)], self.faces)
        if self.text is not None:
            self.pl.add_title(self.text[int(value)])
        if self.color is not None:
            mesh['scalars'] = self.color[int(value)]
        self.output.overwrite(mesh)
        return

class PCCustomRoutine:
    def __init__(self, vert_seq, point_obj, text, pl, color):

        self.seq = vert_seq
        self.text = text
        self.pl = pl
        self.color = color
        # default parameters
        self.kwargs = {'value': 0, }
        self.output = point_obj

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update(value)

    def update(self, value):
        # This is where you call your simulation
        pc = pv.PolyData(self.seq[int(value)])
        if self.text is not None:
            self.pl.add_title(self.text[int(value)])
        if self.color is not None:
            pc['scalars'] = self.color[int(value)]

        self.output.overwrite(pc)
        return

class PCAttnRoutine:
    def __init__(self, vertices1, vertices2, self_corr, gt_corr, point_ids, point_obj1, point_obj2, text, pl, attn_maps):

        self.vertices1 = vertices1
        self.vertices2 = vertices2
        self.self_corr = self_corr
        self.gt_corr = gt_corr
        self.point_ids = point_ids
        self.text = text
        self.pl = pl
        self.color = attn_maps
        self.n_maps = len(attn_maps)
        self.point_id = 0
        # default parameters
        self.kwargs = {'map_id': 0, 'point_id': 0 }
        self.output1 = point_obj1
        self.output2 = point_obj2

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        pc1 = pv.PolyData(self.vertices1)
        pc2 = pv.PolyData(self.vertices2)

        point_idx = int(self.kwargs['point_id'])
        if self.kwargs['map_id'] < self.n_maps:
            pc1['scalars'] = self.color[int(self.kwargs['map_id'])][0, point_idx]
            pc2['scalars'] = self.self_corr[point_idx]
            # pc1['scalars'] = self.gt_corr[point_idx]
        else:
            pc1['scalars'] = self.gt_corr[point_idx]
            pc2['scalars'] = self.self_corr[point_idx]

        self.output1.overwrite(pc1)
        self.output2.overwrite(pc2)
        return


class PCPatchletsRoutine:
    def __init__(self, vertices, point_obj, patchlet_obj, text, pl, patchlets):

        self.vertices = vertices
        self.text = text
        self.pl = pl
        self.patchlets = patchlets

        self.point_id = 0
        # default parameters
        self.kwargs = {'point_id': 0, 'frame_id': 0}
        self.output = point_obj
        self.output_patchlet = patchlet_obj


    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        frame_idx = int(self.kwargs['frame_id'])
        point_idx = int(self.kwargs['point_id'])
        pc = pv.PolyData(self.vertices[frame_idx])
        pc_patchlet = pv.PolyData(self.patchlets[frame_idx, point_idx])
        self.output.overwrite(pc)
        self.output_patchlet.overwrite(pc_patchlet)
        return

class PCPatchletsAllRoutine:
    def __init__(self, vertices, point_obj_list, text, pl):

        self.vertices = vertices
        self.text = text
        self.pl = pl
        self.point_id = 0
        # default parameters
        self.kwargs = {'frame_id': 0}
        self.output = point_obj_list


    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        frame_idx = int(self.kwargs['frame_id'])
        pc = []
        for i, patchlet_points in enumerate(self.vertices[frame_idx]):
            pc.append(pv.PolyData(patchlet_points))
            pc[i]['scalars'] = i * np.ones(len(patchlet_points))
            self.output[i].overwrite(pc[i])
        return

class PCPatchletsPatchRoutine:
    def __init__(self, vertices, point_obj, text, pl, patch_maps):

        self.vertices = vertices
        self.text = text
        self.pl = pl
        self.color = patch_maps
        self.n_maps = len(patch_maps)
        self.point_id = 0
        # default parameters
        self.kwargs = {'point_id': 0, 'frame_id': 0}
        self.output = point_obj


    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        frame_idx = int(self.kwargs['frame_id'])
        point_idx = int(self.kwargs['point_id'])
        pc = pv.PolyData(self.vertices[frame_idx, point_idx])
        pc['scalars'] = self.color[frame_idx, point_idx]
        self.output.overwrite(pc)
        return

def pc_patchlet_vis(verts, patchlets, text=None):
    # n_frames, _, _ = verts.shape
    n_frames, n_points, k, _ = patchlets.shape
    pc = pv.PolyData(verts[0])
    patchlet_pc = pv.PolyData(patchlets[0][0])

    pl = pv.Plotter()
    pl.add_mesh(pc, render_points_as_spheres=True, color='grey', point_size=25, clim=[0, 1])
    pl.add_mesh(patchlet_pc, render_points_as_spheres=True, color='yellow', point_size=25, clim=[0, 1])
    engine = PCPatchletsRoutine(verts, pc, patchlet_pc, text, pl, patchlets)

    pl.add_slider_widget(
        callback=lambda value: engine('point_id', value),
        rng=[0, n_points-1],
        value=0,
        title="point_id",
        pointa=(0.025, 0.2),
        pointb=(0.31, 0.2),
        style='modern',
        event_type='always'
    )
    pl.add_slider_widget(
        callback=lambda value: engine('frame_id', value),
        rng=[0, n_frames - 1],
        value=0,
        title="frame",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        event_type='always'
    )
    if text is not None:
        pl.add_title(text[0])

    pl.camera.position = (1, 1, 1)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(0.5)
    pl.set_background('white', top='white')
    pl.show()

def pc_patchlet_points_vis(verts, text=None):
    n_frames, n_points, k, d = verts.shape
    pv.global_theme.cmap = 'cet_glasbey_bw'

    pl = pv.Plotter()
    pc = []
    for i, patchlet_points in enumerate(verts[0]):
        pc.append( pv.PolyData(patchlet_points))
        pc[i]['scalars'] = i*np.ones(k)
        pl.add_mesh(pc[i], render_points_as_spheres=True, scalars=pc[i]['scalars'], point_size=25, clim=[0, n_points])
    engine = PCPatchletsAllRoutine(verts, pc, text, pl)

    pl.add_slider_widget(
        callback=lambda value: engine('frame_id', value),
        rng=[0, n_frames - 1],
        value=0,
        title="frame",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        event_type='always'
    )
    if text is not None:
        pl.add_title(text[0])

    pl.camera.position = (1, 1, 1)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(0.5)
    pl.show()

def pc_patchlet_patch_vis(patch_verts, point_dist, text=None):
    n_frames, n_points, k, _ = patch_verts.shape
    pc = pv.PolyData(patch_verts[0, 0])
    pc['scalars'] = point_dist[0, 0]

    pl = pv.Plotter()
    pl.add_mesh(pc, render_points_as_spheres=True, point_size=25, clim=[0, 1000])
    engine = PCPatchletsPatchRoutine(patch_verts, pc, text, pl, 1. / (point_dist + 1e-6))

    pl.add_slider_widget(
        callback=lambda value: engine('point_id', value),
        rng=[0, n_points-1],
        value=0,
        title="point_id",
        pointa=(0.025, 0.2),
        pointb=(0.31, 0.2),
        style='modern',
        event_type='always'
    )
    pl.add_slider_widget(
        callback=lambda value: engine('frame_id', value),
        rng=[0, n_frames - 1],
        value=0,
        title="frame",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        event_type='always'
    )
    if text is not None:
        pl.add_title(text[0])

    pl.camera.position = (1, 1, 1)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(0.5)
    pl.show()

def pc_attn_vis(verts1, verts2, self_corr, gt_corr, point_ids, attn_maps, text=None,color_rng=0.00175):
    verts2 = verts2 + (1, 0, 0)
    n_maps = len(attn_maps)
    n_points = len(verts1)
    pc1 = pv.PolyData(verts1)
    pc2 = pv.PolyData(verts2)
    pc1['scalars'] = gt_corr[0]
    pc2['scalars'] = self_corr[0]

    pl = pv.Plotter()
    pl.add_mesh(pc1, render_points_as_spheres=True, scalars=pc1['scalars'], point_size=25, clim=[0, color_rng])
    pl.add_mesh(pc2, render_points_as_spheres=True, scalars=pc2['scalars'], point_size=25, clim=[0, color_rng])
    engine = PCAttnRoutine(verts1, verts2, self_corr, gt_corr, point_ids, pc1, pc2, text, pl, attn_maps)
    pl.add_slider_widget(
        callback=lambda value: engine('map_id', value),
        rng=[0, n_maps],
        value=0,
        title="map_id",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        event_type='always'
    )
    pl.add_slider_widget(
        callback=lambda value: engine('point_id', value),
        rng=[0, n_points-1],
        value=0,
        title="point_id",
        pointa=(0.025, 0.2),
        pointb=(0.31, 0.2),
        style='modern',
        event_type='always'
    )
    if text is not None:
        pl.add_title(text[0])

    pl.camera.position = (1, 1, 1)
    pl.camera.focal_point = (0.5, 0, 0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(0.5)
    pl.show()


def mesh_seq_vis(verts, faces, text=None, color=None):
    faces = np.concatenate([3 * np.ones([faces.shape[0], 1], dtype=np.int16), faces], axis=1)
    n_frames = verts.shape[0]
    if color is None:
        color = 0.5*np.ones([len(verts), len(verts[0])])

    mesh = pv.PolyData(verts[0], faces)
    mesh['scalars'] = color[0]
    pl = pv.Plotter()
    pl.add_mesh(mesh, scalars=mesh['scalars'])
    engine = MeshCustomRoutine(verts, faces, mesh, text, pl, color)
    pl.add_slider_widget(
        callback=lambda value: engine('value', value),
        rng=[0, n_frames - 1],
        value=0,
        title="frame",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        event_type='always'
    )
    if text is not None:
        pl.add_title(text[0])

    pl.camera.position = (1, 1, 1)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(0.5)

    pl.show()

def pc_seq_vis(verts, text=None, color=None, point_size=25):

    n_frames = verts.shape[0]
    if color is None:
        color = 0.5*np.ones([len(verts), len(verts[0])])

    pc = pv.PolyData(verts[0])
    pc['scalars'] = color[0]

    pl = pv.Plotter()
    pl.add_mesh(pc, render_points_as_spheres=True, scalars=pc['scalars'], point_size=point_size)
    engine = PCCustomRoutine(verts, pc, text, pl, color)
    pl.add_slider_widget(
        callback=lambda value: engine('value', value),
        rng=[0, n_frames - 1],
        value=0,
        title="frame",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        event_type='always'
    )
    if text is not None:
        pl.add_title(text[0])

    pl.camera.position = (1, 1, 1)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(0.5)
    pl.show()



def plot_pc_pv(verts, text=None, color=None, cmap=None, point_size=50, ):
    if cmap is not None:
        pv.global_theme.cmap = cmap
    else:
        pv.global_theme.cmap = 'cet_glasbey_bw'

    if color is None:
        color = 0.5*np.ones([len(verts), len(verts[0])])

    pc = pv.PolyData(verts[0])
    pc['scalars'] = color[0]
    pl = pv.Plotter()
    pl.add_mesh(pc, render_points_as_spheres=True, scalars=pc['scalars'], point_size=point_size)
    # pl.add_mesh(pc, render_points_as_spheres=True)
    pl.show()

def get_pc_pv_image(verts, text=None, color=None, point_size=50, cmap='cet_glasbey_bw'):
    pv.global_theme.cmap = cmap
    if color is None:
        color = 0.5*np.ones([len(verts), len(verts[0])])

    pc = pv.PolyData(verts)
    pc['scalars'] = color
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(pc, render_points_as_spheres=True, scalars=pc['scalars'], point_size=point_size)

    # pl.add_mesh(pc, render_points_as_spheres=True)
    pl.camera.position = (1, 1, 1)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(0.5)

    pl.screenshot()
    image = pl.image
    pl.close()
    return image


def plot_correformer_outputs(mat_dict, titles_dict, title_text=''):
    nrows = int(np.ceil(len(mat_dict) / 4))
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))
    row = 0
    col = 0
    for i, key in enumerate(mat_dict.keys()):

        ax[row, col].matshow(mat_dict[key], interpolation='nearest')
        ax[row, col].title.set_text(titles_dict[key])

        col += 1
        if col == ncols :
            row += 1
            col = 0

    fig.suptitle(title_text)

    plt.show()

def plot_attention_maps(attention_maps, points, point_idx=0, title_text='', point_size=25):
    n_points = points.shape[1]
    n_heads = len(attention_maps)
    nrows = 2
    ncols = int(np.ceil((n_heads + 1) / 2))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20))

    row = 0
    col = 0
    # plot attention maps
    for i in range(n_heads):
        att_color = attention_maps[i][:, point_idx].squeeze().cpu().detach().numpy()
        pc_atten_img = get_pc_pv_image(points.squeeze().detach().cpu().numpy(), text=None,
                                         color=att_color, point_size=point_size, cmap='jet')#.transpose(2, 0, 1)
        ax[row, col].matshow(pc_atten_img, interpolation='nearest')
        ax[row, col].title.set_text('Attention map {}'.format(ncols*row + col))
        col += 1
        if col == 3:
            row += 1
            col = 0

    # plot point id
    point_id_color = np.zeros(n_points)
    point_id_color[point_idx] = 1
    pc_atten1_img = get_pc_pv_image(points.squeeze().detach().cpu().numpy(), text=None,
                                    color=point_id_color, point_size=point_size, cmap='jet')  # .transpose(2, 0, 1)
    ax[1, -1].matshow(pc_atten1_img, interpolation='nearest')
    ax[1, -1].title.set_text('point ID'.format(col + 1))

    fig.suptitle(title_text)

    plt.show()


def export_pc_seq(verts, patchlet_points, text=None, color=None, cmap=None, point_size=50, output_path='./',
                  show_patchlets=True, show_full_pc=True, reduce_opacity=False, view='iso'):

    if show_patchlets and show_full_pc:
        id_str = 'both'
    elif show_patchlets:
        id_str = 'patchlets'
    else:
        id_str = 'point_cloud'

    output_path = os.path.join(output_path, id_str, view)
    t = len(verts)
    alpha = 1.0
    os.makedirs(output_path, exist_ok=True)
    color_list = [COLOR_PALLET['dblue'], COLOR_PALLET['maroon'], COLOR_PALLET['cyan']]
    if cmap is not None:
        pv.global_theme.cmap = cmap
    else:
        pv.global_theme.cmap = 'cet_glasbey_bw'

    if color is None:
        # color = [0.92, 0.22, 0.12]
        color = [0.7, 0.33, 0.1]

    #Export visualiztaion
    for i in range(t):
        pl = pv.Plotter(off_screen=True)
        # pl.camera.position = (0, 0, 2)
        pl.camera.position = CAMLOC[view]
        pl.camera.focal_point = (0, 0, 0)
        pl.camera.up = (0.0, 1.0, 0.0)
        pl.camera.zoom(0.5)
        pl.set_background('white', top='white')
        pc = pv.PolyData(verts[i])
        if show_full_pc:
            if reduce_opacity:
                alpha = np.clip(0.99 - 2*i/t, 0., 1.)
            pl.add_mesh(pc, render_points_as_spheres=True, color=color, point_size=point_size, pbr=True,
                        roughness=0.9, diffuse=1.0, metallic=0.05, opacity=alpha)
        if show_patchlets:
            for j, patchlet_pc in enumerate(patchlet_points):
                pc = pv.PolyData(patchlet_pc[i])
                pl.add_mesh(pc, render_points_as_spheres=True, color=color_list[j], point_size=point_size, pbr=True,
                            roughness=0.9, diffuse=1.0, metallic=0.05)

        # set up lighting
        light = pv.Light((5, 5, 5), (0, 0, 0), 'white')
        pl.add_light(light)
        light = pv.Light((0, 2, 0), (0, 0, 0), 'white')
        pl.add_light(light)
        light = pv.Light((2, 0, 0), (0, 0, 0), 'white')
        pl.add_light(light)
        light = pv.Light((0, 0, 2), (0, 0, 0), 'white')
        pl.add_light(light)
        light = pv.Light((0, 0, -2), (0, 0, 0), 'white')
        pl.add_light(light)

        pl.show(screenshot=os.path.join(output_path, str(i).zfill(6) + '.png'))