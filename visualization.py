import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


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

def pc_seq_vis(verts, text=None, color=None):

    n_frames = verts.shape[0]
    if color is None:
        color = 0.5*np.ones([len(verts), len(verts[0])])

    pc = pv.PolyData(verts[0])
    pc['scalars'] = color[0]

    pl = pv.Plotter()
    pl.add_mesh(pc, render_points_as_spheres=True, scalars=pc['scalars'])
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
