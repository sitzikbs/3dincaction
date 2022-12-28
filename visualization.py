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


def plot_pc_pv(verts, text=None, color=None, cmap=None):
    if cmap is not None:
        pv.global_theme.cmap = cmap
    else:
        pv.global_theme.cmap = 'cet_glasbey_bw'

    if color is None:
        color = 0.5*np.ones([len(verts), len(verts[0])])

    pc = pv.PolyData(verts[0])
    pc['scalars'] = color[0]
    pl = pv.Plotter()
    pl.add_mesh(pc, render_points_as_spheres=True, scalars=pc['scalars'])
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


def plot_correformer_outputs(sim_mat, corr21, corr12, best_buddies_mat,
                             gt_corr, max_corr21, max_corr12, err_mat,
                             title_text=''):
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 20))
    cax = ax[0, 0].matshow(sim_mat, interpolation='nearest')
    ax[0, 0].title.set_text('Similarity matrix')

    ax[0, 1].matshow(corr21, interpolation='nearest')
    ax[0, 1].title.set_text('softmax cols')

    ax[0, 2].matshow(corr12, interpolation='nearest')
    ax[0, 2].title.set_text('softmax rows')

    ax[0, 3].matshow(best_buddies_mat, interpolation='nearest')
    ax[0, 3].title.set_text('best buddies')

    # Row 2
    ax[1, 0].matshow(gt_corr, interpolation='nearest')
    ax[1, 0].title.set_text('GT')

    ax[1, 1].matshow(max_corr21, interpolation='nearest')
    ax[1, 1].title.set_text('max_corr21')

    ax[1, 2].matshow(max_corr12, interpolation='nearest')
    ax[1, 2].title.set_text('max_corr12')

    ax[1, 3].matshow(err_mat, interpolation='nearest')
    ax[1, 3].title.set_text('Error mat')

    fig.suptitle(title_text)
    # fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
    plt.show()
