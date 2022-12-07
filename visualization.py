import pyvista as pv
import numpy as np

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


def plot_pc_ov(verts, text=None, color=None):
    pv.global_theme.cmap = 'glasby'
    if color is None:
        color = 0.5*np.ones([len(verts), len(verts[0])])

    pc = pv.PolyData(verts[0])
    pc['scalars'] = color[0]
    pl = pv.Plotter()
    pl.add_mesh(pc, render_points_as_spheres=True, scalars=pc['scalars'])
    # pl.add_mesh(pc, render_points_as_spheres=True)
    pl.show()

def get_pc_pv_image(verts, text=None, color=None, point_size=50):
    pv.global_theme.cmap = 'cet_glasbey_bw'
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