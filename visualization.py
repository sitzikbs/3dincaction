import pyvista as pv
import numpy as np

class MeshCustomRoutine:
    def __init__(self, vert_seq, faces, mesh_obj):

        self.seq = vert_seq
        # default parameters
        self.kwargs = { 'value': 0, }

        self.output = mesh_obj
        self.faces = faces

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update(value)

    def update(self, value):
        # This is where you call your simulation
        mesh = pv.PolyData(self.seq[int(value)], self.faces)
        self.output.overwrite(mesh)
        return

class PCCustomRoutine:
    def __init__(self, vert_seq, point_obj, text):

        self.seq = vert_seq
        self.text = text
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
            pv.global_theme.title = self.text[int(value)]
        self.output.overwrite(pc)
        return


def mesh_seq_vis(verts, faces, text=None):
    faces = np.concatenate([3 * np.ones([faces.shape[0], 1], dtype=np.int16), faces], axis=1)
    n_frames = verts.shape[0]

    mesh = pv.PolyData(verts[0], faces)
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    engine = MeshCustomRoutine(verts, faces, mesh)
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
        pl.add_title(text)


    pl.show()

def pc_seq_vis(verts, text=None):

    n_frames = verts.shape[0]

    pc = pv.PolyData(verts[0])
    pl = pv.Plotter()
    pl.add_mesh(pc, render_points_as_spheres=True)
    engine = PCCustomRoutine(verts, pc, text)
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