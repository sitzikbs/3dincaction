import os
import importlib
import sys

from .pointnet import PointNet1, PointNet1Basic
from .pointnet2_cls_ssg import PointNet2, PointNet2Basic
from .pytorch_3dmfv import FourDmFVNet
from .tpatches import TPatchesInAction
from .set_transformer import SetTransformerTemporal
from .tpatch_trajectory import tPatchTraj
from .DGCNN import DGCNN
from .pstnet import PSTnet
from .PST_Transformer import PSTTransformer
from .P4Transformer import P4Transformer

__all__ = {
    'pn1': PointNet1,
    'pn1_4d_basic': PointNet1Basic,
    'pn2': PointNet2,
    'pn2_4d_basic': PointNet2Basic,
    'tpatches': TPatchesInAction,
    '3dmfv': FourDmFVNet,
    'set_transformer': SetTransformerTemporal,
    'tpatch_trajectory': tPatchTraj,
    'dgcnn': DGCNN,
    'pst_transformer': PSTTransformer,
    'pstnet': PSTnet,
    'p4transformer': P4Transformer,
}

def build_model(model_cfg, num_class, frames_per_clip):
    model = __all__[model_cfg['pc_model']](
        model_cfg=model_cfg, num_class=num_class, n_frames=frames_per_clip
    )
    return model

file_name_dict = {
    'pn1': "pointnet.py",
    'pn1_4d_basic': "pointnet.py",
    'pn2': "pointnet2_cls_ssg.py",
    'pn2_4d_basic': "pointnet2_cls_ssg.py",
    'tpatches': "tpatches.py",
    '3dmfv': "pytorch_3dmfv.py",
    'set_transformer': 'set_transformer.py',
    'tpatch_trajectory': 'tpatch_trajectory.py',
    'dgcnn': 'DGCNN.py',
    'pstnet': 'pstnet.py',
    'pst_transformer': 'PST_Transformer.py',
    'p4transformer': 'P4Transformer.py',
}


class build_model_from_logdir(object):
    def __init__(self, logdir, model_cfg, num_classes, frames_per_clip):
        pc_model = model_cfg.get('pc_model')
        model_instance = __all__[pc_model]
        model_name = model_instance.__name__
        file_name = file_name_dict.get(pc_model)

        spec = importlib.util.spec_from_file_location(model_name, os.path.join(logdir, 'models', file_name))
        import_model = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = import_model
        spec.loader.exec_module(import_model)
        self.model = model_instance(model_cfg=model_cfg, num_class=num_classes, n_frames=frames_per_clip)
    def get(self):
        return self.model
