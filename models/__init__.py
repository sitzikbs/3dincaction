import os
import importlib
import sys

from .pointnet import PointNet1, PointNet1Basic
from .pointnet2_cls_ssg import PointNet2, PointNet2Basic
from .pointnet2_cls_msg import PointNet2MSG, PointNet2MSGBasic
from .pytorch_3dmfv import FourDmFVNet
from .patchlets import PointNet2Patchlets
from .set_transformer import SetTransformerTemporal
from .tpatch_trajectory import tPatchTraj
from .DGCNN import DGCNN
from .pstnet import PSTnet

__all__ = {
    'pn1': PointNet1,
    'pn1_4d_basic': PointNet1Basic,
    'pn2': PointNet2,
    'pn2_4d_basic': PointNet2Basic,
    'pn2_msg': PointNet2MSG,
    'pn2_msg_4d_basic': PointNet2MSGBasic,
    'pn2_patchlets': PointNet2Patchlets,
    '3dmfv': FourDmFVNet,
    'set_transformer': SetTransformerTemporal,
    'tpatch_trajectory': tPatchTraj,
    'dgcnn': DGCNN,
    'pstnet': PSTnet
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
    'pn2_msg': "pointnet2_cls_msg.py",
    'pn2_msg_4d_basic': "pointnet2_cls_msg.py",
    'pn2_patchlets': "patchlets.py",
    '3dmfv': "pytorch_3dmfv.py",
    'set_transformer': 'set_transformer.py',
    'tpatch_trajectory': 'tpatch_trajectory.py',
    'dgcnn': 'DGCNN.py',
    'pstnet': 'pstnet.py'
}

def build_model_from_logdir(logdir, model_cfg, num_classes, frames_per_clip):
    pc_model = model_cfg.get('pc_model')
    model_instance = __all__[pc_model]
    model_name = model_instance.__name__
    file_name = file_name_dict.get(pc_model)

    spec = importlib.util.spec_from_file_location(model_name, os.path.join(logdir, 'models', file_name))
    import_model = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = import_model
    spec.loader.exec_module(import_model)
    model = model_instance(model_cfg=model_cfg, num_class=num_classes, n_frames=frames_per_clip)
    return model
