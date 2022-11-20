from argparse import ArgumentParser
import h5py
import sys
import visualization
import numpy as np
import random

# Subject ids
sids = ['50002', '50004', '50007', '50009', '50020',
        '50021', '50022', '50025', '50026', '50027']
# Sequences available for each subject id are listed in scripts/subjects_and_sequences.txt

parser = ArgumentParser(description='Save sequence registrations as obj')
parser.add_argument('--path', type=str, default='/home/sitzikbs/Datasets/dfaust/registrations_f.hdf5',
                    help='dataset path in hdf5 format')
parser.add_argument('--seq', type=str, default='punching', help='sequence name')
parser.add_argument('--sid', type=str, default='50004', choices=sids, help='subject id')

args = parser.parse_args()

sidseq = args.sid + '_' + args.seq
with h5py.File(args.path, 'r') as f:
    if sidseq not in f:
        print('Sequence %s from subject %s not in %s' % (args.seq, args.sid, args.path))
        f.close()
        sys.exit(1)
    verts = f[sidseq][()].transpose([2, 0, 1])
    faces = f['faces'][()]



# visualization.mesh_seq_vis(verts, faces)
idxs = np.arange(6890)
random.shuffle(idxs)
visualization.pc_seq_vis(verts[:, idxs[0:2048]])