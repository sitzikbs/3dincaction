import time

import torch
import os
import yaml
import argparse
import i3d_utils as utils
import torch.nn as nn
import numpy as np
import random
from models import build_model
from datasets import build_dataloader

import wandb
from tqdm import tqdm


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='./log/', help='path to model save dir')
parser.add_argument('--identifier', type=str, default='timing_and_params', help='unique run identifier')
parser.add_argument('--config', type=str, default='./config_timing.yaml', help='path to yaml config file')
args = parser.parse_args()


def main():
    cfg = yaml.safe_load(open(args.config))
    logdir = os.path.join(args.logdir, args.identifier)
    os.makedirs(logdir, exist_ok=True)

    with open(os.path.join(logdir, 'config.yaml'), 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    # need to add argparse
    run(cfg, logdir)

def run(cfg, logdir):
    # fix random seed
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_of_iter = 50
    models_names = [
        # 'pn1',
        # 'pn1_4d_basic',
        # 'pn2',
        # 'pn2_4d_basic',
        # '3dmfv',
        # 'set_transformer',
        # 'pst_transformer',
        # 'pstnet',
        # 'p4transformer',
        'pn2_patchlets',
    ]

    for model_name in models_names:
        cfg['MODEL']['pc_model'] = model_name
        # build dataloader and dataset
        test_dataloader, test_dataset = build_dataloader(config=cfg, training=False, shuffle=True)
        num_classes = test_dataset.num_classes

        # build model
        model = build_model(cfg['MODEL'], num_classes, cfg['DATA']['frames_per_clip'])
        model.cuda()
        model = nn.DataParallel(model)
        model.eval()

        model_n_params = count_params(model)
        print(f'{cfg["MODEL"].get("pc_model")} number of parameters: {model_n_params:,}')

        patchlets_timing_dict = {
            'patchlet_extractor': 0,
            'patchlet_temporal_conv': 0,
            'patchlet_classifier': 0,
        }

        time_1 = time.time()

        pbar = tqdm(num_of_iter + 1, desc='testing')
        for batchind, data in enumerate(test_dataloader):
            if batchind == 0:
                continue
            inputs, labels, vid_idx, frame_pad = data['inputs'], data['labels'], data['vid_idx'], data['frame_pad']
            in_channel = cfg['MODEL'].get('in_channel', 3)
            inputs = inputs[:, :, 0:in_channel, :]
            inputs = inputs.cuda().contiguous()
            labels = labels.cuda()

            with torch.no_grad():
                out_dict = model(inputs)
                per_frame_logits = out_dict['pred']

                if out_dict.get('time_dict') is not None:
                    for key, val in out_dict['time_dict'].items():
                        patchlets_timing_dict[key] += np.sum(val)

            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))

            pbar.update()
            if batchind == num_of_iter:
                break

        time_2 = time.time()
        time_per_frame = (time_2 - time_1) / num_of_iter
        print(f'Eval time per frame: {time_per_frame} seconds')

        if out_dict.get('time_dict') is not None:
            for key, val in patchlets_timing_dict.items():

                print(f'{key}: {val / num_of_iter}')

        # save in txt file
        save_lines = f'{cfg["MODEL"].get("pc_model")} number of parameters: {model_n_params:,}\n' \
                     f'Eval time per frame: {time_per_frame} seconds\n' \
                     f'\n'
        with open(os.path.join(logdir, 'results.txt'), 'a') as outfile:
            outfile.write(save_lines)


if __name__ == '__main__':
    main()






