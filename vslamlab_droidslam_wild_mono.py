import sys

from pathlib import Path

import numpy as np
import yaml
import torch
import argparse
import os

from src import config
from src.slam import SLAM
from src.utils.datasets import get_dataset
from time import gmtime, strftime
from colorama import Fore,Style
from torch.utils.tensorboard import SummaryWriter

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    print("\nRunning vslamlab_droidslam_wild_mono.py ...\n")

    parser = argparse.ArgumentParser()

    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None)
    parser.add_argument("--verbose", type=str, help="verbose")

    args, _ = parser.parse_known_args()

    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config("configs/Dynamic/VSLAM-LAB/vslamlab_mono.yaml")
    setup_seed(cfg['setup_seed'])
    if cfg['fast_mode']:
        # Force the final refine iterations to be 3000 if in fast mode
        cfg['mapping']['final_refine_iters'] = 3000


    ##########################################################################
    # VSLAM-LAB Integration
    cfg['droidvis'] = bool(args.verbose)
    cfg['sequence_path'] = args.sequence_path
    cfg['rgb_csv'] = args.rgb_csv
    cfg['calibration_yaml'] = args.calibration_yaml
    cfg['exp_folder'] = args.exp_folder
    cfg['exp_it'] = args.exp_it

    cfg['cam_name'] = "rgb_0"

    calibration_yaml = cfg['calibration_yaml']
    cam_name = cfg['cam_name']

    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == cam_name:
            cam = cam_;
            break;

    print(f"\nCamera Name: {cam['cam_name']}")
    print(f"Camera Type: {cam['cam_type']}")
    print(f"Camera Model: {cam['cam_model']}")
    print(f"Focal Length: {cam['focal_length']}")
    print(f"Principal Point: {cam['principal_point']}")
    has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
    if has_dist:
        print(f"Distortion Type Dimension: {cam['distortion_type']}")
        print(f"Distortion Coefficients: {cam['distortion_coefficients']}")
    print(f"Image Dimension: {cam['image_dimension']}")
    print(f"Fps: {cam['fps']}")

    cfg['cam']['fx'], cfg['cam']['fy'] = cam['focal_length']
    cfg['cam']['cx'], cfg['cam']['cy'] = cam['principal_point']
    cfg['cam']['H'], cfg['cam']['W'] = cam['image_dimension']

    if has_dist:
        cfg['cam']['distortion'] = np.array(cam['distortion_coefficients'], dtype=np.float32)
    ##########################################################################

    output_dir = cfg['data']['output']
    output_dir = output_dir+f"/{cfg['scene']}"

    # clean the rerun_stream.rrd
    if os.path.exists(f"{output_dir}/rerun_stream.rrd"):
        os.remove(f"{output_dir}/rerun_stream.rrd")

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = "-"*30+Fore.YELLOW+\
                 f"\nStart WildGS-SLAM at {start_time},\n"+Style.RESET_ALL+ \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   output: {output_dir}\n"+ \
                 "-"*30
    print(start_info)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    dataset = get_dataset(cfg)

    slam = SLAM(cfg,dataset)
    slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("-"*30+Fore.LIGHTRED_EX+f"\nWildGS-SLAM finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)

