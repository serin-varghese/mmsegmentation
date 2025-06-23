# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
from time import sleep

from mmseg.apis import inference_model, init_model, show_result_pyplot


def parse_args():
    parser = argparse.ArgumentParser(description='Inference from checkpoint')
    parser.add_argument(
        '--config_path',
        type=str,
        default=
        '/home/varghese/STVFormer/Code/STVFormer/configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py',
        help='Path to the model config file',
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=
        '/home/varghese/STVFormer/Code/STVFormer/tools/inference/hrnet/checkpoint/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth',
        help='Path to the model checkpoint file',
    )
    parser.add_argument(
        '--img_dir',
        type=str,
        default=
        '/home/varghese/STVFormer/Code/STVFormer/tools/inference/data/stuttgart_00/**/*.png',
        help='Path to the directory containing images',
    )
    return parser.parse_args()


def create_batches(img_list, batch_size):
    """
    Splits a list of image paths into smaller batches.

    Args:
        img_list (list): A list of image file paths.
        batch_size (int): The number of images in each batch.

    Returns:
        list: A list of batches, where each batch is a list of image file paths.
    """
    return [
        img_list[i:i + batch_size] for i in range(0, len(img_list), batch_size)
    ]


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    img_list = glob.glob(args.img_dir, recursive=True)
    print(f'Config path: {config_path}')
    print(f'Checkpoint path: {checkpoint_path}')
    print(f'Image directory: {args.img_dir}')
    sleep(2)  # Sleep for 2 seconds to allow the user to read the output

    print(f'Found {len(img_list)} images in {args.img_dir}')
    print(f'Config path: {config_path}')
    print(f'Checkpoint path: {checkpoint_path}')

    img_list = create_batches(img_list, 4)
    model = init_model(config_path, checkpoint_path, device='cuda:0')
    for img_batch in img_list:
        results = inference_model(model, img_batch,
                                  'HRNet_Pseudo_Labels_SegMasks')
        for i, result in enumerate(results):
            show_result_pyplot(
                model,
                img_batch[i],
                result,
                save_dir='test',
                out_file=
                f'HRNet_Pseudo_Labels_ImageResults/result_{os.path.basename(img_batch[i])}',
                show=False,
            )
