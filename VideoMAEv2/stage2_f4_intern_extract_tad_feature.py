"""Extract features for temporal action detection datasets"""
import argparse
import os
import random
import io

import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms
from torchvision.transforms.functional import resize
import cv2

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader
from intern_stage2_models import *
from torchvision.transforms import InterpolationMode

from tqdm import tqdm

from stage2_config import (Config,
                    eval_dict_leaf)

from stage2_utils import (retrieve_text,
                  _frame_from_video,
                  setup_internvideo2)
from torch.cuda.amp import autocast


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)



import torch.nn.functional as F

def resize_with_torch(frames, size=(224, 224)):
    # frames: torch.Size([batch_size, 3, H, W])
    frames = frames.float()  # uint8 -> float32 변환
    frames = frames / 255.0  # [0, 1] 범위로 정규화
    frames = F.interpolate(frames, size=size, mode='bicubic', align_corners=False)
    return frames



### https://github.com/OpenGVLab/InternVideo/blob/eca2cdc5a67d7442063d19963515b5bd0feef627/InternVideo2/multi_modality/dataset/__init__.py#L133-L154
def resize_with_opencv(frames):
    # frames: torch.Size([8, 3, H, W])
    resized_frames = []
    for frame in frames:
        # Convert to numpy for OpenCV
        frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, C]
        resized_frame = cv2.resize(frame_np, (224, 224), interpolation=cv2.INTER_CUBIC)
        resized_frames.append(torch.from_numpy(resized_frame).permute(2, 0, 1))  # [C, H, W]
    return torch.stack(resized_frames)
    
def get_test_transform_batch(frames):
    # frames: torch.Size([8, 3, H, W])
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Step 1: Resize all frames in the batch
    resized_frames = torch.stack([resize(frame, (224, 224), interpolation=InterpolationMode.BICUBIC) for frame in frames])

    # Step 2: Normalize all frames in the batch
    resized_frames = resized_frames.float().div(255.0)  # Convert to float and normalize to [0, 1]
    resized_frames = transforms.Normalize(mean, std)(resized_frames)

    return resized_frames

def get_test_transform(test_file):

    # loaded images and videos are torch.Tensor of torch.uint8 format,
    # ordered as (T, 1 or 3, H, W) where T=1 for image
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean, std)
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=InterpolationMode.BICUBIC,
            ),
            type_transform,
            normalize,
        ]
    )
    return test_transform(test_file)


####################################################################################



def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--data_set',
        default='THUMOS14',
        choices=['THUMOS14', 'FINEACTION'],
        type=str,
        help='dataset')

    parser.add_argument(
        '--data_path',
        default='/home/jovyan/fileviewer/MMG/VideoEncoder/ucf_avi',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--save_path',
        default='../feature_ucf_subset/intern2_s2',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='InternVideo2_Stage2',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='YOUR_PATH/vit_g_hyrbid_pt_1200e_k710_ft.pth',
        help='load from checkpoint')

    return parser.parse_args()


def get_start_idx_range(data_set):

    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    else:
        raise NotImplementedError()


def internvideo2_range_f8(num_frames):
    return range(0, num_frames - 7, 8)

def internvideo2_range_f4(num_frames):
    return range(0, num_frames - 3, 4)


def extract_feature(args):
    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    video_loader = get_video_loader()


    # get video path
    vid_list = os.listdir(args.data_path)
    random.shuffle(vid_list)


    # get model & load ckpt
    '''
    model = create_model(
        "InternVideo2_Stage2",
        is_pretrain=False,
        num_classes=10,
        num_frames=4,
        tubelet_size=1,
        drop_path_rate=0.3
    )
    '''

    #ckpt = torch.load(args.ckpt_path, map_location='cpu')
    weight_path_8 = "/home/jovyan/fileviewer/MMG/VideoEncoder/models--OpenGVLab--InternVideo2-Stage1-1B-224p-f8/snapshots/feaa042a8a39351fd1120bdcb9dd93c026ac7da1/pretrain.pth"
    weight_path_4 = "/home/jovyan/fileviewer/MMG/VideoEncoder/models--OpenGVLab--InternVideo2-Stage2_1B-224p-f4/snapshots/4362e1f88a992e7edbfd7696f7f78b7f79426dfd/InternVideo2-stage2_1b-224p-f4.pt"



    config = Config.from_file('internvideo2_stage2_config.py')
    config = eval_dict_leaf(config)
    config['pretrained_path'] = weight_path_4
    model = setup_internvideo2(config)


    model.eval()
    model.cuda()
    model = model.half()  # Converts model to torch.float16

    # extract feature
    num_videos = len(vid_list)


    for idx, vid_name in tqdm(enumerate(vid_list), total=len(vid_list), desc="Processing videos"):
        url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        if os.path.exists(url):
            continue
    
        video_path = os.path.join(args.data_path, vid_name)
        vr = video_loader(video_path)
    
        batch_frames = []
        for start_idx in internvideo2_range_f4(len(vr)):
            data = vr.get_batch(np.arange(start_idx, start_idx + 4)).asnumpy()
            frame = torch.from_numpy(data).permute(0, 3, 1, 2)  # [batch_size, 3, H, W]
            batch_frames.append(frame)
    
        # 모든 프레임을 하나의 텐서로 결합
        batch_frames = torch.cat(batch_frames, dim=0)  # [total_frames, 3, H, W]
        
        # 리사이징 및 정규화
        batch_frames = resize_with_torch(batch_frames)  # [total_frames, 3, 224, 224]
        # 정규화(mean, std 적용)
        batch_frames = transforms.Normalize(mean, std)(batch_frames)
        # 배치 차원 추가
        input_data = batch_frames.unsqueeze(0).to('cuda', non_blocking=True) 
        #print("input_data1", input_data.shape)  # [1, total_frames, 3, 224, 224]
        input_data = input_data.permute(0,2,1,3,4) 
        #print("input_data2", input_data.shape) # [batch_size, 3, total_frames, 224, 224]

        grouped_batches = input_data.shape[2] // 4
        input_data = input_data.reshape(grouped_batches, 3, 4, 224, 224)
        #print("input_data3", input_data.shape) # [batch_size, 3, total_frames, 224, 224]

        with torch.no_grad():
            with autocast():
                feature = model.vision_encoder._forward_feature(input_data)
                feature = feature.reshape(-1, feature.shape[-1])  # [total_frames, feature_dim]
    
        # CPU로 이동하여 numpy로 변환
        final_result = feature.cpu().numpy()
    
        # 결과 저장
        print("final_result shape:", final_result.shape)
        np.save(url, final_result)
        print(f'[{idx} / {num_videos}]: save feature on {url}')


if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
