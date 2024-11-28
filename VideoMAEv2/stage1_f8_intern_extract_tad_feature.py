"""Extract features for temporal action detection datasets"""
import argparse
import os
import random

import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms
from torchvision.transforms.functional import resize
import cv2

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader
from intern_stage1_models import *
from torchvision.transforms import InterpolationMode

from tqdm import tqdm

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


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
        default='../feature_ucf_subset_all/intern2_s1',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='internvideo2_1B_patch14_224',
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
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=10,
        num_frames=8,
        tubelet_size=1,
        drop_path_rate=0.3
    )

    #ckpt = torch.load(args.ckpt_path, map_location='cpu')
    weight_path_8 = "/home/jovyan/fileviewer/MMG/VideoEncoder/models--OpenGVLab--InternVideo2-Stage1-1B-224p-f8/snapshots/feaa042a8a39351fd1120bdcb9dd93c026ac7da1/pretrain.pth"
    weight_path_4 = "/home/jovyan/fileviewer/MMG/VideoEncoder/models--OpenGVLab--InternVideo2-Stage2_1B-224p-f4/snapshots/4362e1f88a992e7edbfd7696f7f78b7f79426dfd/InternVideo2-stage2_1b-224p-f4.pt"

    ckpt = torch.load(weight_path_8)

    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
            
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.cuda()
    model = model.half()  # Converts model to torch.float16

    # extract feature
    num_videos = len(vid_list)
    '''
    for idx, vid_name in enumerate(vid_list):
        url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        if os.path.exists(url):
            continue

        video_path = os.path.join(args.data_path, vid_name)
        vr = video_loader(video_path)

        feature_list = []
        for start_idx in internvideo2_range_f8(len(vr)):
            data = vr.get_batch(np.arange(start_idx, start_idx + 8)).asnumpy()
            frame = torch.from_numpy(data)  # frame1 torch.Size([8, 240, 320, 3])
            frame = frame.permute(0, 3, 1, 2)  # frame2 torch.Size([8, 3, 240, 320])
            frame_q = torch.stack([get_test_transform(f) for f in frame]) # frame_q torch.Size([8, 3, 224, 224])
            frame_q = frame_q.permute(1,0,2,3) # [3, 8, 224, 224]
            input_data = frame_q.unsqueeze(0).cuda() # input_data torch.Size([1, 3, 8, 224, 224])
            with torch.no_grad():
                feature = model._forward_features(input_data) # [1, 8, 1408]
                feature = feature.squeeze(0)  # Remove batch dimension
                feature_list.append(feature.cpu().numpy())  # Append numpy array to list
        #print(len(feature_list)) # 13
        final_result = np.vstack(feature_list)
        print("final_result", final_result.shape)

        # (frames, channel) -> (8*n, 1408)
        np.save(url, final_result)
        print(f'[{idx} / {num_videos}]: save feature on {url}')
    '''
    for idx, vid_name in tqdm(enumerate(vid_list), total=len(vid_list), desc="Processing videos"):
        url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        if os.path.exists(url):
            continue
    
        video_path = os.path.join(args.data_path, vid_name)
        vr = video_loader(video_path)
    
        feature_list = []
        batch_frames = []  # 배치로 처리할 프레임 저장 리스트
    
        for start_idx in internvideo2_range_f8(len(vr)):
            data = vr.get_batch(np.arange(start_idx, start_idx + 8)).asnumpy()
            frame = torch.from_numpy(data)  # frame1 torch.Size([8, 240, 320, 3])
            ### original ###
            #frame = frame.permute(0, 3, 1, 2)  # frame2 torch.Size([8, 3, 240, 320])
            #frame_q = torch.stack([get_test_transform(f) for f in frame])  # frame_q torch.Size([8, 3, 224, 224])
            #frame_q = frame_q.permute(1, 0, 2, 3)  # [3, 8, 224, 224]
            
            ### batch ###
            #frame = torch.from_numpy(data).permute(0, 3, 1, 2)  # [8, 3, H, W]
            #frame_q = get_test_transform_batch(frame)  # Batch-wise transform
            #frame_q = frame_q.permute(1, 0, 2, 3)  # [3, 8, 224, 224]
            
            ### batch + cv2 ###
            frame = torch.from_numpy(data).permute(0, 3, 1, 2)  # [8, 3, H, W]
            frame_resized = resize_with_opencv(frame)  # OpenCV-based resizing
            frame_q = frame_resized.float().div(255.0)  # Normalize to [0, 1]
            frame_q = transforms.Normalize(mean, std)(frame_q)  # Apply normalization
            frame_q = frame_q.permute(1, 0, 2, 3)  # [3, 8, 224, 224]


            
            batch_frames.append(frame_q)  # 배치 리스트에 추가
    
        # 배치로 처리
        input_data = torch.stack(batch_frames).cuda()  # torch.Size([batch_size, 3, 8, 224, 224])
        #print("input_data shape:", input_data.shape) # input_data shape: torch.Size([B, 3, 8, 224, 224])
    
        with torch.no_grad():
            feature = model._forward_features(input_data)  # torch.Size([batch_size, 8, 1408])
            feature = feature.reshape(-1, feature.shape[-1])  # Flatten across frames: [batch_size * 8, 1408]
            feature_list.append(feature.cpu().numpy())  # Append to feature list
    
        # Concatenate all features
        final_result = np.concatenate(feature_list, axis=0)
        print("final_result shape:", final_result.shape)
    
        # Save features
        np.save(url, final_result)
        print(f'[{idx} / {num_videos}]: save feature on {url}')
    
if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
