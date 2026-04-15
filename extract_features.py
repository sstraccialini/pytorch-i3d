import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow', default='rgb')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-save_dir', type=str)
parser.add_argument('-frames_per_segment', type=int, default=16, help='Number of frames per contiguous segment')
parser.add_argument('-target_T', type=int, default=2500, help='Pad or truncate to exactly target_T segments')
parser.add_argument('-pad_mode', type=str, default='zero', choices=['zero', 'repeat_last'], help='How to pad if num_segments < target_T')
parser.add_argument('-save_format', type=str, default='npy', choices=['npy', 'npz', 'both'], help='Format to save the features')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

import cv2
import glob

def load_and_transform_chunk(frames_paths, mode, test_transforms):
    frames = []
    for f in frames_paths:
        img = cv2.imread(f)
        if img is None: continue
        if mode == 'rgb':
            img = img[:, :, [2, 1, 0]]
        else: # flow
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)

        w, h = img.shape[:2]
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img, dsize=(0,0), fx=sc, fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
        
    if not frames:
        return None

    imgs = np.asarray(frames, dtype=np.float32)
    if test_transforms:
        imgs = test_transforms(imgs)

    # T x H x W x C -> C x T x H x W
    imgs = torch.from_numpy(imgs.transpose([3,0,1,2]))
    # Add batch dim -> 1 x C x T x H x W
    return imgs.unsqueeze(0)


def run(mode='rgb', root='', load_model='', save_dir='', frames_per_segment=16, target_T=2500, pad_mode='zero', save_format='npy'):
    os.makedirs(save_dir, exist_ok=True)
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])     

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    print(f"[Init] Loading weights from {load_model}")
    state_dict = torch.load(load_model)
    num_classes = state_dict['logits.conv3d.weight'].shape[0]
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(state_dict)
    
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode
    
    videos = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    print(f"[Init] Found {len(videos)} video folders in {root}")

    for step, vid in enumerate(videos):
        print(f"\n[Run] Starting video {vid} ({step+1}/{len(videos)})")
        
        save_path_base = os.path.join(save_dir, vid)
        if save_format in ['npy', 'both'] and os.path.exists(save_path_base+'.npy'):
            print(f"[Run] {vid}.npy already exists. Skipping.")
            continue
        if save_format in ['npz', 'both'] and os.path.exists(save_path_base+'.npz'):
            print(f"[Run] {vid}.npz already exists. Skipping.")
            continue

        vid_dir = os.path.join(root, vid)
        frames_paths = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')) + glob.glob(os.path.join(vid_dir, '*.png')))
        total_frames = len(frames_paths)
        
        if total_frames == 0:
            print(f"[Run] Video {vid} has 0 frames. Skipping.")
            continue
            
        num_segments = total_frames // frames_per_segment
        print(f"[Run] Video {vid}: {total_frames} frames -> {num_segments} segments (size {frames_per_segment})")
        
        if num_segments == 0:
            print(f"[Run] Video {vid} has fewer frames than {frames_per_segment}. Skipping.")
            continue

        segment_features = []
        
        # Disable gradient calculations completely to save GPU memory
        with torch.no_grad():
            for s in range(num_segments):
                start_idx = s * frames_per_segment
                end_idx = start_idx + frames_per_segment
                chunk_paths = frames_paths[start_idx:end_idx]
                
                inputs = load_and_transform_chunk(chunk_paths, mode, test_transforms)
                if inputs is None:
                    continue
                    
                inputs = inputs.cuda()
                
                # I3D output shape: [Batch, 1024, T_out, H_out, W_out]
                feat = i3d.extract_features(inputs)
                
                # MS-Temba expects one vector per segment: [1024]
                # Global average pool explicitly over spatial and temporal dims
                feat = feat.mean(dim=[2, 3, 4]).squeeze(0)  # shape [1024]
                
                # Move to CPU immediately, convert to numpy, to prevent GPU OOM
                segment_features.append(feat.cpu().numpy())

        if len(segment_features) == 0:
            continue
            
        # Stack all segment features -> shape [num_segments, 1024]
        features_raw = np.stack(segment_features, axis=0) 
        num_extracted_segments = features_raw.shape[0]

        # Apply Zero-padding or Repeat-last padding for target_T (e.g., 2500)
        features_padded = features_raw
        if target_T is not None and target_T > 0:
            if num_extracted_segments > target_T:
                # Truncate to target length
                features_padded = features_raw[:target_T, :]
            elif num_extracted_segments < target_T:
                # Pad to target length
                pad_len = target_T - num_extracted_segments
                if pad_mode == 'zero':
                    padding = np.zeros((pad_len, features_raw.shape[1]), dtype=np.float32)
                    features_padded = np.concatenate([features_raw, padding], axis=0)
                elif pad_mode == 'repeat_last':
                    padding = np.repeat(features_raw[-1:], pad_len, axis=0)
                    features_padded = np.concatenate([features_raw, padding], axis=0)

        # Build Metadata dictionary
        metadata = {
            'video_id': vid,
            'num_frames': total_frames,
            'frames_per_segment': frames_per_segment,
            'num_segments_raw': num_extracted_segments,
            'target_T': target_T,
            'feature_dim': features_padded.shape[1]
        }
        
        # Save explicitly depending on the chosen format
        if save_format in ['npy', 'both']:
            np.save(save_path_base + '.npy', features_padded)
        if save_format in ['npz', 'both']:
            # npz allows including raw untouched segment features plus metadata
            np.savez(save_path_base + '.npz', 
                     features=features_padded, 
                     raw_features=features_raw, 
                     **metadata)
                     
        print(f"[Run] Finished {vid}. Extracted {num_extracted_segments} segs. Final saved shape: {features_padded.shape}")

if __name__ == '__main__':
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir,
        frames_per_segment=args.frames_per_segment, target_T=args.target_T, 
        pad_mode=args.pad_mode, save_format=args.save_format)

