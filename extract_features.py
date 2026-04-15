import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transforms=None, save_dir=''):
        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.save_dir = save_dir
        self.videos = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid = self.videos[idx]
        print(f"[Dataset] Processing video index {idx}: {vid}")
        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            print(f"[Dataset] Video {vid} already extracted, returning dummy data")
            # return dummy data if already extracted
            return torch.zeros(1), torch.zeros(1), vid

        vid_dir = os.path.join(self.root, vid)
        frames_paths = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')) + glob.glob(os.path.join(vid_dir, '*.png')))
        frames = []

        print(f"[Dataset] Video {vid} has {len(frames_paths)} frames. Starting to load...")
        for i, f in enumerate(frames_paths):
            if i % 1000 == 0:
                print(f"[Dataset] Video {vid} loading frame {i}/{len(frames_paths)}")
            img = cv2.imread(f)
            if img is None: continue
            if self.mode == 'rgb':
                img = img[:, :, [2, 1, 0]]
            else: # flow handling (assuming x and y are channels or greyscale)  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=2)

            w, h = img.shape[:2]
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img, dsize=(0,0), fx=sc, fy=sc)
            img = (img/255.)*2 - 1
            frames.append(img)

        print(f"[Dataset] Video {vid} all frames loaded. Converting to numpy array...")
        imgs = np.asarray(frames, dtype=np.float32)
        print(f"[Dataset] Video {vid} numpy array shape: {imgs.shape} and memory size: {imgs.nbytes / (1024**2):.2f} MB")
        
        if self.transforms:
            print(f"[Dataset] Video {vid} applying transforms...")
            imgs = self.transforms(imgs)

        print(f"[Dataset] Video {vid} transposing and converting to PyTorch tensor...")
        # T x H x W x C -> C x T x H x W
        imgs = torch.from_numpy(imgs.transpose([3,0,1,2]))

        print(f"[Dataset] Video {vid} tensor created with shape {imgs.shape}. Returning from __getitem__.")
def run(max_steps=64e3, mode='rgb', root='/ssd2/charades/Charades_v1_rgb', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])     

    dataset = CustomDataset(root, mode, test_transforms, save_dir=save_dir)
    # Set num_workers=0 to prevent OOM errors from multiple workers loading large frame tensors into RAM
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    dataloaders = {'val': dataloader}
    datasets = {'val': dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
        
    state_dict = torch.load(load_model)
    num_classes = state_dict['logits.conv3d.weight'].shape[0]
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(state_dict)
    
    i3d.cuda()

    for phase in ['val']:
        i3d.train(False)  # Set model to evaluate mode

        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        for step, data in enumerate(dataloaders[phase]):
            print(f"\n[Run] Starting step {step}")
            # get the inputs
            inputs, labels, name = data
            print(f"[Run] Received batch for video: {name}, inputs shape: {inputs.shape}")
            if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
                print(f"[Run] {name[0]} already extracted, skipping forward pass.")
                continue

            b,c,t,h,w = inputs.shape
            print(f"[Run] Passing inputs to CUDA, batch size={b}, channels={c}, time={t}")
            if t > 1600:
                print(f"[Run] Sequence too long (t={t} > 1600). Processing in chunks.")
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    print(f"[Run] Processing chunk start={start}, end={end}")
                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                    print(f"[Run] Feature extraction for chunk...")
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                print(f"[Run] Concatenating and saving array to {save_dir}")
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                print(f"[Run] Sequence short enough for direct processing.")
                # wrap them in Variable
                inputs = Variable(inputs.cuda(), volatile=True)
                print(f"[Run] Feature extraction for sequence...")
                features = i3d.extract_features(inputs)
                print(f"[Run] Saving array to {save_dir}")
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())
            
            print(f"[Run] Finished step {step}\n")


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
