import torch 
import os 
import json
import glob 
from torch.utils.data import DataLoader
import timm
from src.dataset.dataset import UniDataset
import h5py
import argparse
import tqdm

# python -m bin.uni_embeddings --patches_dir=/raid/sonali/project_mvs/data/tupro/binary_he_rois_test --device=cuda:0
# parser
parser = argparse.ArgumentParser(description="Configurations for getting features embeddings from uni model")
parser.add_argument("--weights_path", type=str, required=False, default='/home/sonali/raid_st/foundation_models/uni_v1/pytorch_model.bin',help="Path to uni checkpoint file")
parser.add_argument("--patches_dir", type=str, required=False, default='/raid/sonali/project_mvs/data/tupro/patches/binary_he_patchs', help="Path to he numpy files")
parser.add_argument("--device", type=str,required=False, default='cuda:0', help="device used for running the model")

args = parser.parse_args()
    

device = args.device
patches_dir = args.patches_dir
base_path = os.path.dirname(patches_dir)
embed_path = os.path.join(base_path, 'embeddings_'+ os.path.basename(patches_dir) +'.h5')
print(embed_path)

# loading uni models
weights_path = args.weights_path
model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
model.to(device)
model.eval()
print("Model loaded")

# get all npy files and pass to dataloader
patches_paths = glob.glob(patches_dir + '/*.npy')
print(len(patches_paths), patches_paths[0])
uni_dataset = UniDataset(patches_paths)

# dataloader
uni_loader = DataLoader(uni_dataset,
                        batch_size=128, 
                        shuffle=False,
                        pin_memory=True, 
                        num_workers=4, 
                        drop_last=False)

print("Dataloader loaded")

with h5py.File(embed_path, "w") as h5_dict:
    for batch in tqdm.tqdm(uni_loader):
    # for batch in uni_loader:
        sample_names = batch['sample']   
        img = batch['img'].to(device)
        with torch.no_grad():
            embeddings = model(img)
            batch_data = {sample_names[i]: embeddings[i].cpu().numpy() for i in range(len(sample_names))}
            h5_dict.update(batch_data)  # Write batch in one go
            
# # example on how to read embeddings         
# h5_dict = h5py.File('embeddings_binary_he_rois_test.h5', "r") 
# feature= h5_dict['MYNELIC_F3'][:]  # pass sample name
# print(feature.shape) 