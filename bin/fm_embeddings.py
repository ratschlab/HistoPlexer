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

# python -m bin.fm_embeddings --fm_model=uni_v1 --img_size=864 --patches_dir=/raid/sonali/project_mvs/data/tupro/he_rois_test/binary_he_rois_test # for test rois
# python -m bin.fm_embeddings --fm_model=uni_v2 --img_size=868 --patches_dir=/raid/sonali/project_mvs/data/tupro/he_rois_test/binary_he_rois_test
# python -m bin.fm_embeddings --fm_model=uni_v1  # for train patches 

# parser
parser = argparse.ArgumentParser(description="Configurations for getting features embeddings from uni model")
parser.add_argument("--fm_model", type=str, required=False, default='uni_v1',help="Which foundation model to use uni_v1, uni_v2 or virchow_v2")
parser.add_argument("--weights_path", type=str, required=False, default='/home/sonali/raid_st/foundation_models/',help="Path to all foundation models")
parser.add_argument("--patches_dir", type=str, required=False, default='/raid/sonali/project_mvs/data/tupro/patches/binary_he_patchs', help="Path to he numpy files")
parser.add_argument("--device", type=str, required=False, default='cuda:0', help="device used for running the model")
parser.add_argument("--img_size", type=int, required=False, default=224, help="the image size used for the model")

args = parser.parse_args()
    
device = args.device
patches_dir = args.patches_dir
base_path = os.path.dirname(patches_dir)
embed_path = os.path.join(base_path, 'embeddings-' + args.fm_model +'.h5')
print(embed_path)

# loading uni models
weights_path = os.path.join(args.weights_path, args.fm_model, 'pytorch_model.bin')
assert os.path.exists(weights_path), f"Path to model weights does not exist: {weights_path}"

if args.fm_model == 'uni_v1':
    timm_kwargs = {        
        'model_name': 'vit_large_patch16_224', 'img_size': 224, 'patch_size': 16,
        'init_values': 1e-5, 'num_classes': 0, 'dynamic_img_size': True
        }
elif args.fm_model == 'uni_v2':
    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224', 'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2, 'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 'reg_tokens': 8, 'dynamic_img_size': True
        }
elif args.fm_model == 'virchow_v2':
    from timm.layers import SwiGLUPacked
    timm_kwargs = {
        'model_name': 'vit_huge_patch14_224', 'img_size': 224,
        'init_values': 1e-5, 'mlp_ratio': 5.3375, 'num_classes': 0,
        'mlp_layer': SwiGLUPacked, 'act_layer': torch.nn.SiLU, 'reg_tokens': 4, 'dynamic_img_size': True
        }
    
model = timm.create_model(pretrained=False, **timm_kwargs)
model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
model.to(device)
model.eval()
print("Model loaded")

# get all npy files and pass to dataloader
patches_paths = glob.glob(patches_dir + '/*.npy')
print(len(patches_paths), patches_paths[0])
uni_dataset = UniDataset(patches_paths, img_size=args.img_size)

# dataloader
uni_loader = DataLoader(uni_dataset,
                        batch_size=1, 
                        shuffle=False,
                        pin_memory=True, 
                        num_workers=8, 
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