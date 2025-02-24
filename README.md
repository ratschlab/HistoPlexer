# Training and Testing Instructions

## Training
To train the model, use the following command:

```bash
python train.py --dataroot path/to/parent/folder/of/he/and/imc \
               --name cyclegan \
               --model cycle_gan \
               --netG [resnet_9blocks | resnet_6blocks | unet_256 | unet_128] \
               --dataset_mode aligned \
               --lambda_identity 0 \
               --n_epochs <number_of_epochs_with_initial_learning_rate> \
               --n_epochs_decay <number_of_epochs_to_decay_learning_rate_to_zero> \
               --gpu_ids 0
               --output_nc 11
               --seed 0
```

### Explanation:
- `--dataroot`: Path to the parent folder containing HE and IMC datasets.
- `--name`: Name of the experiment.
- `--model`: Specifies the model type (cycle_gan in this case).
- `--netG`: Specifies the generator network architecture. Options include:
  - `resnet_9blocks`
  - `resnet_6blocks`
  - `unet_256`
  - `unet_128`
- `--dataset_mode`: Defines the dataset structure (aligned in this case).
- `--lambda_identity`: Identity loss weight (set to 0 in this case).
- `--n_epochs`: Number of epochs with the initial learning rate.
- `--n_epochs_decay`: Number of epochs to decay the learning rate linearly to zero.
- `--gpu_ids`: Specifies which GPU(s) to use.
- `--output_nc`: Output channel dimension.
- `--channel`: For singleplex add this arg. E.g., 0 for the first channel of the IMC image.
- `--seed`: Random seed.

### Input Requirements:
- The model expects **H&E images** to be **1024x1024**.
- The model expects **IMC images** to be **256x256**.

## Testing
To test the trained model, use the following command:

```bash
python test.py --dataroot path/to/he/folder \
              --name cyclegan \
              --model test \
              --netG [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
              --output_nc 11
```

### Explanation:
- `--dataroot`: Path to the HE dataset folder.
- `--name`: Name of the experiment (must match the training experiment name).
- `--model`: Specifies that testing is being performed.
- `--netG`: Specifies the generator architecture (must match the architecture used during training).
- `--output_nc`: Output channel dimension.

### Testing Process:
- The model expects **H&E images** to be **1024x1024** for testing.
- Since large **H&E ROIs** may be bigger than the model input, you need to **patch** the H&E ROI into **1024x1024** segments.
- Perform prediction separately on each patch.
- **Stitch back** the predicted patches to reconstruct the full ROI.


