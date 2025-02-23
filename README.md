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

## Testing
To test the trained model, use the following command:

```bash
python test.py --dataroot path/to/he/folder \
              --name cyclegan \
              --model test \
              --netG [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
```

### Explanation:
- `--dataroot`: Path to the HE dataset folder.
- `--name`: Name of the experiment (must match the training experiment name).
- `--model`: Specifies that testing is being performed.
- `--netG`: Specifies the generator architecture (must match the architecture used during training).
