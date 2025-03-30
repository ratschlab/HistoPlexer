# Attention-based MIL

## Data

The data consists of:
1. An `.h5` file which has two key words: `coord` and `features`. The coordinates and features are stored in `.npy` format.
2. The metadata for survival analysis containing the labels is stored in `.csv` format on Google drive ([here](https://drive.google.com/drive/u/0/folders/1kWC8yDeMnkXyi3xIrkmyRbzKl3emD9dd)). See [here]( https://colab.research.google.com/drive/11BmFojURIRU4Qp1uNix_6TmAaSq_JOKP?usp=drive_link) how it is generated. Here is an example of how the metadata looks like:
```
|   | slide_id     | disc_label | case_id     | sample_type   | pat_label | gender | age_at_initial_pathologic_diagnosis | censorship | survival          |
|---|--------------|------------|-------------|---------------|-----------|--------|-------------------------------------|------------|-------------------|
| 0 | TCGA-3N-A9WB | 0.0        | TCGA-3N-A9WB| Metastatic    | 0         | MALE   | 71.0                                | 0.0        | 17.018480492813143|
| 1 | TCGA-3N-A9WC | 2.0        | TCGA-3N-A9WC| Metastatic    | 5         | MALE   | 82.0                                | 1.0        | 66.43121149897331 |
| 2 | TCGA-3N-A9WD | 0.0        | TCGA-3N-A9WD| Metastatic    | 0         | MALE   | 82.0                                | 0.0        | 12.97741273100616 |
| 3 | TCGA-BF-A1PU | 0.0        | TCGA-BF-A1PU| Primary Tumor | 1         | FEMALE | 46.0                                | 1.0        | 12.714579055441478|
| 4 | TCGA-BF-A1PV | 0.0        | TCGA-BF-A1PV| Primary Tumor | 1         | FEMALE | 74.0                                | 1.0        | 0.459958932238193 |

```
3. The splits are stored in `.csv` format on Google drive ([here](https://drive.google.com/drive/u/0/folders/1kWC8yDeMnkXyi3xIrkmyRbzKl3emD9dd)), which has three columns: `train`, `val`, `test`; each column contains the `slide_id` of all the train/val/test slides. You can also use [here]( https://colab.research.google.com/drive/11BmFojURIRU4Qp1uNix_6TmAaSq_JOKP?usp=drive_link) notebook to generate your own split.

## Extracting features for HE and IMC pred modalities for multimodal experiments 
For extracting features fom HE TCGA WSIs, use
```bash
python -m bin.he_feats --input_slide=<path-to-he-wsi> --checkpoint=<path-to-resnet-or-mia-model> --output_dir=<path-to-save-outputs> --tile_size=<tile-size> --out_size=<resize-tile-to-size> --device=<device-used>
```

## Interactive Job or Customizing Submitting Script

For interactive jobs or to customize the submission script, use:

```bash
python -m bin.train --config_path src/config/sample_config.json
```
for running the training script.

## Automatic Job Submission with Grid Search Hyperparameters

To submit several jobs automatically using grid search hyperparameters, use:

```bash
python -m bin.run_config_grid --config_path src/config/sample_config_grid.json
```

For the grid search, pass the hyperparameters as a list in the config dictionary in the JSON file. For example, to perform a grid search on the discriminator's learning rate, include `{"lr_D": [1e-3, 1e-4, 1e-5]}` in the config. This will generate three different `config.json` files and corresponding `run.sh` scripts, saved in three different folders under `/path/to/results`.

To modify the job submitting script, see [here](https://github.com/ratschlab/HistoPlexer/blob/main/downstream_task/tcga/src/config/config_grid.py#L25-L33).

## Configuration Settings

The project uses a configuration dictionary to set various parameters. Here's some args you might need to modify:

### Paths
- `save_path`: Directory for saving results. 
- `data_path`: Directory containing h5 data.
- `csv_path`: Directory to the `.csv` file containing the meta data of all labels.
- `split_path`: Directory to the `.csv` file of the split to use. 

### Dataset
- `label_col`: Which column to use as labels. Default: `"3_year_survival"`.
- `is_weighted_sampler`: If use weighted sampler to account for class imbalance. Default: `True`.

### Model
- `in_feat_dim`: Input feature dim. Default: `512`.
- `drop_out`: Drop-out prob. Default: None (i.e., no drop-out).