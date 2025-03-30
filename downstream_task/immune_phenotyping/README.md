# Immune phenotyping

## Immune Phenotyping analysis
Immune phenotyping involves predicting the cell types in whole slide images (WSIs) using trained random forest. This process uses nuclei coordinates extracted from HE files and predictions from trained models to classify cells.

```bash
python -m bin.immune_phenotyping 
```
```
arguments:
    --annotation_path     Path to tumor and invasive margin annotations on HE WSI from Halo
    --tissue_regions_path Path to TIF files from Halo with tissue regions
    --wsi_pred_path       Path to WSI predictions from the HistoPlexer model
    --hovernet_path       Path to HoVer-Net output for HE WSI for nuclei coordinates
    --rf_base_path        Path to random forest model used for cell typing
    --meta_path           Metadata file for samples containing ground truth immune phenotype
    --save_annotation_path Path to save tumor IM annotations using Halo annotations

optional arguments:
    --radius              Radius of the circle to be drawn around the XY cell coordinates for cell typing of pseudo cells
    --level               which level to use from wsi predictions
    --cell_types          Which cell types to use for cell typing
    --resolution_he       Resolution of HE image in um/px
```

## Visualize Cell Typing
This step visualizes the predicted cell types on HE slides. It overlays annotations on HE images to show the spatial distribution of cell types.

```bash
python -m bin.visualize_cell_typing 
```
```
arguments:
    --wsi_celltyping_path Path to where cell typing for immune phenotyping are saved from previous step
    --meta_path           Metadata file for samples containing ground truth immune phenotype
    --saved_annotation_path Path to saved tumor IM annotations using Halo annotations, saved in previous step
    --immune_gt_path      Path to excel files with CD8 quantification
    --he_basepath         Path to where HE wsi files are saved
    --annotation_path     Path to tumor and invasive margin annotations on HE WSI from Halo
    --save_path           Path where HE images with cell typing results are saved
```

## Immune phenotyping stratification plots
Stratify the data into different immune subtypes. This step generates box plots for various stratified groups. Also performs classificaiton using iCD8 and sCD8 densities as features 

```bash
python -m bin.stratify_code 
```
```
arguments:
    --saved_path_analysis Path to where cell typing for immune phenotyping are saved
    --meta_path           Metadata file for samples containing ground truth immune phenotype
    --immune_gt_path      Path to excel files with CD8 quantification
    --gt_labels_path      Path to excel files with gt immune subtype labels
    --annotation_meta_path  Metadata file for HALO annotations, eg quality of annotations
    --save_path           Path where box plots and results are saved
```

## HTA Score
Calculate the HTA heterogeneity score for the given dataset. This score quantifies the heterogeneity of immune cell types in the tissue.

```bash
python -m bin.hta_score 
```
```
arguments:
    --meta_path           Metadata file for samples containing ground truth immune phenotype
    --wsi_celltyping_path Path to where cell typing for immune phenotyping are saved from immune phenotyping analysis
    --saved_annotation_path Path to saved tumor IM annotations using Halo annotations, saved in immune phenotyping analysis

```