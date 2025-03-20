# Coexpression of protein markers 

## pseudo cells
The first step is to get average expression per cell after model inference is done on data. For this, it is expected to already have the nuclei coordinates extracted from HE files using HoVer-Net 
```bash
python -m bin.pseudo_cell --input_path=<Path-to-predictions-from-the-model> --save_path=<Path-where-avg-expression-per-cell-will-be-saved> --he_nuclei_path=<Path-to-csv-file-with-nuclei-coordinates-from-hovernet>
```
```
optional arguments:
    --radius              Radius of the circle to be drawn around the XY cell coordinates
    --markers_list        List of markers for which model was trained
    --level               Depends on the dataset and diff in resolution between HE and IMC. 0 if same resolution. 2 for tupro data as IMC 2**2 times smaller than HE 
    --data_set             Which set from split to use {test, train}. Used to define save_path if not passed
    --overwrite           pass if overwrite existing output files  
    --experiment_type, 
    --experiment_name     both used to find path to predictions from the model if input_path is not passed
    --base_path           Used to define input_path and save_path if not passed
```

## co-expression/co-localisation patterns
To obtain the mean square error between spearman correlation coefficient of ground truth and predicted IMC, we use:

```bash
python -m bin.coexpression_markers --save_path=<Path-where-mse-will-be-saved> --gt_scdata_path=<Path-to-ground-truth-avg-expression-per-cell>
```
```
optional arguments:
    --which_pairs         Which protein pairs to plot. all, or pairs with positive_corr or negative_corr.
    --level               Depends on the dataset and diff in resolution between HE and IMC. 0 if same resolution. 2 for tupro data as IMC 2**2 times smaller than HE 
    --data_set            Which set from split to use {test, train}. Used to define save_path if not passed
    --plot                pass if want to get the plot of spearman correlation coefficient for protein pairs. 
    --co_exp_setting      Use pruned or all marker pairs. By default pairs with small positive or negative correlation are excluded. 
    --base_path           Path to the experiments director
```
Note: In the script, adapt `keywords` to find all experiments for different baselines. 

## co-localisation patterns t-sne
This is used obtain and compare the t-sne for ground truth and predicted IMC data 

```bash
python -m bin.coexpression_tsne --scale_to01_forplot --gt_scdata_path=<Path-to-ground-truth-avg-expression-per-cell> --pred_scdata_path=<Path-to-predicted-avg-expression-per-cell> --save_path=<Path-where-tsne-results-will-be-saved> 
```
```
optional arguments:
    --data_set            Which set from split to use {test, train}. Used to define save_path if not passed
    --seed                Random seed for obtaining tsne
    --scale_to01          Scale to [0,1] before tSNE embedding.
    --scale_to01_forplot  Scale to [0,1] for plotting only.
```
Note: In the script, adapt `protein_sets` for plotting as needed. 
