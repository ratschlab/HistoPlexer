# HistoPlexer 

## HistoPlexer: Histopathology-based Protein Multiplex Generation using Deep Learning
 
**Abstract:** Multiplexed imaging technologies provide crucial insights into interactions between tumors and their surrounding tumor microenvironment (TME), but their widespread adoption is limited by cost, time, and tissue availability. We introduce HistoPlexer, a deep learning (DL) framework that generates spatially-resolved protein multiplexes directly from histopathology images. HistoPlexer employs the conditional generative adversarial networks with custom loss functions that mitigate slice-to-slice variations and preserve spatial protein correlations. In a comprehensive evaluation on metastatic melanoma samples, HistoPlexer consistently outperforms existing approaches, achieving superior Multiscale Structural Similarity Index and Peak Signal-to-Noise Ratio. Qualitative evaluation by domain experts demonstrates that the generated protein multiplexes closely resemble the real ones, evidenced by Human Eye Perceptual Evaluation error rates exceeding the 50\% threshold for perceived realism. Importantly, HistoPlexer preserves crucial biological relationships, accurately capturing spatial co-localization patterns among proteins. In addition, the spatial distribution of cell types derived from HistoPlexer-generated protein multiplex enables effective stratification of tumors into immune hot versus cold subtypes. When applied to an independent cohort, incorporating additional features from HistoPlexer-generated multiplexes enhances the performance of the DL model for survival prediction and immune subtyping, outperforming the model reliant solely on Hematoxylin & Eosin (H&E) image features. By enabling the generation of whole-slide protein multiplex from the H&E image, HistoPlexer offers a cost- and time-effective approach to understanding the TME, and holds promise for advancing precision oncology.

## Installation
First clone the repo and cd into the directory:
```shell
git clone https://github.com/ratschlab/HistoPlexer.git
cd HistoPlexer
```
Then create a conda env and install the dependencies:
```shell
conda env create -f environment.yml
conda activate histoplexer
```
## Running the code 
### Training 
### Inference 
### Downstream tasks 

<!-- Details about our model architecture and implementation can be found in our [preprint](). If you use this code, please consider citing our work: -->


## Downloading Ultivue dataset 
Manuscript under review. Available upon acceptance.
