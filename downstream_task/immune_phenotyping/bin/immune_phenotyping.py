import argparse
from src.immune_phenotyping_utils import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Configurations for getting immune phenotyping")
    parser.add_argument("--annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/regions_annotations', help="Path to tumor and invasive margin annotations on HE wsi")
    parser.add_argument("--tissue_regions_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/classifier_mask', help="Path to tif files from halo with tissue regions")
    parser.add_argument("--wsi_pred_path", type=str, required=False, default='/raid/sonali/project_mvs/nmi_results/ours/tupro_ours_channels-all_seed-3/all-tupro_wsis/step_495000', help="Path to wsi predictions from the model")
    parser.add_argument("--level", type=int, required=False, default=4, help="which level to use from wsi predictions")
    parser.add_argument("--hovernet_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/hovernet_output', help="Path to hovernet output for HE wsi")
    parser.add_argument("--rf_base_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/cell_typing/rf/split3/', help="Path to random forest model for cell typing")
    parser.add_argument("--cell_types", type=str, required=False, default='tumor_CD8', help="Which cell types to use for cell typing: tumor_CD8, tumor_CD8_CD4, tumor_CD8_CD4_CD20, all")
    parser.add_argument("--radius", type=int, required=False, default=5, help="radius of the cell in um")
    parser.add_argument("--resolution_he", type=float, required=False, default=0.23, help="resoluton of HE image in um/px")
    parser.add_argument("--meta_path", type=str, required=False, default='/raid/sonali/project_mvs/meta/tupro/melanoma-merged_clinical_data-v8.tsv', help="metadata file for samples containing ground truth immune phenotype")
    parser.add_argument("--save_annotation_path", type=str, required=False, default='/raid/sonali/project_mvs/downstream_tasks/immune_phenotyping/tupro/halo_annotations/tumor_IM_annotations', help="Path to saved tumor IM annotations using halo annotations")
    args = parser.parse_args()

    process_and_save_annotations(args.tissue_regions_path, args.annotation_path, args.save_annotation_path)
    perform_wsi_cell_typing(args)