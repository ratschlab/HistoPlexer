import os
from pathlib import Path

# - constants: Important collection of constants, including sample lists, path constants, value constants, name maps etc. Make sure to read this !


# define constants:
M41_SAMPLES = ['MACEGEJ', 'MACOLUD', 'MADAJEJ', 'MADUFEM', 'MAJEFEV', 'MAKYGIW', 'MECUGUH', 'MECYGYR', 'MEFOCUR', 'MEGEGUJ',
'MEHUFEF', 'MELUKUZ', 'MICEGOK', 'MIHIFIB', 'MOBICUN', 'MOCELOJ', 'MODOHOX', 'MOJYMOC', 'MUBIBIT', 'MUDEFAW', 'MUGAHEK',
'MUGAKOL', 'MULELEZ', 'MUMIFAD', 'MYHAJIS', 'MYJUFAJ', 'MYKYPAZ', 'MYLURAZ']

def get_m41_samples():
    return M41_SAMPLES

MELANOMA_SAMPLES = ['MEHUFEF', 'MYDACIM', 'MIDOJYB', 'MELYPEB', 'MEMEMUH', 'MIJULED', 'MEDUCIN', 'MORAXOD', 'MECADAC', 'MUFACIK',
'MODUDOL', 'MAHOBAM', 'MEKAKYD', 'MULYMUP', 'MYBYHER', 'MYJILAS', 'MABABAB', 'MAJEFEV', 'MEPELEX', 'MODIGOS', 'MOLOLYB', 'MIKYCYN',
'MUKAGOX', 'MIMANAR', 'MEKOBAB', 'MADEGOD', 'MUGAKOL', 'MOQAVIJ', 'MIDEKOG', 'MICEGOK', 'MIBUBIR', 'MAPOXUB', 'MATIWAQ', 'MEBIGAL',
'MOTAMUH', 'MADIBUG', 'MYLURAZ', 'MELIPIT', 'MADUBIP', 'MEHYLOB', 'MYNESYB', 'MUHYBAF', 'MIPYNAP', 'MACOLUD', 'MUCADOP', 'MACEGEJ',
'MEDEFUC', 'MIBAFUK', 'MIGEDEK', 'MYHAJIS', 'MOGYHOJ', 'MODOHOX', 'MOBICUN', 'MOBYLUD', 'MOGYLIP', 'MECUGUH', 'MUDUKEF', 'MIGEKUT',
'MIFOGIL', 'MOCELOJ', 'MIKOBID', 'MUDEFAW', 'MUBYJOF', 'MIGOFIW', 'MYGIFUD', 'MODALEG', 'MIJYDYP', 'MYJUFAJ', 'MEWORAT', 'MORUWUP',
'MOBUBOT', 'MILABYL', 'MOPYPUS', 'MIMUVYF', 'MISYPUP', 'MUFOFOP', 'MEDYCAR', 'MIFIMAB', 'MAKYGIW', 'MOCUKYG', 'MEHEHUM', 'MOMUSIG',
'MEGEGUJ', 'MEQUQEG', 'MUMIFAD', 'MECYGYR', 'MAROMIW', 'MUGAHEK', 'MADUFEM', 'MAHACEB', 'MALYLEJ', 'MOROPEX', 'MEXUXEH', 'MADAJEJ',
'MYKOKIG', 'MANOFYB', 'MAJOFIJ', 'MEFYFAK', 'MUDIFOB', 'MEVIXYV', 'MYNELIC', 'MUBOMEF', 'MULELEZ', 'MUFYDUM', 'MAHEFOG', 'MYBYFUW',
'MEFOCUR', 'MYKYPAZ', 'MOVAZYQ', 'MOJYMOC', 'MELUKUZ', 'MIHIFIB', 'MEMIGOG', 'MAFIBAF', 'MUBIBIT', 'MEZYWEG',
'MOFYCAG', 'MODAJOH']

def get_melanoma_samples():
    return MELANOMA_SAMPLES

BATCH_CORRECTED_SAMPLES = ['M3UI6', 'M8WRI', 'MACEGEJ', 'MACOLUD', 'MACYHYS', 'MADAJEJ', 'MADEGOD', 'MADIBUG', 'MADUBIP', 'MADUFEM',
'MAFIBAF', 'MAHEFOG', 'MAHOBAM', 'MAJEFEV', 'MAJOFIJ', 'MAKYGIW', 'MALYLEJ', 'MANOFYB', 'MEBIGAL', 'MECADAC', 'MECUGUH', 'MECYGYR',
'MEDEFUC', 'MEDUCIN', 'MEDYCAR', 'MEFOCUR', 'MEFYFAK', 'MEGEGUJ', 'MEHEHUM', 'MEHUFEF', 'MEHYLOB', 'MEKAKYD', 'MEKOBAB', 'MELIPIT',
'MELUKUZ', 'MELYPEB', 'MEMEMUH', 'MEMIGOG', 'MENEMAN', 'MEVIXYV', 'MEZYWEG', 'MIBAFUK', 'MIBIBUJ', 'MIBUBIR', 'MICEGOK', 'MIDEKOG',
'MIDOJYB', 'MIFIMAB', 'MIFOGIL', 'MIGEKUT', 'MIHIFIB', 'MIJYDYP', 'MIKOBID', 'MILABYL', 'MIMANAR', 'MIMOHIS', 'MIMUVYF', 'MIPYNAP',
'MISYPUP', 'MOBICUN', 'MOBUBOT', 'MOBYLUD', 'MOCEGIB', 'MOCELOJ', 'MODAJOH', 'MODALEG', 'MODIGOS', 'MODOHOX', 'MOGYHOJ', 'MOGYLIP',
'MOJYMOC', 'MOMUSIG', 'MOPYPUS', 'MOQAVIJ', 'MOTAMUH', 'MOVAZYQ', 'MUBIBIT', 'MUBOMEF', 'MUBYJOF', 'MUCADOP', 'MUDEFAW', 'MUDIFOB',
'MUDUKEF', 'MUFACIK', 'MUFOFOP', 'MUFOGUC', 'MUFYDUM', 'MUGAHEK', 'MUGAKOL', 'MUHYBAF', 'MUJAGIF', 'MUKAGOX', 'MULELEZ', 'MULYMUP',
'MUMIFAD', 'MY5BB', 'MYBYFUW', 'MYBYHER', 'MYDACIM', 'MYGIFUD', 'MYHAJIS', 'MYJILAS', 'MYJUFAJ', 'MYKOKIG', 'MYKYPAZ', 'MYLURAZ', 'MYNELIC']

def get_batch_corrected_blob_samples():
    return BATCH_CORRECTED_SAMPLES

# based on visual assessment of H&E, see /cluster/work/grlab/projects/projects2021-multivstain/meta/HE-QC.tsv
DENY_ROIS = ['MADUFEM_F4', 'MADUFEM_F5', 'MUDIFOB_F1', 'MUDIFOB_F2']

# original protein list from Simon
PROTEIN_LIST = ['MelanA', 'gp100', 'S100', 'SOX10', 'SOX9', 'TYRP1', 'Vimentin', 'GLUT1', 'PD-L1', 'HLA-ABC', 'HLA-DR', 'SMA', 'Caveolin-1',
'E-cadherin', 'CD31', 'b-catenin', 'Ki-67', 'pRb', 'EGFR', 'NGFR', 'pH2AX', 'PARP-CASP3', 'pErk1-2', 'pmTOR', 'CD45RO-RA', 'CD3',
'CD4', 'CD7', 'CD8a', 'FOXP3', 'PD-1', 'LAG-3', 'CD20', 'CD38', 'CD11c', 'CD16', 'CD68', 'Histone H3']
# with Irridium added 
PROTEIN_LIST_IR = ['MelanA', 'gp100', 'S100', 'SOX10', 'SOX9', 'TYRP1', 'Vimentin', 'GLUT1', 'PD-L1', 'HLA-ABC', 'HLA-DR', 'SMA', 'Caveolin-1',
'E-cadherin', 'CD31', 'b-catenin', 'Ki-67', 'pRb', 'EGFR', 'NGFR', 'pH2AX', 'PARP-CASP3', 'pErk1-2', 'pmTOR', 'CD45RO-RA', 'CD3',
'CD4', 'CD7', 'CD8a', 'FOXP3', 'PD-1', 'LAG-3', 'CD20', 'CD38', 'CD11c', 'CD16', 'CD68', 'Histone H3','Iridium191', 'Iridium193']

# Updated protein list (Simon has removed 'MPO' and 'CD15', we have added it back)
PROTEIN_LIST_FULL = ['MelanA', 'gp100', 'S100', 'SOX10', 'SOX9', 'TYRP1', 'Vimentin', 'GLUT1', 'PD-L1', 'HLA-ABC', 'HLA-DR', 'SMA', 'Caveolin-1',
'E-cadherin', 'CD31', 'b-catenin', 'Ki-67', 'pRb', 'EGFR', 'NGFR', 'pH2AX', 'PARP-CASP3', 'pErk1-2', 'pmTOR', 'CD45RO-RA', 'CD3',
'CD4', 'CD7', 'CD8a', 'FOXP3', 'PD-1', 'LAG-3', 'CD20', 'CD38', 'CD11c', 'CD16', 'CD68', 'Histone H3', 'MPO','CD15']
# Updated protein list (Simon has removed 'MPO' and 'CD15', we have added it back + Iridium)
PROTEIN_LIST_FULL_IR = ['MelanA', 'gp100', 'S100', 'SOX10', 'SOX9', 'TYRP1', 'Vimentin', 'GLUT1', 'PD-L1', 'HLA-ABC', 'HLA-DR', 'SMA', 'Caveolin-1',
'E-cadherin', 'CD31', 'b-catenin', 'Ki-67', 'pRb', 'EGFR', 'NGFR', 'pH2AX', 'PARP-CASP3', 'pErk1-2', 'pmTOR', 'CD45RO-RA', 'CD3',
'CD4', 'CD7', 'CD8a', 'FOXP3', 'PD-1', 'LAG-3', 'CD20', 'CD38', 'CD11c', 'CD16', 'CD68', 'Histone H3', 'MPO','CD15', 'Iridium191', 'Iridium193']
# The default protein list: all proteins found across the cohort
PROTEIN_LIST_MVS = PROTEIN_LIST_IR

# def get_protein_list():
#     return PROTEIN_LIST_MVS

def get_protein_list_by_name(protein_list_name='PROTEIN_LIST_MVS'):
    if protein_list_name=='PROTEIN_LIST_MVS':
        protein_list = PROTEIN_LIST_MVS
    elif protein_list_name=='PROTEIN_LIST':
        protein_list = PROTEIN_LIST
    elif protein_list_name=='PROTEIN_LIST_IR':
        protein_list = PROTEIN_LIST_IR
    elif protein_list_name=='PROTEIN_LIST_FULL':
        protein_list = PROTEIN_LIST_FULL
    elif protein_list_name=='PROTEIN_LIST_FULL_IR':
        protein_list = PROTEIN_LIST_FULL_IR
    else:
        print('Selected protein_list_name not supported!')
    return protein_list

# Replacing the default protein list for the one used to generate all new data
#protein2index = {prot: PROTEIN_LIST.index(prot) for prot in PROTEIN_LIST}
protein2index = {prot: PROTEIN_LIST_MVS.index(prot) for prot in PROTEIN_LIST_MVS}


# define a threshold for per-ROI means of protein expression (arsinh transformed):
# If the mean expression is lower than these values, they are believed to be faulty or missing.
# These values are somewhat arbitrary, chosen by visual inspection of the per-channel histograms.
protein_exprs_filter_vals = {
    'MelanA': 0.14,
    'gp100': 0.18,
    'S100': 0.2,
    'SOX10': 0.22,
    'SOX9': 0.07,
    'TYRP1': 0.1,
    'Vimentin': 0.0,  # looks good
    'GLUT1': 0.0,  # looks good
    'PD-L1': 0.0,  # looks good
    'HLA-ABC': 0.0,  # looks good
    'HLA-DR': 0.0,  # looks good
    'SMA': 0.0,  # looks good
    'Caveolin-1': 0.0,  # looks good
    'E-cadherin': 0.0,  # looks good
    'CD31': 0.0,  # looks good
    'b-catenin': 0.0,  # looks good
    'Ki-67': 0.0,  # looks good
    'pRb': 0.0,  # looks good
    'EGFR': 0.08,
    'NGFR': 0.0,  # looks good
    'pH2AX': 0.0,  # looks good
    'PARP-CASP3': 0.0,  # looks good
    'pErk1-2': 0.0,  # looks bad, but known to be noisy
    'pmTOR': 0.0,  # looks good
    'CD45RO-RA': 0.0,  # looks good
    'CD3': 0.0,  # looks good
    'CD4': 0.0,  # looks good
    'CD7': 0.0,  # looks good
    'CD8a': 0.0,  # looks good
    'FOXP3': 0.1,
    'PD-1': 0.0,  # looks good
    'LAG-3': 0.0,  # looks good
    'CD20': 0.08,
    'CD38': 0.0,  # looks good
    'CD11c': 0.0,  # looks good
    'CD16': 0.0,  # looks good
    'CD68': 0.0,  # looks good
    'Histone H3': 0.0  # looks good
}

# according to a feature importance analysis from sklearn's RandomForest,
# these are the 10 most important proteins regarding cell type classification
celltype_relevant_prots = ["CD16", "CD3", "CD8a", "HLA-DR", "SOX10", "CD20", "MelanA", "CD31", "CD45RO-RA", "CD4"]
# replaced sox10 with sox9 in "celltype_relevant_prots"
selected_prots = ["CD16", "CD3", "CD8a", "HLA-DR", "SOX9", "CD20", "MelanA", "CD31", "CD45RO-RA", "CD4"]
# predict 11 proteins, chosen by Ruben on 03.02.23 -- used for MICCAI
selected_prots_snr = ["CD16", "CD20", "CD3", "CD31", "CD8a", "GLUT1", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100"]
# predict 10 proteins -- same as miccai but w/o glut1; used for dgm4h
selected_prots_snr_dgm4h = ["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100"]
# same as dgm4h + sox10 as imp marker in ultivue 
selected_prots_snr_nature = ["CD16", "CD20", "CD3", "CD31", "CD8a", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SOX10"]

# predict 12 proteins, on 17/24.02.23, Ruben recommended to replace GLUT1 from selected_prots_snr with ki-67 and add SMA
selected_prots_snr_v2 = ["CD16", "CD20", "CD3", "CD31", "CD8a", "Ki-67", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "SMA"]
# predict 16 proteins, chosen by Ruben on 03.02.23, including the ones with low snr ratio
selected_prots_snr_ext = ["CD16", "CD20", "CD3", "CD31", "CD8a", "GLUT1", "gp100", "HLA-ABC", "HLA-DR", "MelanA", "S100", "PD-1", "PD-L1", 'CD4'] #"Iridium193", "Iridium191", "SOX9"]
# for miccai 10 markers 
prots_pseudo_multiplex = ["MelanA", "CD16", "CD20", "CD3", "CD8a", "HLA-ABC", "HLA-DR", "S100", "Ki-67", "gp100"]
# for miccai, TLS related
prots_tls = ["MelanA","CD8a", "CD3", "CD20", "S100"]
# for miccai, CD16 and ones with positive correaltion with CD16
prots_cd16_correlates = ["CD16", "HLA-DR", "MelanA", "S100"]
# for miccai, Ki-67 and ones with positive correaltion with Ki-67 
prots_ki67_correlates = ["Ki-67", "MelanA", "S100"]

# extended set of proteins
celltype_relevant_prots_ext = ["CD16", "CD3", "CD8a", "HLA-DR", "SOX10", "CD20", "MelanA", "CD31", "CD45RO-RA", "CD4", "CD68", "CD11c"]
celltype_relevant_prots_ext_ir = ["CD16", "CD3", "CD8a", "HLA-DR", "SOX10", "CD20", "MelanA", "CD31", "CD45RO-RA", "CD4", "CD68", "CD11c", 'Iridium191', 'Iridium193']
immune_prots = ["CD3", "CD8a", "CD20", "CD45RO-RA", "CD4"]
tumor_prots = ["SOX9", "MelanA"]
tumor_cd8 = ["MelanA", "CD8a"]
tcell_prots = ["CD3","CD8a","CD4"]
ir_cd8 = ['Iridium193',"CD8a"]
cd3_cd8 = ["CD3","CD8a"]
tumor_cd8_cd3 = ["CD3","CD8a","MelanA"]

prot_names_raw2deriv = {
    'Myelope': 'MPO',           # no longer included in batch-corrected data
    'MPO': 'MPO',               # no longer included in batch-corrected data
    'Histone': 'Histone H3',
    'SMA': 'SMA',
    'E-Cadhe': 'E-cadherin',    # deriv used to be 'E-cadherin'
    'MART1Me': 'MelanA',
    'CD38': 'CD38',
    'HLA-DR': 'HLA-DR',
    'S100': 'S100',
    'Melanom': 'gp100',
    'Sox9': 'SOX9',
    'Glucose': 'GLUT1',
    'CD20': 'CD20',
    'CD68': 'CD68',
    'Erk12': 'pErk1-2',         # deriv used to be 'pErk1/2'
    'CD3': 'CD3',
    'LAG-3': 'LAG-3',
    'CD11c': 'CD11c',
    'CD279': 'PD-1',
    'H2AX': 'pH2AX',
    'CD16': 'CD16',
    'p75': 'NGFR',              # deriv used to be 'p75 NGFR', raw can apparently sometimes be 'p75NGF'
    'CD274': 'PD-L1',
    'CD45RA': 'CD45RO-RA',      # deriv used to be 'CD45RO/RA'
    'FOXP3': 'FOXP3',
    'TRP1': 'TYRP1',
    'b-Caten': 'b-catenin',
    'CD8a': 'CD8a',
    'CD7': 'CD7',
    'Ki-67': 'Ki-67',
    'EGFR': 'EGFR',
    'SOX10': 'SOX10',
    'CD4': 'CD4',
    'CD15': 'CD15',             # no longer included in batch-corrected data
    'CD31': 'CD31',
    'mTOR': 'pmTOR',
    'HLA-ABC': 'HLA-ABC',
    'Rb_phospho': 'pRb',
    'Rb': 'pRb',
    'cleaved': 'PARP-CASP3',    # deriv used to be 'cleaved PARP/CASP3'
    'Caveoli': 'Caveolin-1',
    'Vimenti': 'Vimentin',
    '191Ir': 'Iridium191',
    '193Ir': 'Iridium193'
}

ROOT_DIR_ROI_COORDS = "/cluster/work/tumorp/data-drop/imc_he_rois/study/melanoma/"
ROOT_DIR_STUDY = "/cluster/work/tumorp/data_repository/study/"
FILTERED_SCE_STORAGE = "/cluster/work/grlab/projects/projects2021-imc_morphology/filtered_sce/"
MANUAL_ALIGNS_STORAGE = "/cluster/work/grlab/projects/projects2021-imc_morphology/template_matching/manual_aligns/"
ADATA_STORAGE = '/cluster/work/grlab/projescts/tumor_profiler/data/study/'

# relative paths to project_path
BINARY_IMC_ROI_STORAGE = "data/tupro/binary_imc_rois/"
BINARY_HE_ROI_STORAGE = "data/tupro/binary_he_rois/"
RESULTS_DIR = 'results/'
PRETRAINED_DIR = 'results/models/'
LOGS_DIR = 'logs/'
META_DIR = 'meta/'
DATA_DIR = 'data/tupro/'
COHORT_STATS_PATH = 'data/tupro/agg_stats_qc/'
# v4 includes removal of ROIs with poor quality HE, v4 was used for miccai submission, v6 contains weights for sparse markers for upweighting samples with high count of sparse markers 
CV_SPLIT_ROIS_PATH = 'meta/tupro/mvs-cv_split-rois-v4.json'
CV_SPLIT_SAMPLES_PATH = 'meta/tupro/mvs-cv_split-samples-v4.json'
CV_SPLIT_DICT_PATH = 'meta/tupro/mvs-cv_split-dict-v4.json'
# CV_SPLIT_ROIS_PATH = 'meta/tupro/mvs-cv_split-rois-v6.json'
# CV_SPLIT_SAMPLES_PATH = 'meta/tupro/mvs-cv_split-samples-v6.json'
# CV_SPLIT_DICT_PATH = 'meta/tupro/mvs-cv_split-dict-v6.json'

def get_he_roi_storage(data_path, which_HE):
    ''' Function to return path the HE data based on if it is stored in local scratch or shared dir 
    '''
    if which_HE=='new':
        return Path(data_path).joinpath('binary_he_rois/')
    elif which_HE=='old': 
        return Path(data_path).joinpath('binary_he_rois_old/')
    elif which_HE=='new_normalised':
        return Path(data_path).joinpath('binary_he_rois_normalised/')
    
    

def get_imc_roi_storage(data_path, imc_prep_seq='raw', standardize_imc=True, scale01_imc=True, cv_split='split3'):
    ''' Function to return path the IMC data preprocessed according to imc_prep_seq
    imc_prep_seq: Sequence of IMC preprocessing steps {raw, raw_clip99, raw_clip99_arc, raw_median_arc, raw_smooth_arc}
    '''
    imc_path = str(Path(data_path).joinpath('binary_imc_rois_' + imc_prep_seq + (scale01_imc or standardize_imc)*('_' + standardize_imc*'std_'+ scale01_imc*'minmax_'+ cv_split)+'/'))
    if os.path.exists(imc_path):
        return imc_path
    else: 
        return str(Path(data_path).joinpath('binary_imc_rois_' + imc_prep_seq + '/'))
    # return data_path + '/binary_imc_rois_' + imc_prep_seq + (scale01_imc or standardize_imc)*('_' + standardize_imc*'std_'+ scale01_imc*'minmax_'+ cv_split) + '/'


IMC_RESOLUTION = 1.0  # in um/px

# This is used to standardize transformed protein expression. The transformation used was arsinh with cofactor 1.
# Data was obtained from batch corrected /cluster/work/tumorp/share/melanoma_cohort/batch_normalized_data/imc/Cohort-batchCorrection/Cohort-batchCorrection_bc.rds 
EXPRS_AVG = [1.4556346875382629, 1.5823743911266717, 1.769994502394774, 1.0390995720790535, 0.46368821125433507,
    0.7305862604521794, 1.8868455082223556, 1.80931714868742, 0.9152619039726677, 2.50273052958669, 1.3927806659417175,
    0.3955638500315327, 0.9349052910639893, 1.0645080214440712, 0.763581492015808, 2.4362084507634956, 0.8383515542329654,
    1.3041666783428814, 0.4430554533743364, 0.708625549764007, 0.9166934215392144, 0.48249269883259177, 0.7370910708481483,
    0.755021054263609, 1.9565692259242247, 0.7911760810908135, 0.8766538555839076, 1.1106016003730008, 0.8710548402373615,
    0.613711747169342, 0.457980934941386, 0.2754991327643782, 0.4655180199856597, 0.4892132800080844, 0.9717525067375753,
    1.135085807361201, 1.3035353895141941, 1.1900617924396166]

EXPRS_STDEV = [1.2142834257592858, 0.9809487233912172, 1.2676768234541813, 0.6855125971146513, 0.5474948238270885,
    0.8640254470728359, 0.6758659984185073, 0.8245436894889623, 0.5705551591645466, 0.9649434298855799, 1.174638537886921,
    0.4800475870362921, 0.5817826375852084, 0.5510362382595341, 0.5748760548533872, 0.9257700048018475, 1.0257762601016698,
    0.7169788845424322, 0.3559013739786092, 0.4405512055232236, 0.7196853274329059, 0.4379451488008557, 0.6233517490012672,
    0.4796604171741563, 0.8528983893693014, 0.6059887869870207, 0.5223025798560519, 0.7927829228412263, 0.7414801847418141,
    0.35699413825980447, 0.3074401669726019, 0.20752239191308994, 0.5737402924424652, 0.3682429792314788, 0.6405162748020702,
    1.0411340697692342, 0.8036611414305034, 0.6193969878273543]

REF_ROIS = ['Ctrl', 'Ctrl1', 'Ctrl2', 'M', 'M.end', 'M.start', 'O', 'OP', 'OP', 'OP.end', 'OP.start',
               'P1.end', 'P1.start', 'P2.end', 'P2.start', 'P40', 'P40', 'P40.2','P40.end', 'P40.start', 'P80',
               'P80.end', 'P80.start', 'T', 'T.end', 'T.start', 'TLS1', 'TLS2', 'control', 'control1',
               'control2', 'ctrl1', 'ctrl2', 'ctrl3', 'ctrl']

IMC_ROIS = ['A','B','C','D','E','F','G','C1','C2','C3','C4','C5','C6','F1','F2','F3','F4','F5','F6']

def get_ref_rois():
    return REF_ROIS

# the different cell types that this projects considers.
# Some subgroups were merged together for simplicity and to increase predictive power.
# Since there already were two type-related fields in the batch-corrected dataframe (namely "cell_simple" and "cell"),
# and to complete the confusion, this information is accessible with column ID "YACT" (yet another cell type)
# Note that this cell type derives from "cell", see cell2YACT for the precise mapping.
# Note that "UNKNOWN" indicates that the "cell" cell type could be found in the cell2YACT keys (should only concern < 1% of all cells)

# should be in alphabetical order!
CELL_TYPES = ['Bcells', 'Tcells.CD4', 'Tcells.CD8', 'myeloid', 'other','tumor', 'vessels']

cell2YACT = {"B-cells": "B-cells",
             "CD4+PD1+": "CD4",
             "CD8+PD1+": "CD8",
             "CD4 T-cells": "CD4",
             "CD8 T-cells": "CD8",
             "byCD4": "CD4",
             "byCD8": "CD8",
             "granulocytes": "myeloid",
             "myeloid": "myeloid",
             "plasma cells": "plasma cells",
             #"stroma": "stroma",
             "stroma": "other",
             "NK": "other",
             "nan": "other",
             "tumor": "tumor",
             "vessels": "vessels",
             "other":"other"}


# This section describes the data-split that was used for the results in the written report.
# As of 2021-09-27, it is also what code.utils.dataset_utils.clean_train_val_test_sep_for_manual_aligns_ordered_by_quality() returns,
# but that may change if the set of manual alignment results is changed.

TRAIN_SAMPLES = ['MACOLUD', 'MADAJEJ', 'MADEGOD', 'MADUBIP', 'MADUFEM', 'MAFIBAF', 'MAHEFOG', 'MAHOBAM', 'MAJEFEV', 'MAJOFIJ', 'MAKYGIW', 'MANOFYB',
'MEBIGAL', 'MECADAC', 'MECUGUH', 'MECYGYR', 'MEDEFUC', 'MEDYCAR', 'MEFYFAK', 'MEGEGUJ', 'MEHEHUM', 'MEKAKYD', 'MEKOBAB', 'MELYPEB', 'MEMEMUH', 'MEMIGOG',
'MEVIXYV', 'MEZYWEG', 'MIBAFUK', 'MICEGOK', 'MIDEKOG', 'MIFOGIL', 'MIGEKUT', 'MIHIFIB', 'MIJYDYP', 'MIKOBID', 'MIPYNAP', 'MISYPUP', 'MOBICUN', 'MOCELOJ',
'MODALEG', 'MODIGOS', 'MODOHOX', 'MOGYHOJ', 'MOJYMOC', 'MOPYPUS', 'MOQAVIJ', 'MOVAZYQ', 'MUBIBIT', 'MUBOMEF', 'MUBYJOF', 'MUCADOP', 'MUDEFAW', 'MUDIFOB',
'MUDUKEF', 'MUFYDUM', 'MUGAHEK', 'MUGAKOL', 'MUHYBAF', 'MUKAGOX', 'MULELEZ', 'MULYMUP', 'MYBYHER', 'MYDACIM', 'MYGIFUD', 'MYHAJIS', 'MYJILAS', 'MYJUFAJ',
'MYKOKIG', 'MYKYPAZ', 'MYLURAZ', 'MYNELIC']

VALID_SAMPLES = ['MEFOCUR', 'MOBUBOT', 'MUMIFAD', 'MYBYFUW']

TEST_SAMPLES = ['MALYLEJ', 'MEHUFEF', 'MOBYLUD', 'MOMUSIG']


# load all the MVS samples
# originally used samples (all well-aligned 80 samples)
MVS_SAMPLES = ['MACOLUD', 'MADAJEJ', 'MADEGOD', 'MADUBIP', 'MAFIBAF', 'MAHEFOG', 'MAHOBAM', 'MAJEFEV', 'MAJOFIJ', 'MAKYGIW', 'MANOFYB',
'MEBIGAL', 'MECADAC', 'MECUGUH', 'MECYGYR', 'MEDEFUC', 'MEDYCAR', 'MEFYFAK', 'MEGEGUJ', 'MEHEHUM', 'MEKAKYD', 'MEKOBAB', 'MELYPEB', 'MEMEMUH', 'MEMIGOG',
'MEVIXYV', 'MEZYWEG', 'MIBAFUK', 'MICEGOK', 'MIDEKOG', 'MIFOGIL', 'MIGEKUT', 'MIHIFIB', 'MIJYDYP', 'MIKOBID', 'MIPYNAP', 'MISYPUP', 'MOBICUN', 'MOCELOJ',
'MODALEG', 'MODIGOS', 'MODOHOX', 'MOGYHOJ', 'MOJYMOC', 'MOPYPUS', 'MOQAVIJ', 'MOVAZYQ', 'MUBIBIT', 'MUBOMEF', 'MUBYJOF', 'MUCADOP', 'MUDEFAW',
'MUDUKEF', 'MUFYDUM', 'MUGAHEK', 'MUGAKOL', 'MUHYBAF', 'MUKAGOX', 'MULELEZ', 'MULYMUP', 'MYBYHER', 'MYDACIM', 'MYGIFUD', 'MYHAJIS', 'MYJILAS', 'MYJUFAJ',
'MYKOKIG', 'MYKYPAZ', 'MYLURAZ', 'MYNELIC', 'MEFOCUR', 'MOBUBOT', 'MUMIFAD', 'MYBYFUW','MALYLEJ', 'MEHUFEF', 'MOBYLUD', 'MOMUSIG'] # removed after HE QC: 'MADUFEM', 'MUDIFOB'

mask_special_cases = ['MAJOFIJ_A', 'MAJOFIJ_C', 'MAJOFIJ_D', 'MAJOFIJ_E', 'MANOFYB_A', 'MANOFYB_B', 'MANOFYB_C', 'MANOFYB_E', 'MEDEFUC_A', 'MEDEFUC_C', 'MEDEFUC_D', 'MEDEFUC_E', 'MEKOBAB_A', 'MEKOBAB_B', 'MEKOBAB_C', 'MEKOBAB_D', 'MEKOBAB_E', 'MEKOBAB_F', 'MEVIXYV_C1', 'MEVIXYV_C3', 'MEVIXYV_C4', 'MEVIXYV_C5', 'MEVIXYV_C6', 'MISYPUP_C1', 'MISYPUP_C2', 'MISYPUP_C4', 'MISYPUP_C5', 'MOPYPUS_C2', 'MOPYPUS_C3', 'MOPYPUS_F2', 'MOBUBOT_C3', 'MOBUBOT_F3', 'MOBYLUD_C2', 'MOBYLUD_F2']