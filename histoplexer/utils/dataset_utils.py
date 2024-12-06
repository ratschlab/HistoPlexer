import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from histoplexer.utils.constants import *

#  Holds code for dataset creation / filtering.

def clean_train_val_test_sep_for_manual_aligns_ordered_by_quality(report_set=True, val_part=0.05, test_part=0.05):
    ''' This function creates a clean train-val-test separation for manual alignment results.
    report_set: If True, the data split used in the written report is returned. If false, it computes a new data split based on the list of manual alignment result files in the MANUAL_ALIGNS_STORAGE
    val_part: Validation part of the split.
    test_part: Test part of the split.
    '''
    manual_aligns_path = MANUAL_ALIGNS_STORAGE
    align_results_paths = [os.path.join(manual_aligns_path, f.name) for f in os.scandir(manual_aligns_path) if f.is_file() and f.name.endswith(".json")]

    align_results = []
    for arp in align_results_paths:
        with open(arp) as json_file:
            align_results.append(json.load(json_file))
            json_file.close()

    assert len([ar for ar in align_results if ar["precise_enough_for_l1"] and len(ar["good_areas"]) > 0]) == 0, "Inconsistency in align results !"

    if report_set:
        train_aligns = [ar for ar in align_results if ar["sample"] in TRAIN_SAMPLES]
        val_aligns = [ar for ar in align_results if ar["sample"] in VALID_SAMPLES]
        test_aligns = [ar for ar in align_results if ar["sample"] in TEST_SAMPLES]

        assert len([ar for ar in train_aligns if ar in val_aligns or ar in test_aligns]) == 0, "Data split not clean, train set"
        assert len([ar for ar in test_aligns if ar in train_aligns or ar in val_aligns]) == 0, "Data split not clean, test set"
    else:
        # compute data split with intended ordering
        # get list of all patients:
        patients = np.unique([ar["patient"] for ar in align_results])

        # the strategy now is to order the patients by decreasing quality: first samples with precise ARs, then those that at least have some good areas, then the rest.
        # the train set is then chosen from the start of the list, and the val/test set from the back, ensuring that we get the maximum of good data for train set.
        # use patients IDs, so we ensure that no leaking is happening
        patients_with_precise_rois = []
        patients_no_precise_rois_but_good_areas = []
        patients_imprecise = []  # the *alignments* are still good, the morphological differences are simply very large for this set.

        for p in patients:
            ars_for_patient = [ar for ar in align_results if ar["patient"] == p]

            if any([ar["precise_enough_for_l1"] for ar in ars_for_patient]):
                patients_with_precise_rois.append(p)
            elif any([len(ar["good_areas"]) > 0 for ar in ars_for_patient]):
                patients_no_precise_rois_but_good_areas.append(p)
            else:
                patients_imprecise.append(p)

        patients_ordered_by_quality = patients_with_precise_rois + patients_no_precise_rois_but_good_areas + patients_imprecise
        assert len(patients) == len(patients_ordered_by_quality), "Fatal logical inconsistency !"

        # round up number of val/test patients
        num_val_patients, num_test_patients = int(val_part * len(patients) + 0.5), int(test_part * len(patients) + 0.5)

        train_end = len(patients) - num_val_patients - num_test_patients
        val_end = train_end + num_val_patients
        test_end = len(patients)

        train_patients = patients_ordered_by_quality[0: train_end]
        val_patients = patients_ordered_by_quality[train_end: val_end]
        test_patients = patients_ordered_by_quality[val_end: test_end]

        # finally, convert back to align results
        train_aligns = [ar for ar in align_results if ar["patient"] in train_patients]
        val_aligns = [ar for ar in align_results if ar["patient"] in val_patients]
        test_aligns = [ar for ar in align_results if ar["patient"] in test_patients]

    return train_aligns, val_aligns, test_aligns
            
# get patient-sample mapping:
def get_patient_samples(data_dir = '/cluster/work/tumorp/data_repository/', phase='study', indication=None):
    ''' Get dataframe with all patient/sample pairs for a given indication and project phase
    data_dir: root directory containing data
    phase: project phase {study, poc, post-poc}
    indication: cancer indication {None, melanoma, ovca, aml}
    '''
    
    patients = os.listdir(Path(data_dir).joinpath(phase))
    if indication is not None:
        if indication=='melanoma':
            patients = [x for x in patients if x[3]=='M']
        elif idication=='ovca':
            patients = [x for x in patients if x[3]=='G']
        elif idication=='aml':
            patients = [x for x in patients if x[3]=='A']
            
    patient_sample_dict = dict()
    for patient in patients:
        patient_sample_dict[patient] = pd.Series(os.listdir(Path(data_dir).joinpath(phase, patient)))
        
    patient_sample_df = pd.concat(patient_sample_dict).reset_index()
    patient_sample_df.columns = ['patient', 'idx', 'sample']
    patient_sample_df = patient_sample_df.loc[:,['patient', 'sample']]
    
    return patient_sample_df

