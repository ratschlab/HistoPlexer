import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix as cfm


def get_manual_aggregation(rf, x, types, min_p=0, decide_at_random=True, seed=42):
    '''
    This function is needed because the sklearn RF for some reason returned all-zero prediction results in some cases.
    This function makes sure this doesn't happen.
    types: one of the types must be 'other'
    min_p: probability threshold (all votes with lower probability will be discarded, default 0)
    decide_at_random: if the algorithm is unsure about the final call, decide at random between options (otherwise assigned to "other")
    seed: random seed
    '''
    assert 'other' in types, 'One of the cell types must be "other" to capture unassigned cells'
    idx_other = [i for i,x in enumerate(types) if x == "other"]
    
    raw_result = np.zeros((x.shape[0], len(types)))
    for i in range(len(rf.estimators_)):
        raw_result += rf.estimators_[i].predict(x)
        
    # remove all entries below min_p
    if min_p > 0:
        import pdb; pdb.set_trace()
        raw_result[raw_result < min_p] = 0
    
    # do max-voting:
    aggr_result = np.zeros((raw_result.shape[0], raw_result.shape[1]))
    for i in range(raw_result.shape[0]):
        is_maximum = (raw_result[i, :] == raw_result[i, :].max())
        num_of_max_entries = is_maximum.sum()
        
        if num_of_max_entries == 1:
            aggr_result[i, np.argmax(raw_result[i, :])] = 1
        else:
            if decide_at_random:
                np.random.seed(seed)
                candidates = np.nonzero(is_maximum)[0].tolist()
                aggr_result[i, random.choice(candidates)] = 1
            else:
                # if no clear candidate found, then set to "other"
                aggr_result[i, idx_other] = 1
            
    return aggr_result
            
def plot_feature_imp(features, feature_imp):
    ''' Plot a barplot of feature importances
    features: list of feature names
    feature_imp: feature importance matrix from random foret (rf.feature_importances_)
    '''
    fi_df = pd.DataFrame({'names': features, 'importance': feature_imp}) 
    fi_df.sort_values(by=['importance'], ascending=False, inplace=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='importance', y='names', data=fi_df, color='lightblue')
    plt.title('Random Forest classification feature importance by protein')
    plt.xlabel('Feature Importance')
    plt.ylabel('')
    
    
def plot_cfm(y_gt, y_pred, labels, normalize='true'):
    ''' Function to plot confusion matrix
    y_gt: vector with ground truth labels
    y_pred: vector with predicted labels
    labels: list of labels (names)
    normalize: how to normalize counts ('true': by the GT sum per label)
    '''
    fig, ax = plt.subplots(figsize=(8,8))
    confusion_matrix = cfm(y_gt, y_pred, normalize=normalize)
    # round to the second decimal place
    confusion_matrix = np.around(confusion_matrix,2)
    cfm_display = ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
    cfm_display.plot(ax=ax)
    plt.xticks(rotation=90)