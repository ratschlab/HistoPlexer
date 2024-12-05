import importlib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

class TBLogger(object):
    def __init__(self, log_dir=None):
        super(TBLogger, self).__init__()
        self.log_dir = log_dir
        tb_module = importlib.import_module("torch.utils.tensorboard")
        self.tb_logger = getattr(tb_module, "SummaryWriter")(log_dir=self.log_dir)
    
    def end(self):
        self.tb_logger.flush()
        self.tb_logger.close()
    
    def run(self, func_name, *args, mode="tb", **kwargs):
        if func_name == "log_scalars":
            return self.tb_log_scalars(*args, **kwargs)
        else:
            tb_log_func = getattr(self.tb_logger, func_name)
            return tb_log_func(*args, **kwargs)
        return None

    def tb_log_scalars(self, metric_dict, step):
        for k, v in metric_dict.items():
            self.tb_logger.add_scalar(k, v, step)

class MetricLogger(object):
    def __init__(self):
        super(MetricLogger, self).__init__()
        self.y_scores = []  # Storing scores for each class
        self.y_true = []

    def log(self, scores, Y):
        # Assuming `scores` is a tensor of shape (n_samples, n_classes) with probabilities for each class
        # and `Y` is the true class labels
        scores = scores.cpu().numpy()  # Convert scores to a NumPy array if it's a PyTorch tensor
        Y = Y.cpu().numpy()  # Convert Y to a NumPy array if it's a PyTorch tensor
        self.y_scores.extend(scores)  # Append the scores for each class
        self.y_true.extend(Y)  # Append the true labels

    def get_summary(self):
        # Convert lists to NumPy arrays for computation
        y_scores = np.array(self.y_scores)
        y_true = np.array(self.y_true)

        acc = accuracy_score(y_true=y_true, y_pred=np.argmax(y_scores, axis=1))  # Accuracy
        f1 = f1_score(y_true=y_true, y_pred=np.argmax(y_scores, axis=1), average=None)  # F1 score for each class
        weighted_f1 = f1_score(y_true=y_true, y_pred=np.argmax(y_scores, axis=1), average='weighted')  # Weighted F1 score
        kappa = cohen_kappa_score(y1=y_true, y2=np.argmax(y_scores, axis=1), weights='quadratic')  # Cohen's kappa
        # auc = roc_auc_score(y_true=y_true, y_score=y_scores, multi_class='ovr')  # AUC for multi-class

        # Printing metrics
        print('*** Metrics ***')
        print('* Accuracy: {}'.format(acc))
        for i, score in enumerate(f1):
            print('* Class {} f1-score: {}'.format(i, score))
        print('* Weighted f1-score: {}'.format(weighted_f1))
        print('* Kappa score: {}'.format(kappa))
        # print('* AUC: {}'.format(auc))

        # Creating summary dictionary
        summary = {'accuracy': acc, 'weighted_f1': weighted_f1, 'kappa': kappa}
        for i, score in enumerate(f1):
            summary[f'class_{i}_f1'] = score
        
        return summary

    def get_confusion_matrix(self):
        y_pred = np.argmax(np.array(self.y_scores), axis=1)  # Predicted classes
        cf_matrix = confusion_matrix(y_true=np.array(self.y_true), y_pred=y_pred)  # Confusion matrix
        return cf_matrix