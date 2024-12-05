import os

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
from pycox.models.utils import pad_col, make_subgrid

from src.utils.io import save_pkl
from src.utils.monitor import Monitor
from src.utils.logger import MetricLogger, TBLogger
from src.utils.eval import EvalSurv
from src.utils.loss import CrossEntropySurvLoss, NLLSurvLoss, CoxSurvLoss
from src.models.attention import AttnMIL, MCATMIL, AttnSurvMIL, MCATSurvMIL

# ----------- classification trainer ----------- #

class ClassificationTrainer(object):
    def __init__(self, args, datasets, fold):

        # general config
        self.config = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fold = fold
        self.max_epochs = args.max_epochs
        self.save_path = os.path.join(args.save_path, f"fold_{fold}")
        self.pretrained_path = args.pretrained_path
        
        # logger config
        self.tb_logger = TBLogger(log_dir=self.save_path)
        
        # dataloader config
        train_dataset, val_dataset, test_dataset = datasets
        if args.is_weighted_sampler:
            N = float(len(train_dataset))
            weight_per_class = [N/len(train_dataset.slide_cls_ids[c]) for c in range(len(train_dataset.slide_cls_ids))] 
            weight = [0] * int(N)
            for idx in range(len(train_dataset)):
                y = train_dataset.get_label(idx)
                weight[idx] = weight_per_class[y]
            weight = torch.DoubleTensor(weight)
            self.train_loader = DataLoader(train_dataset, batch_size=1, sampler=WeightedRandomSampler(weight, len(weight)), num_workers=args.num_workers)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=1, sampler=RandomSampler(train_dataset), num_workers=args.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=1, sampler=SequentialSampler(val_dataset), num_workers=args.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset), num_workers=args.num_workers)
        
        # model config
        self.n_cls = args.n_cls
    
        if args.model == 'ABMIL':
            self.model = AttnMIL(in_feat_dim_he=args.in_feat_dim_he, in_feat_dim_imc=args.in_feat_dim_imc, hidden_feat_dim=256, out_feat_dim=256, dropout=args.drop_out, n_cls=self.n_cls, fusion=args.fusion)
        elif args.model == 'MCAT':
            self.model = MCATMIL(in_feat_dim_he=args.in_feat_dim_he, in_feat_dim_imc=args.in_feat_dim_imc, hidden_feat_dim=256, out_feat_dim=256, dropout=args.drop_out, n_cls=self.n_cls, fusion=args.fusion)
        else:
            raise NotImplementedError

        # loss config
        self.loss_fn = nn.CrossEntropyLoss()

        # optimizer config
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.reg)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=args.scheduler_decay_rate, patience=args.scheduler_patience)
        
        # model saver config
        self.early_stop = args.early_stop
        self.save_metric = args.save_metric
        self.monitor = Monitor(save_path=self.save_path, metric=self.save_metric)

    def _print_model(self):
        num_params = 0
        num_params_train = 0
        print(self.model)
        
        for param in self.model.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        
        print('Total number of parameters: %d' % num_params)
        print('Total number of trainable parameters: %d' % num_params_train)

    def _train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        train_loss = 0.

        for _, data in tqdm(enumerate(self.train_loader)):
            feats = (data[0][0].to(self.device), data[0][1].to(self.device))
            label = data[1].to(self.device)

            # forward pass
            logits, _, _ = self.model(feats)

            # loss
            loss = self.loss_fn(logits, label)
            train_loss += loss.item()
            loss.backward()

            # optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        train_loss /= len(self.train_loader)
        return train_loss
    
    def _val_one_epoch(self):
        self.model.eval()
        
        val_logger = MetricLogger()
        val_loss = 0.

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.val_loader)):
                feats = (data[0][0].to(self.device), data[0][1].to(self.device))
                label = data[1].to(self.device)
                
                # forward pass
                logits, _, Y_prob = self.model(feats)
                val_logger.log(Y_prob, label)
                
                # loss
                loss = self.loss_fn(logits, label)
                val_loss += loss.item()

        print('*** VALIDATION ***')
        val_summary = val_logger.get_summary()
        val_summary = {'val_{}'.format(key): val for key, val in val_summary.items()}
        val_loss /= len(self.val_loader)
        val_summary["val_loss"] = val_loss

        return val_loss, val_summary
    
    def train(self):    
        print("Start training...")
        for epoch in range(self.max_epochs):
            train_loss = self._train_one_epoch()
            val_loss, val_summary = self._val_one_epoch()
            self.monitor(epoch, val_summary, self.model)
            if self.monitor.stop and self.early_stop:
                break
            self.scheduler.step(val_loss)
            train_summary = {
                "train_loss": train_loss, 
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            self.tb_logger.run(func_name="log_scalars", mode="tb", metric_dict=train_summary, step=epoch)
            self.tb_logger.run(func_name="log_scalars", mode="tb", metric_dict=val_summary, step=epoch)
            
        print("Training finished!")

    def test(self):
        
        if self.pretrained_path:
            print(f'Loading pretrained model from {self.pretrained_path}...')
            self.model.load_state_dict(torch.load(self.pretrained_path)) # test pretrained model if provided
        else:
            print(f'Loading best {self.save_metric} model from {os.path.join(self.save_path, f"model_best_{self.save_metric}.pt")}...')
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, f'model_best_{self.save_metric}.pt')))
        self.model.eval()
            
        self.model.eval()

        test_logger = MetricLogger()

        with torch.no_grad():
            for _, data in tqdm(enumerate(self.test_loader)):
                feats = (data[0][0].to(self.device), data[0][1].to(self.device))
                label = data[1].to(self.device)
                coords = data[2]
                slide_id = data[3][0]
                
                # forward pass
                _, Y_hat, Y_prob, A1, A2 = self.model(feats, output_attention=True)
                test_logger.log(Y_prob, label)
                
                # attention_scores = A.squeeze().cpu().numpy()
                # coords = coords.squeeze().numpy()
                
                # attention_path = os.path.join(self.save_path, 'attentions')
                # os.makedirs(attention_path, exist_ok=True)
                # if label == Y_hat:
                #     with h5py.File(os.path.join(attention_path, f'{slide_id}_{label.item()}.h5'), 'w') as hf:
                #         hf.create_dataset('coords', data=coords, compression="gzip", compression_opts=9)
                #         hf.create_dataset('attention_score', data=attention_scores, compression="gzip", compression_opts=9)
                        
                #         # Calculate high and low attention scores and their indices
                #         indices_high_to_low = np.argsort(attention_scores)[::-1]
                #         top_100_indices = indices_high_to_low[:100]
                #         bottom_100_indices = indices_high_to_low[-100:]
                        
                #         coords_high = coords[top_100_indices].copy()
                #         attention_scores_high = attention_scores[top_100_indices].copy()
                #         coords_low = coords[bottom_100_indices].copy()
                #         attention_scores_low = attention_scores[bottom_100_indices].copy()
                        
                #         # Save the high and low attention scores and their coords
                #         hf.create_dataset('coords_high', data=coords_high, compression="gzip", compression_opts=9)
                #         hf.create_dataset('attention_scores_high', data=attention_scores_high, compression="gzip", compression_opts=9)
                #         hf.create_dataset('coords_low', data=coords_low, compression="gzip", compression_opts=9)
                #         hf.create_dataset('attention_scores_low', data=attention_scores_low, compression="gzip", compression_opts=9)
                # else:
                #     with h5py.File(os.path.join(attention_path, f'{slide_id}_{label.item()}_pred_{Y_hat.item()}.h5'), 'w') as hf:
                #         hf.create_dataset('coords', data=coords, compression="gzip", compression_opts=9)
                #         hf.create_dataset('attention_score', data=attention_scores, compression="gzip", compression_opts=9)
                        
                #         # Calculate high and low attention scores and their indices
                #         indices_high_to_low = np.argsort(attention_scores)[::-1]
                #         top_100_indices = indices_high_to_low[:100]
                #         bottom_100_indices = indices_high_to_low[-100:]
                        
                #         coords_high = coords[top_100_indices].copy()
                #         attention_scores_high = attention_scores[top_100_indices].copy()
                #         coords_low = coords[bottom_100_indices].copy()
                #         attention_scores_low = attention_scores[bottom_100_indices].copy()
                        
                #         # Save the high and low attention scores and their coords
                #         hf.create_dataset('coords_high', data=coords_high, compression="gzip", compression_opts=9)
                #         hf.create_dataset('attention_scores_high', data=attention_scores_high, compression="gzip", compression_opts=9)
                #         hf.create_dataset('coords_low', data=coords_low, compression="gzip", compression_opts=9)
                #         hf.create_dataset('attention_scores_low', data=attention_scores_low, compression="gzip", compression_opts=9)
                
        print('*** TEST ***')
        test_summary = test_logger.get_summary()
        test_summary = {'test_{}'.format(key): val for key, val in test_summary.items()}
        
        self.tb_logger.run(func_name="log_scalars", mode="tb", metric_dict=test_summary, step=0)
        
        # save confusion matrix
        cf_matrix = test_logger.get_confusion_matrix()
        ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
        ax.set_title('Confusion Matrix\n\n')
        ax.set_xlabel('\nPrediction')
        ax.set_ylabel('Ground Truth')
        ticklabels = [f'Class {i}' for i in range(self.n_cls)]
        ax.xaxis.set_ticklabels(ticklabels)
        ax.yaxis.set_ticklabels(ticklabels)
        plt.savefig(os.path.join(self.save_path, f'Confusion_Matrix.png'))
        plt.clf()
        print("Testing finished!")
        
        return test_summary


# ----------- survival trainer ----------- #

class SurvTrainer(object):
    def __init__(self, args, datasets, fold):

        # general config
        self.config = args
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epochs = args.max_epochs
        self.save_path = os.path.join(args.save_path, f"fold_{fold}")
        self.pretrained_path = args.pretrained_path
        
        # logger config
        self.tb_logger = TBLogger(log_dir=self.save_path)
        
        # dataloader config
        train_dataset, val_dataset, test_dataset = datasets
        if args.is_weighted_sampler:
            N = float(len(train_dataset))
            weight_per_class = [N/len(train_dataset.slide_cls_ids[c]) for c in range(len(train_dataset.slide_cls_ids))] 
            weight = [0] * int(N)
            for idx in range(len(train_dataset)):
                y = train_dataset.get_label(idx)
                weight[idx] = weight_per_class[y]
            weight = torch.DoubleTensor(weight)
            self.train_loader = DataLoader(train_dataset, batch_size=1, sampler=WeightedRandomSampler(weight, len(weight)), num_workers=args.num_workers)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=1, sampler=RandomSampler(train_dataset), num_workers=args.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=1, sampler=SequentialSampler(val_dataset), num_workers=args.num_workers)
        self.test_loader = DataLoader(test_dataset, batch_size=1, sampler=SequentialSampler(test_dataset), num_workers=args.num_workers)
        
        # model config
        self.n_cls = args.n_cls
        
        if args.model == 'ABMIL':
            self.model = AttnSurvMIL(in_feat_dim_he=args.in_feat_dim_he, in_feat_dim_imc=args.in_feat_dim_imc, hidden_feat_dim=256, out_feat_dim=256, dropout=args.drop_out, n_cls=self.n_cls, fusion=args.fusion)
        elif args.model == 'MCAT':
            self.model = MCATSurvMIL(in_feat_dim_he=args.in_feat_dim_he, in_feat_dim_imc=args.in_feat_dim_imc, hidden_feat_dim=256, out_feat_dim=256, dropout=args.drop_out, n_cls=self.n_cls, fusion=args.fusion)
        else:
            raise NotImplementedError

        # loss config
        if args.loss == 'ce_surv':
            self.loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
        elif args.loss == 'nll_surv':
            self.loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
        elif args.oss == 'cox_surv':
            self.loss_fn = CoxSurvLoss()
        else:
            raise NotImplementedError

        # optimizer config
        self.gc = args.gc # gradient accumulation
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.reg)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=args.scheduler_decay_rate, patience=args.scheduler_patience)
        
        # model saver config
        self.early_stop = args.early_stop
        self.save_metric = args.save_metric
        self.monitor = Monitor(save_path=self.save_path, metric=self.save_metric)
        
        # eval config
        self.sub = args.sub # number of sub-intervals for concordance index
        self.bins = args.bins # list of time points for defining intervals

    def _print_model(self):
        num_params = 0
        num_params_train = 0
        print(self.model)
        
        for param in self.model.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        
        print('Total number of parameters: %d' % num_params)
        print('Total number of trainable parameters: %d' % num_params_train)

    def _train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        train_loss = 0.
        
        all_risk_scores = np.zeros((len(self.train_loader)))
        all_censorships = np.zeros((len(self.train_loader)))
        all_event_times = np.zeros((len(self.train_loader)))

        for batch_idx, batch in tqdm(enumerate(self.train_loader)):
            feats = (batch[0][0].to(self.device), batch[0][1].to(self.device))
            label = batch[1].to(self.device)
            event_time = batch[2].to(self.device)
            c = batch[3].to(self.device)

            # forward pass
            hazards, S, _ = self.model(feats)

            # loss
            loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
            train_loss += loss.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            
            loss /= self.gc
            loss.backward()
            
            if (batch_idx + 1) % self.gc == 0: 
                # optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

        train_loss /= len(self.train_loader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        
        # train summary
        train_summary = {
            "train_loss": train_loss,
            "train_c_index": c_index
        }
    
        return train_summary
    
    def _val_one_epoch(self, epoch):
        self.model.eval()

        val_loss = 0.
        
        all_risk_scores = np.zeros((len(self.val_loader)))
        all_censorships = np.zeros((len(self.val_loader)))
        all_event_times = np.zeros((len(self.val_loader)))

        for batch_idx, batch in tqdm(enumerate(self.val_loader)):
            feats = (batch[0][0].to(self.device), batch[0][1].to(self.device))
            label = batch[1].to(self.device)
            event_time = batch[2].to(self.device)
            c = batch[3].to(self.device)

            # forward pass
            with torch.no_grad():
                hazards, S, _ = self.model(feats)

            # loss
            loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
            val_loss += loss.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
        
        val_loss /= len(self.val_loader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        
        # val summary
        val_summary = {
            "val_loss": val_loss,
            "val_c_index": c_index
        }
        
        print('*** VALIDATION ***')
        print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, c_index))
        
        return val_summary
    
    def train(self):    
        print("Start training...")
        for epoch in range(self.max_epochs):
            train_summary = self._train_one_epoch()
            val_summary = self._val_one_epoch(epoch)
            self.monitor(epoch, val_summary, self.model)
            if self.monitor.stop and self.early_stop:
                break
            self.scheduler.step(val_summary["val_loss"])
            train_summary["learning_rate"] = self.optimizer.param_groups[0]['lr']
            self.tb_logger.run(func_name="log_scalars", mode="tb", metric_dict=train_summary, step=epoch)
            self.tb_logger.run(func_name="log_scalars", mode="tb", metric_dict=val_summary, step=epoch)
            
        print("Training finished!")

    def test(self):
        
        if self.pretrained_path:
            print(f'Loading pretrained model from {self.pretrained_path}...')
            self.model.load_state_dict(torch.load(self.pretrained_path)) # test pretrained model if provided
        else:
            print(f'Loading best {self.save_metric} model from {os.path.join(self.save_path, f"model_best_{self.save_metric}.pt")}...')
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, f'model_best_{self.save_metric}.pt')))
        self.model.eval()

        all_risk_scores = np.zeros((len(self.test_loader)))
        all_censorships = np.zeros((len(self.test_loader)))
        all_event_times = np.zeros((len(self.test_loader)))
        all_survival = []
        
        slide_ids = self.test_loader.dataset.slide_data['slide_id']
        patient_results = {}

        for batch_idx, batch in tqdm(enumerate(self.test_loader)):
            feats = (batch[0][0].to(self.device), batch[0][1].to(self.device))
            label = batch[1].to(self.device)
            event_time = batch[2].to(self.device)
            c = batch[3].to(self.device)
            coords = batch[4]
            slide_id = batch[5][0]
            # slide_id = slide_ids.iloc[batch_idx]
            assert slide_id == slide_ids.iloc[batch_idx]

            # forward pass
            with torch.no_grad():
                hazards, S, _, A1, A2 = self.model(feats, output_attention=True)

            batch_size = hazards.size(0)
            hazards = hazards.view(-1, 1).repeat(1, self.sub).view(batch_size, -1).div(self.sub) # interpolate sub-intervals
            hazards = pad_col(hazards, where='start') # pad the first column with 1 --> survival = 1 for t=0
            all_survival.append(torch.cumprod(1 - hazards, dim=1).detach().cpu().numpy())
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time.cpu().numpy()
            
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk.item(), 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})
        
        all_survival = np.concatenate(all_survival, axis=0)
        n_patients = all_survival.shape[0]
        index = make_subgrid(self.bins, self.sub)
        all_survival_df = pd.DataFrame(all_survival.T, index=index, columns=range(0, n_patients))
        ev = EvalSurv(all_survival_df, all_event_times, (1-all_censorships).astype(int), censor_surv='km')
        
        ctd = ev.concordance_td('antolini')
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        
        print('*** TEST ***')
        print('Test time dependent c-Index: {:.4f}'.format(ctd))
        print('Test c-Index: {:.4f}'.format(c_index))
        
        save_pkl(os.path.join(self.save_path, 'results.pkl'), patient_results)
        
        return {
            "ctd": ctd,
            "c_index": c_index
        }