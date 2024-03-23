# Contains functions to train pytorch models as
# well as scikit-learn models.
import os
import json
import warnings
import math
import time
from pathlib import Path
from os.path import join
from os.path import isfile
from os.path import basename
from itertools import compress
from collections import defaultdict

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.utils import parallel_backend
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from IPython import embed

import pytorch_lightning as pl

# Define indices for data
idx_ts = 0 # Time Series
idx_lengths = 1 # Lengths
idx_labels = 2 # Gold standard labels
idx_exer_stress = 3 # Indicator of exercise stress (vs. pharma stress)
idx_pat_ids = 4 # patient ids
idx_clin = 5 # Clinical data
idx_qrs_labels = 6 # QRS labels
idx_pheno_labels = 7 # Stress type labels
idx_channel_labels = 8 # channel labels
idx_ext_pheno_labels = 9 # Extended stress type labels labels
MPSSSS_labels = 10 # MPSSSS labels
MPSSRS_labels = 11 # MPSSRS labels
MPSSDS_labels = 12 # MPSSDS labels

class LitModel(pl.LightningModule):
    def __init__(self, model, permute_last_dim, clinical_features, aux_tasks,
                optimizer_func, lr, wd, agg):
        super().__init__()
        self.model = model
        self.permute_last_dim = permute_last_dim
        self.clin_features = clinical_features
        self.aux_tasks = aux_tasks
        self.opt_func = optimizer_func
        self.lr = lr
        self.wd = wd
        self.agg = agg
        self.clean_input = False

    def prepare_main_data(self, data, label_smoothing=False):
        ts_seq = data[idx_ts]

        if self.permute_last_dim:
            ts_seq = ts_seq.permute(0, 2, 1)
        ts_seq = ts_seq.to(self.device, dtype=torch.float)

        lengths = data[idx_lengths]
        lengths = lengths.to(self.device, dtype=torch.int)
        stress_type_ind = data[idx_exer_stress]
        stress_type_ind = stress_type_ind.to(self.device, dtype=torch.int)
        pat_ids = data[idx_pat_ids]

        y = data[idx_labels]
        y = y.to(self.device, dtype=torch.float)

        return ts_seq, lengths, y, stress_type_ind, pat_ids

    def forward(self, data):
        # Prepare data
        if not self.clean_input:
            ts_seq, lengths, y, stress_ind, pat_ids = \
                    self.prepare_main_data(data)

            clin_data = None
            if self.clin_features:
                clin_data = data[idx_clin].to(self.device, torch.float).squeeze(1)
        else:
            # This is for SHAP only where we assume that clin data is
            # concatenated
            ts_seq = data[:, :-8].unsqueeze(1)
            clin_data = data[:, -8:]  

        # Gather auxiliary task labels
        #aux_task_labels = self.get_aux_task_labels(self.aux_tasks, data)

        # forward pass

        y_hat, _, _ = self.model([ts_seq, clin_data])
        if not self.clean_input:
            return y_hat
        else:
            return y_hat[0]

    def configure_optimizers(self):
        optimizer = self.opt_func(self.parameters(), lr=self.lr,
                                  weight_decay=self.wd)
        return optimizer

    @staticmethod
    def _group_by_id(id_lengths, data, func=torch.max):
        vs = torch.split(data, tuple(id_lengths))
        return torch.stack([func(v) for v in vs])

    def training_step(self, batch, batch_idx):
        # Prepare data
        return_data = {}
        ts_seq, lengths, y, stress_ind, pat_ids = \
                self.prepare_main_data(batch)

        return_data.update({'stress_ind': stress_ind,
                            'pat_ids': torch.from_numpy(pat_ids.astype(int))})

        clin_data = None
        if self.clin_features:
            clin_data = batch[idx_clin].to(self.device, torch.float).squeeze(1)

        # Gather auxiliary task labels
        aux_task_labels = self.get_aux_task_labels(self.aux_tasks, batch)

        # forward pass
        y_hat, l_funcs, l_weights = self.model([ts_seq, clin_data])
        
        # Record all losses
        total_class_loss = []
        total_loss = []

        # If wanted aggregate predictions over patients
        if self.agg:
            idx, pat_counts = torch.unique_consecutive(torch.Tensor([pat_ids]),
                                     return_counts=True)
            y_hat_grouped = self._group_by_id(pat_counts, y_hat[0], torch.max)
            y_grouped = self._group_by_id(pat_counts, y, torch.mean)

            y_hat[0] = y_hat_grouped.unsqueeze(1)
            y = y_grouped

        loss = l_weights[0] * l_funcs[0](y_hat[0], y.unsqueeze(1))
        total_class_loss = torch.tensor([[loss.item()]])
        for i in range(1, len(self.model.output_tasks)):
            task_name = self.model.output_tasks[i]
            task_pred = y_hat[i].squeeze(1)
            task_truth = aux_task_labels[task_name]
            task_loss = l_weights[i] * l_funcs[i](task_pred, task_truth)

            loss += task_loss

            return_data[f'{task_name}_pred'] = task_pred
            return_data[f'{task_name}_truth'] = task_truth
            
        total_loss = torch.tensor([[loss.item()]])
        class_pred = y_hat[0].squeeze(1)
        
        return_data.update({'loss': loss, 'class_pred': y_hat[0],
                       'class_truth': y.unsqueeze(1),
                       'total_class_loss': total_class_loss})
        return return_data

    def training_epoch_end(self, outputs):
        train_results = {}
        for k in outputs[0].keys():
            if k != 'loss':
                res = torch.cat([o[k] for o in outputs])
            else:
                res = torch.cat([torch.Tensor([[o[k].item()]]) for o in outputs])
            train_results[k] = res
        self._log_results(train_results['class_pred'],
                          train_results['class_truth'],
                          train_results['stress_ind'],
                          train_results['total_class_loss'],
                          train_results['loss'], 'train')

    def validation_step(self, batch, batch_idx):
        # Prepare data
        return_data = {}
        ts_seq, lengths, y, stress_ind, pat_ids = \
                self.prepare_main_data(batch)

        return_data.update({'stress_ind': stress_ind,
                            'pat_ids': torch.from_numpy(pat_ids.astype(int))})

        clin_data = None
        if self.clin_features:
            clin_data = batch[idx_clin].to(self.device, torch.float).squeeze(1)

        # Gather auxiliary task labels
        aux_task_labels = self.get_aux_task_labels(self.aux_tasks, batch)

        # forward pass
        y_hat, l_funcs, l_weights = self.model([ts_seq, clin_data])
        
        # Record all losses
        total_class_loss = []

        # If wanted aggregate predictions over patients
        if self.agg:
            idx, pat_counts = torch.unique_consecutive(torch.Tensor([pat_ids]),
                                     return_counts=True)
            y_hat_grouped = self._group_by_id(pat_counts, y_hat[0], torch.max)
            y_grouped = self._group_by_id(pat_counts, y, torch.mean)

            y_hat[0] = y_hat_grouped.unsqueeze(1)
            y = y_grouped

        loss = l_weights[0] * l_funcs[0](y_hat[0], y.unsqueeze(1))
        total_class_loss = torch.tensor([[loss.item()]])
        for i in range(1, len(self.model.output_tasks)):
            task_name = self.model.output_tasks[i]
            task_pred = y_hat[i].squeeze(1)
            task_truth = aux_task_labels[task_name]
            task_loss = l_weights[i] * l_funcs[i](task_pred, task_truth)

            loss += task_loss

            return_data[f'{task_name}_pred'] = task_pred
            return_data[f'{task_name}_truth'] = task_truth
            
        total_loss = torch.tensor([[loss.item()]])
        return_data.update({'loss': total_loss, 'class_pred': y_hat[0],
                            'class_truth': y.unsqueeze(1),
                            'total_class_loss': total_class_loss})
        return return_data

    def validation_epoch_end(self, outputs):
        val_results = {}
        for k in outputs[0].keys():
            res = torch.cat([o[k] for o in outputs])
            val_results[k] = res
        self._log_results(val_results['class_pred'],
                          val_results['class_truth'],
                          val_results['stress_ind'],
                          val_results['total_class_loss'],
                          val_results['loss'], 'val')


    def test_step(self, batch, batch_idx):
        # Prepare data
        return_data = {}
        ts_seq, lengths, y, stress_ind, pat_ids = \
                self.prepare_main_data(batch)

        return_data.update({'stress_ind': stress_ind,
                            'pat_ids': torch.from_numpy(pat_ids.astype(int))})

        clin_data = None
        if self.clin_features:
            clin_data = batch[idx_clin].to(self.device, torch.float).squeeze(1)

        # Gather auxiliary task labels
        aux_task_labels = self.get_aux_task_labels(self.aux_tasks, batch)

        # forward pass
        y_hat, l_funcs, l_weights = self.model([ts_seq, clin_data])
        
        # Record all losses
        loss = l_weights[0] * l_funcs[0](y_hat[0], y.unsqueeze(1))
        for i in range(1, len(self.model.output_tasks)):
            task_name = self.model.output_tasks[i]
            task_pred = y_hat[i].squeeze(1)
            task_truth = aux_task_labels[task_name]
            task_loss = l_weights[i] * l_funcs[i](task_pred, task_truth)

            loss += task_loss

            return_data[f'{task_name}_pred'] = task_pred
            return_data[f'{task_name}_truth'] = task_truth
            
        sig = nn.Sigmoid()
        class_pred = sig(y_hat[0])
        return_data.update({'loss': loss, 'class_pred': class_pred,
                            'class_truth': y.unsqueeze(1)})
        return return_data

    def test_epoch_end(self, outputs):
        # We should combine all outputs here!
        self.test_results = outputs


    def _log_results(self, class_pred, y, stress_ind, total_class_loss,
                     total_loss, split='train'):
        # As we use BCEWithLogits, we need to use a sigmoid layer to
        # make actual predictions.
        sig = nn.Sigmoid()
        class_pred = sig(class_pred)

        # calculate accuracy
        correct_predictions = torch.eq(class_pred > 0.5, y).sum().item()
        total_samples = y.float().size()[0]

        all_predictions = class_pred.tolist()
        all_labels = y.tolist()
        all_stress_type_indicator = stress_ind.tolist()

        # Calculate performance measures
        accuracy = correct_predictions / total_samples
        auprc, auroc, auprc_exer, auroc_exer, auprc_pharma, auroc_pharma = \
                _get_performance_metrics(all_predictions, all_labels,
                                         all_stress_type_indicator,
                                         split=='test')

        mean_class_loss = torch.mean(total_class_loss)
        mean_total_loss = torch.mean(total_loss)
        self.log_dict({f'class_loss_{split}': mean_class_loss,
                       f'acc_{split}': accuracy,
                       f'auroc_exer_{split}': auroc_exer,
                       f'auprc_exer_{split}': auprc_exer,
                       f'auroc_pharma_{split}': auroc_pharma,
                       f'auprc_pharma_{split}': auprc_pharma},
                     on_step=False, on_epoch=True, prog_bar=False)

        self.log_dict({f'total_loss_{split}': mean_total_loss,
                       f'auprc_{split}': auprc,
                       f'auroc_{split}': auroc},
                     on_step=False, on_epoch=True, prog_bar=True)

    def get_aux_task_labels(self, aux_tasks, data):
        aux_task_idx_dict = {'PhenoPred': 7, 'ChannelPred': 8,
                             'QRSPred': 6, 'ExtPhenoPred': 9,
                             'MPSSSSPred': 10, 'MPSSRSPred': 11,
                             'MPSSDSPred': 12}
        
        aux_task_labels = {}
        for aux_task in aux_tasks:
            data_idx = aux_task_idx_dict[aux_task['name']]
            task_data = data[data_idx]
            aux_task_labels[aux_task['name']] = task_data.to(self.device,
                                                             aux_task['dtype'])

        return aux_task_labels

def _get_performance_metrics(all_predictions,
                             all_labels, all_stress_type_indicator, tmp=False):
    all_stress_type_indicator = np.asarray(all_stress_type_indicator)
    all_exer_labels = list(compress(all_labels, all_stress_type_indicator))
    all_exer_preds = list(compress(all_predictions, all_stress_type_indicator))
    all_pharma_labels = list(compress(all_labels, 1 - all_stress_type_indicator))
    all_pharma_preds = list(compress(all_predictions,
                                1 - all_stress_type_indicator))

    # Compute AUPRCs
    try:
        auprc = average_precision_score(all_labels, all_predictions)
    except:
        auprc = 0.0
    try:
        auprc_exer = average_precision_score(all_exer_labels, all_exer_preds)
    except:
        auprc_exer = 0.0
    try:
        auprc_pharma = average_precision_score(all_pharma_labels, all_pharma_preds)
    except:
        auprc_pharma = 0.0

    # Compute AUROCs
    try:
        auroc = roc_auc_score(all_labels, all_predictions)
    except:
        auroc = 0.0
    try:
        auroc_exer = roc_auc_score(all_exer_labels, all_exer_preds)
    except:
        auroc_exer = 0.0
    try:
        auroc_pharma = roc_auc_score(all_pharma_labels, all_pharma_preds)
    except:
        auroc_pharma = 0.0

    return auprc, auroc, auprc_exer, auroc_exer, auprc_pharma, auroc_pharma
