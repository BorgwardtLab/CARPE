import os
import pickle
from glob import glob
from pathlib import Path
from joblib import load
from os.path import join, basename
import warnings
warnings.filterwarnings('ignore')


import torch
from torch import nn
from torch.optim import AdamW
from torchmtl import MTLModel
from torchmtl.wrapping_layers import SimpleSelect
from torchmtl.wrapping_layers import Concat

# CARPE-specific imports
import model_builder_reduced
from model_builder_reduced import get_eimi_layers
from model_builder_reduced import get_clin_layers
from model_builder_reduced import get_MPSSXS_layers
from model_builder_reduced import get_ext_pheno_layers
from Training_Lit import LitModel

class RMSELoss(nn.Module):
    def __init__(self, reduction):
        super(RMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, X, Y):
        return torch.sqrt(self.loss(X, Y))

def get_checkpoints(logdir='', run_name=None):
    log_path = f'checkpoints/{logdir}/{run_name}'
    checkpoints = {}
    for i in range(5):
        checkpoint_file = join(os.environ['ECGDATAROOT'], log_path,
                               f'model_it_{i}*.ckpt')
        file = glob(checkpoint_file)[0]
        checkpoints[i] = basename(file)
    return checkpoints

def load_model_args(logdir, run_name):
    log_path = f'checkpoints/{logdir}/{run_name}'

    with open(join(os.environ['ECGDATAROOT'], log_path, 'args.pkl'), 'rb') as f:
        model_args = pickle.load(f)
    model_args.data_config = join(os.environ['ECGDATAROOT'],
                                  '/'.join(model_args.data_config.split('/')[-3:]))

    with open(join(os.environ['ECGDATAROOT'], log_path, 'model_config.pkl'), 'rb') as f:
        model_config = pickle.load(f)
    with open(join(os.environ['ECGDATAROOT'], log_path, 'aux_tasks.pkl'), 'rb') as f:
        aux_tasks = pickle.load(f)

    return model_args, model_config, aux_tasks

def _get_mtl_tasks(model_config, args, ts_length=5000):
    """ This sets up the MTLModel from torchmtl. Currently only the tasks of the
    best model are supported."""

    # Load our main model (currently Conv_Resnet in module model_builder_reduced)
    model = getattr(model_builder_reduced, model_config['base_model'])
    model = model(model_config, ts_length, args)
    shared_layer_out_dim = model.penultimate_input_dim

    if model_config['clinical_features']:
        clin_features_out = 32
        clin_features_in = model_config['clin_dim']
        sharing_layer = 'CombLayer'
    else:
        clin_features_out = 0
        sharing_layer = 'ResNet'

    mtl_tasks = [
        {
            'name': "selectTS",
            'layers': SimpleSelect(selection_axis=0),
        },
        {
            'name': "ResNet",
            'layers': model,
            'anchor_layer': 'selectTS'
        },
        {
            'name': "EIMIPred",
            'layers': get_eimi_layers(shared_layer_out_dim + clin_features_out, 1),
            'loss_weight': 1.0,
            'loss': nn.BCEWithLogitsLoss(),
            'anchor_layer': sharing_layer,
        }
    ]
    output_tasks = ['EIMIPred'] # This is always the first output tasks

    if model_config['clinical_features']:
        mtl_tasks.append(
            {
                'name': "selectClin",
                'layers': SimpleSelect(selection_axis=1),
            }
        )
        mtl_tasks.append(
            {
                'name': "clinLayer",
                'layers': get_clin_layers(clin_features_in, clin_features_out),
                'anchor_layer': 'selectClin'
            }
        )
        mtl_tasks.append(
            {
                'name': "CombLayer",
                'layers': Concat(dim=1),
                'anchor_layer': ['clinLayer', 'ResNet']
            }
        )

    if args.MPSSSS_loss_reg:
        mtl_tasks.append(
            {
                'name': "MPSSSSPred",
                'layers': get_MPSSXS_layers(shared_layer_out_dim + clin_features_out, 1),
                'loss_weight': args.MPSSSS_loss_reg,
                'loss': RMSELoss(reduction='mean'),
                'anchor_layer': sharing_layer,
            }
        )
        output_tasks.append('MPSSSSPred')
    if args.MPSSRS_loss_reg:
        mtl_tasks.append(
            {
                'name': "MPSSRSPred",
                'layers': get_MPSSXS_layers(shared_layer_out_dim + clin_features_out, 1),
                'loss_weight': args.MPSSRS_loss_reg,
                'loss': RMSELoss(reduction='mean'),
                'anchor_layer': sharing_layer,
            }
        )
        output_tasks.append('MPSSRSPred')
    if args.ext_pheno_loss_reg:
        mtl_tasks.append(
            {
                'name': "ExtPhenoPred",
                'layers': get_ext_pheno_layers(shared_layer_out_dim + clin_features_out, 5),
                'loss_weight': args.ext_pheno_loss_reg,
                'loss': nn.CrossEntropyLoss(reduction='mean'),
                'anchor_layer': sharing_layer,
            }
        )
        output_tasks.append('ExtPhenoPred')
    return mtl_tasks, output_tasks

class CARPE_ECG_model:

    def __init__(self, split): 
        # Set required environment variable to workdir
        working_dir = Path(os.getcwd())
        data_root = working_dir.parent.absolute()
        os.environ['ECGDATAROOT'] = str(data_root)

        # Load model checkpoints
        checkpoints = get_checkpoints(logdir='', run_name='')
        model_args, model_config, aux_tasks = load_model_args(logdir='', run_name='')

        # Load CARPE_ECG checkpoint
        model_build_func = MTLModel

        # Create a CARP_ECG model. This can be used as a starting point to train
        # the multi-task architecture on your data.
        model_mtl = model_build_func(*_get_mtl_tasks(model_config, model_args))
        lit_model = LitModel(model_mtl, model_config['permute_last_dim'],
                             model_config['clinical_features'],
                             aux_tasks, AdamW, model_args.lr,
                             model_args.weight_decay, False)

        # Load the model checkpoint.
        model = lit_model.load_from_checkpoint(f'../checkpoints/{checkpoints[split]}',
                                               model=model_mtl,
                                               permute_last_dim=model_config['permute_last_dim'],
                                               clinical_features=model_config['clinical_features'],
                                               aux_tasks=aux_tasks,
                                               optimizer_func=AdamW,
                                               lr=model_args.lr,
                                               wd=model_args.weight_decay,
                                               agg=False)
        model.to("cpu")
        model.eval()
        model.clean_input = True # To work without PyTorch Dataloader
        self.model = model
        self.mtl_model = model_mtl
    
    def predict(self, x):
        with torch.no_grad():
            preds = torch.sigmoid(self.model(torch.Tensor(x).to(dtype=torch.float32, device='cpu')))

        return preds

class CARPE_Clin_model:

    def __init__(self, split):
        self.model = load(f'../models/RF_{split}.joblib')

    def predict(self, x):
        return self.model.predict_proba(x)