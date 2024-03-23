from glob import glob
import numpy as np
from pathlib import Path
from os.path import join, isfile

import torch
from torch.utils.data import Dataset

from THEW_helper import THEWPatientHelper
from THEW_helper import get_input_seq

class THEWDataset(Dataset):

    def __init__(self, prepro='preprocessed', clindata_path=None):
        ph = THEWPatientHelper(clindata_path)
        #data_path = f'/links/groups/borgwardt/Projects/IschemiaPrediction/data/anonymized/THEW/{prepro}'
        data_path = f'../THEW/{prepro}'
        
        self.labels = []
        self.X_clin = []
        self.X = []
        self.pat_ids = []

        for file in sorted(glob('../THEW/data/ECG/*.ecg')):
            ID = Path(file).stem
            if not ph.clin_data.query('ID == @ID').empty:
                if isfile(join(data_path, f'{ID}.ecg.npz')):
                    print("Trying to load", join(data_path, f'{ID}.ecg.npz'))
                    data = np.load(join(data_path, f'{ID}.ecg.npz'))
                    data = data['data']
                    ts_seq = get_input_seq(data, file, leads=[11])
                    clin_data = ph.get_NIP_data(ID)
                    for seq in ts_seq:
                        if seq.shape[1] == 5000:
                            self.X.append(seq)
                            self.pat_ids.append(ID)
                            self.labels.append(ph.get_label(ID))
                            self.X_clin.append(clin_data)
                else:
                    continue
            

    def __getitem__(self, idx):
        print(np.asarray(self.X[idx]).shape)
        #X_tensor = torch.from_numpy(np.asarray(self.X[idx])).permute(1,0)
        X_tensor = np.asarray(self.X[idx]).permute(1,0)
        results = [X_tensor, 5000, self.labels[idx], 0, self.pat_ids[idx]]
        results += [self.X_clin[idx]]

        results += [None]
        results += [None]
        results += [None]
        results += [None]
        results += [None]
        results += [None]

        return results

    def __len__(self):
        return len(self.labels)

