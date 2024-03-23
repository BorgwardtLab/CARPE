# Coronary ARtery disease PrEdictor (CARPE)
<p align="center">
  <img src="https://github.com/BorgwardtLab/CARPE/blob/main/logo.png?raw=true" width="200" title="CARPE Logo"><br/>
  This is the official repository for the paper <br/> <a href="https://link">Enhancing the diagnosis of functionally relevant coronary artery disease with machine learning</a>. <br/>
  <img src="https://img.shields.io/badge/python-3.8-green.svg">
</span>
</p>

## Prerequisites
Before checking out the repository, make sure you have [git lfs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) installed. This is necessary to be able to check out the model checkpoints exceeding 100MB due to github's file size limit.

Once `git lfs` is installed, check out the repository and install all dependencies with `pip install -r requirements.txt`. The code is tested with `Python 3.8`.

## Generating Predictions
Take a look at our [sample notebook](https://github.com/BorgwardtLab/CARPE/blob/main/CARPE/src/sample_prediction_generation.ipynb) to learn how to use $CARPE_{\text{Clin.}}$, our random forest trained on a small set of static clinical data, and our neural network approach $CARPE_{\text{ECG}}$ which takes both ECG signals and static date as inputs.

## Data Preprocessing
To preprocess your custom ECG signals, you will have to write your own data loader depending on your file format. We recommend inheriting from [`THEWParser` ](link). The main function you have to implement is `_get_raw` which loads the raw ECG signal according to your data format. Loading should result in a `numpy` array of dimensions `[T, num_leads]`, where $T$ is the length of the signal. The sampling rate of you signal should be either 500Hz or 1000Hz (take a look at the paper for more details). Once you can load your data into your custom parser the following code snippet applies all preprocessing steps that we used in the manuscript.

```python
import numpy as np
parser = THEWParser(filepath) # Replace with your parser

# Preprocess
band = [0.05, 150.0]
parser.apply_butter(parser.data, [band[0]/(parser.freq/2), band[1]/(parser.freq/2)])
parser.apply_median(parser.data)
parser.apply_smoothing(parser.data)
parser.apply_winsorizing(parser.data, 0.05, 100 - 0.05)

downsampled = signal.decimate(parser.data, 2, axis=0)

np.savez(OUTPUT_PATH, data=downsampled.T)
```

### 2-6-2 Sequence Extraction
You will likely not have access to the exact times when the stress phase started/ended. Instead, you can use the the time point of the maximum heart rate as the last point from which stress windows are extracted. To extract the first 2-6-2 sequence, take the first 2 seconds of the ECG signal, the last 6 seconds that preceed the time point with maximal HR, and the last 2 seconds of the ECG signal. For the second 2-6-2 sequence, stride the first window forward by 2 seconds, the second window back by 6 seconds, and the third window back by 2 seconds. Continue until you extracted all 2-6-2 sequences. 
Take a look at [this function](https://github.com/BorgwardtLab/CARPE/blob/main/CARPE/src/THEW_helper.py#L220) for an implementation. 
