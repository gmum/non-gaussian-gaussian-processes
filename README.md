Official implementation of the paper:

#### *Non-Gaussian Gaussian Processes for Few-Shot Regression*

Full text of the paper could be found on arXiv (https://arxiv.org/abs/2110.13561) and NeurIPS Proceedings 2021 (https://proceedings.neurips.cc/paper/2021/hash/54f3bc04830d762a3b56a789b6ff62df-Abstract.html). 

## Overview

*Gaussian Processes (GPs) have been widely used in machine learning to model distributions over functions, with applications including multi-modal regression, time-series prediction, and few-shot learning. GPs are particularly useful in the last application
since they rely on Normal distributions and, hence, enable closed-form computation of the posterior probability function.
Unfortunately, because the resulting posterior is not flexible enough to capture complex distributions, GPs assume high similarity between subsequent tasks -- a~requirement rarely met in real-world conditions.
In this work, we address this limitation by leveraging the flexibility of Normalizing Flows to modulate the posterior predictive distribution of the GP. This makes the GP posterior locally non-Gaussian, therefore we name our method Non-Gaussian Gaussian Processes (NGGPs). 
We propose an invertible ODE-based mapping that operates on each component of the random variable vectors and shares the parameters across all of them. 
We empirically tested the flexibility of NGGPs on various few-shot learning regression datasets, showing that the mapping can incorporate context embedding information to model different noise levels for periodic functions.
As a result, our method shares the structure of the problem between subsequent tasks, but the contextualization allows for adaptation to dissimilarities.
NGGPs outperform the competing state-of-the-art approaches on a diversified set of benchmarks and applications.*

## Requirements
All necessary libraries are in `environment.yml`. To create the environment enter the main directory of the repository and use:
```
conda env create -n env_name -f environment.yml
```
For the `NASDAQ` and `EEG` experiments please use the `environment_nasdaq.yml` specification instead. It uses a newer version of gpytorch needed for adding jitter noise for stability. Other experiments use the same gpytorch version as in https://github.com/BayesWatch/deep-kernel-transfer](https://github.com/BayesWatch/deep-kernel-transfer) 


## Running the code

Exemplary `NGGP` usage:

```
source activate object_tracking

python run_regression.py \
--dataset <dataset_name> \
--model=<model> \
--method="NGGP" \
--output_dim=40 \
--seed=1 \
--save_dir ./save/nggp_rbf" \
--kernel_type rbf \
--stop_epoch 50000 \
--all_lr 1e-3 \
--use_conditional True \
--context_type backbone 
```

where `<dataset_name>` is the name of the dataset, and `<model>` is the architecture of the used backbone. Possible choices and description of the parameters can be obtained by running:  

```
python run_regression.py --help
```

In order to change the save directory set `--save_dir <your_save_dir>`.


## Datasets:

**sines** - the sines dataset, adapted from [https://github.com/BayesWatch/deep-kernel-transfer](https://github.com/BayesWatch/deep-kernel-transfer). The dataset is generated "on-the-fly". See `data/data_generator.py`. In order to select this dataset pass `--dataset sines`.    

**QMUL** - The Queen Mary University of London multiview face dataset \[1\]. The dataset need to be downloaded by running `./filelists/QMUL/download_QMUL.sh`. See the instructions at [https://github.com/BayesWatch/deep-kernel-transfer](https://github.com/BayesWatch/deep-kernel-transfer). In order to select this dataset pass `--dataset qmul`.     

**Objects** - The Object Pose Prediction dataset from \[3\]. Refer to [https://github.com/google-research/google-research/tree/master/meta_learning_without_memorization](original repository of paper \[3\]) for inystuctions on how to obatin the dataset. In order to select this dataset pass `--dataset objects`. On how to adapt it for NGGP please refer to out paper.    

**NASDAQ** - The NASDAQ100 dataset \[2\]. It needs to be downloaded by running `./filelists/Nasdaq_100/nasdaq_100_padding.sh`. In order to select this dataset pass `--dataset nasdaq`.     

**EEG** - The EEG dataset \[4\]. It needs to be downloaded. See the link under https://archive.ics.uci.edu/ml/datasets/EEG+Steady-State+Visual+Evoked+Potential+Signals. Note: we use only one signal named `A001SB1_1.csv`. In order to select this dataset pass `--dataset eeg`.    


***NOTE***: The default paths for each dataset are stored in `training/configs.py`. Change this configuration if you want to store the dataset somewhere else. 


## Experiments


To run the experiments on a given dataset enter `./scripts/train_regression_<dataset_name>.sh <kernel_type>`. To evaluate the model run `./scripts/test_regression_<dataset_name>.sh <kernel_type>`.   

Note: For the sines datset the above script will run the experiment with standard gaussian noise and will perform the evaluation  both on the in-range and out-of-range data. The out-of-range results are saved with an `outr` suffix. In order to change the noise type to heteroscedastic set the `--noise` parameter to `hetero_multi`. 


## Acknowledgements

This repository is a fork of: [https://github.com/BayesWatch/deep-kernel-transfer](https://github.com/BayesWatch/deep-kernel-transfer)

## Bibliography

\[1\]. Shaogang Gong, Stephen McKenna, and John J Collins. An investigation into face pose distributions. In
Proceedings of the Second International Conference on Automatic Face and Gesture Recognition, pages
265–270. IEEE, 1996

\[2\] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, and Garrison Cottrell. A dual-stage
attention-based recurrent neural network for time series prediction, 2017. Accessed: 2021-05-25.

\[3\] Mingzhang Yin, George Tucker, Mingyuan Zhou, Sergey Levine, and Chelsea Finn. Meta-learning without
memorization. arXiv preprint arXiv:1912.03820, 2019.

\[4\] SM Fernandez-Fraga, MA Aceves-Fernandez, JC Pedraza-Ortega, and JM Ramos-Arreguin. Screen task
experiments for eeg signals based on ssvep brain computer interface. International Journal of Advanced
Research, 6(2):1718–1732, 2018. Accessed: 2021-05-25
