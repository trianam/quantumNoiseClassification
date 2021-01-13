# quantumNoiseClassification
Source code for the data generation and experiments in paper "Machine learning approach for quantum non-Markovian noise classification". The paper can be found here: https://arxiv.org/abs/2101.03221 . If you found this code useful, please cite the paper with:
```
@misc{martina2021machine,
      title={Machine learning approach for quantum non-Markovian noise classification}, 
      author={Stefano Martina and Stefano Gherardini and Filippo Caruso},
      year={2021},
      eprint={2101.03221},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

### Prerequisites
* python 
* sklearn  
* pyTorch
* torchsummary  
* Ray Tune
* Hyperopt  
* tensorboardx

### Set mail notifier
If you are interested in setting the mail notifier, copy `notifierConfigTEMPLATE.py` in `notifierConfig.py` and compile it with data from a mail account. If you are not interested in the mail notifier, comment the corresponding line at the end of `tuneRunPytorch.py`. 

### Create dataset
Launch `./createDatasetScratch.py` and choose the desired dataset to create. Create both the data and the training/validation/test split.

### Run
The experiments to fill the paper tables are in `tuneConfigurations.py`. To launch a hyperparameter optimization run, use: `./tuneRunPytorch.py configNUM` where `configNUM` is one of the blocks in `tuneConfigurations.py`. After the run, to recover the best results (along validation set, reported on test set) use `./calculateTuneTestMetrics.py`.
