# skatr
A general-purpose summary transformer for processing lightcones observed by the Square Kilometer Array (SKA).

## Getting started
- Clone the repository
- Create a virtual (conda) environment and install packages `requirements.txt`


## Basic usage
The script `main.py` is used to run experiments. Each experiment consists of training/evaluating a model as well as creating plots. For example, to run the default experiment, simply use:
```
python3 main.py
```
Experiment settings can be adjusted from the command line. For example, to use a different neural network, simply add command line arguments:
```
python3 main.py net=<network_config>
```
To change experiment, a different syntax is needed. The `--config_name` (or  just `-cn`) option should be set:
```
python3 main.py -cn <experiment_config>
```

The following is a description of currently-implemented experiments:
| Experiment name | Description | Compatible networks |
| :-------- | :------- | :------- |
| `pretraining` | Trains a network in a self-supervised manner in order to learn informative lightcone summaries. | `vit` |
| `regression` | Trains a network to predict cosmological parameters given a lightcone. The model performance is measured in terms of the NRMSE. Produces parameter recovery plots. | `cnn`, `vit` |
| `inference` | Trains a generative model for the posterior distribution for cosmological parameters conditioned on a given lightcone (Neural Posterior Estimation). Lightcones are summarised by a network before being passed to the generative model. Produces triangle posterior plots for a selection of test lightcones and a coverage calibration plot.| `mlp` |
## Further usage
This project uses [Hydra](https://hydra.cc/docs/intro/) to configure experiments. The default settings are given in `config/default.yaml` and each can be overridden via the command line. For example, to use a different learning rate and to load the entire dataset into memory:
```
./main.py training.lr=1e-4 data.file_by_file=False
```
Using the same syntax, one can also submit an experiment to run on a cluster, possibly with specific resources
```
./main.py submit=True
```
```
./main.py submit=True cluster.queue=<name_of_queue> cluster.mem=12gb
```
