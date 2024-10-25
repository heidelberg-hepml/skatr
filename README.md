# skatr
A general-purpose summary transformer for processing lightcones observed by the Square Kilometer Array (SKA).

Read the arXiv paper: [2410.18899](https://arxiv.org/abs/2410.18899)

## Getting started
- Clone the repository
- Create the conda environment:
```
conda env create -f env.yaml
```

## Basic usage
This project uses [Hydra](https://hydra.cc/docs/intro/) to configure experiments. The default settings are given in `config/default.yaml` and each can be overridden via the command line.

The script `main.py` is used to run experiments. Each experiment consists of training/evaluating a model as well as creating plots. For example, to run a regression experiment, simply use:
```
python main.py -cn regression data.dir=/path/to/light/cones
```
In the above, `-cn` is short for `--config_name`.

Experiment settings can be adjusted from the command line. For example, to select a different neural network use:
```
python main.py ... net=cnn
```

The following is a description of currently-implemented experiments:
| Experiment name | Description | Compatible networks |
| :-------- | :------- | :------- |
| `jepa` | Trains a network in a self-supervised manner in order to learn informative lightcone summaries. | `vit` |
| `regression` | Trains a network to predict simulation parameters given a light cone. The model performance is measured in terms of a normalized mean relative absolute error. Produces parameter recovery plots. | `cnn`, `vit` |
| `summarized_regression` | Runs a regression experiment based on summarized light cones. | `mlp` + `pretrained_vit` |
| `inference` | Trains a normalizing flow to fit the posterior distribution for simulation parameters conditioned on a given light cone (Neural Posterior Estimation). Light cones are summarized by a network before being passed to the generative model. Produces triangle posterior plots for a selection of test light cones and a coverage calibration plot. | `inn` |

## Specifying a summary network
The experiments `inference` and `summarized_regression` make use of a summary network. The default architecture for `inference` is `vit`, but this can be changed with:
```
python main.py -cn inference ... net@summary_net=cnn
```
In this case, a CNN will be trained jointly with the normalizing flow.

To initialise the summary network from a pretrained `vit`, use:
```
python main.py ... net@summary_net=pretrained_vit summary_net.backbone_dir=/path/to/pretraing/exp
```
By default, the weights will be loaded then frozen.

When using a summary network, there is also the option to create a summarized dataset:
```
python main.py ... data.summarize=True
```
However, it only makes sense to do this if the summary network is pretrained.


## Continuing an experiment
One often needs to re-run a previous experiment. This can be achieved simply from the command line. Common examples include:

- Continuing training from a saved checkpoint:
```
python main.py -cn <prev_config_name> prev_exp_dir=/path/to/prev/exp training.warm_start=True  
python main.py ... training.warm_start_epoch=30
```
- Repeating evaluation and/or plotting from using a saved model:
```
python3 main.py -cn <prev_config_name> prev_exp_dir=/path/to/prev/exp train=False evaluate=False 
```
The specific configuration will be loaded from the previous experiment. Command line overrides are also applied.

## Further settings
The following is a description of parameters in `config/default.yaml` that may not be self explanatory or can affect performance.
| Parameter name | Description |
| :------- | :------- |
| `data.file_by_file` | Whether to save memory by reading each light cone from disk when forming a batch. |
|  `data.summarize` | Whether to summarize light cones upfront, to avoid repeated calls of the summary network. |
| `submit` | Whether or not to submit the experiment as a job on a cluster. Submission configuration is controlled through the `cluster` parameter group. |
