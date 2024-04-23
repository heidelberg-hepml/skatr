# skatr
A general-purpose transformer for processing lightcones observed by the Square Kilometer Array (SKA).

## Getting started

## Basic usage
The script `main.py` is used to run `experiment`s. An experiment consists of training/evaluating a model as well as creating plots. To run the default experiment simply run:
```
./main.py
```

The default experiment is `RegressionExperiment`, which trains a convolutional neural network (`CNN`) to predict cosmological parameters given a lightcone. To run a different experiment, or use a different neural network, simply add command line arguments:
```
./main.py experiment=<name of an experiment> net=<name of a network architecture>
```

The following is a description of currently-implemented experiments:
| Experiment name | Description | Compatible networks |
| :-------- | :------- | :------- |
| `regression` | Trains a network to predict cosmological parameters given a lightcone. The model performance is measured in terms of the mean squared error. | `cnn`, `vit` |

## Further options
This project uses [Hydra](https://hydra.cc/docs/intro/) to configure experiments. The default settings are given in `config/default.yaml` and each can be overridden via the command line. For example, to use a different learning rate and data split:
```
./main.py training.lr=1e-4 data.splits={train: 0.8, val: 0.1, test: 0.1}
```
Using the same syntax, one can also submit an experiment to run on a cluster, possibly with specific resources
```
./main.py submit=True
```
```
./main.py submit=True cluster.queue=<name of queue> cluster.mem=12gb
```

