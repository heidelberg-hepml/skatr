# skatr
A general-purpose transformer for processing lightcones observed by the Square Kilometer Array (SKA).

## Getting started

## Usage
The script `main.py` is used to run `experiment`s. An experiment consists of training/evaluating a model as well as creating plots. To run the default experiment simply run:
```
python3 main.py
```

The default experiment is `RegressionExperiment`, which trains a convolutional neural network (`CNN`) to predict cosmological parameters given a lightcone. To run a different experiment, or use a different neural network, simply add command line arguments:
```
python3 main.py experiment=<name of an experiment> net=<name of a network architecture>
```

The following is a description of currently-implemented experiments:
| Experiment name | Description | Compatible networks |
| :-------- | :------- | :------- |
| regression | Trains a network to predict cosmological parameters given a lightcone. The model performance is measured in terms of the mean squared error. | `cnn`, `vit` |

{::comment}
# Advanced usage
This project uses [Hydra](https://hydra.cc/docs/intro/) to configure
{:/comment}
