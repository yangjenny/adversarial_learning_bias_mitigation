# Adversarial Learning for Bias Mitigation

This repository hosts the version of the code used for the publication ["An adversarial training framework for mitigating algorithmic biases in clinical machine learning"](https://www.nature.com/articles/s41746-023-00805-y).

## Dependencies

This will be updated soon.

## Getting Started

### adversarial_training_framework.py

This file contains the Adv_Model class and performance metrics.

### trainer.py

This file takes a JSON configuration file as input and uses the Adv_Model class to train either a basic model or an adversarial model. An example of a JSON configuration file:

```
{
  "Xtrain": "data/X_train.pkl",
  "Xvalid": "data/X_valid.pkl",
  "ytrain": "data/y_train.pkl",
  "yvalid": "data/y_valid.pkl",
  "ztrain": "data/Z_ethnicity_train.pkl",
  "zvalid": "data/Z_ethnicity_valid.pkl",
  "method": "adv",
  "num_classes": 2,
  "hyperparams": {
    "learning_rate": [1e-3],
    "num_iters": [4000],
    "num_nodes": [30],
    "num_nodes_adv": [10],
    "dropout_rate": [0.3],
    "alpha": [1]
  }
}
```
To train a model, use:
```
run trainer.py "/path/to/JSONCONFIG"
```

## Citation

If you found our work useful, please consider citing:

Yang, J., Soltan, A. A., Eyre, D. W., Yang, Y., & Clifton, D. A. (2023). An adversarial training framework for mitigating algorithmic biases in clinical machine learning. NPJ Digital Medicine, 6(1), 55.
