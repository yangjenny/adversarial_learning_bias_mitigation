## adversarial_training

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
run trainor.py "/path/to/JSONCONFIG"
```
