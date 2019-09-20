# Ensemblizer

**ensemblizer** is a small package for various ensemble methods

**ModelCollection** is a simple way to aggregate models so that they can be trained together.

**CatEnsemble** takes a **ModelCollection** object (or the same input as one) with an additional ensemble model.  The ensemble model will train on the predicted probabilities (or just predictions) of the **ModelCollection** and make a separate prediction.  This ensemble method allows for different weighting schemes of the outputs of each model as well as stacking the original data with the predictions when training the ensemble model.  Additionally, this ensemble class allows you to pretrain the **ModelCollection** object prior to training the ensemble model or training the **ModelCollection** object *with* the ensemble model.  This allows for more efficient hyperparameter tuning (albeit much longer training and tuning times).

Hyperparameters can be trained like any other scikit-learn-esque model.  When setting parameters for the ensemble model, parameters that begin with *name__* will set the hyperparameter of the *name* model in the collection or ensemble model (default name for ensemble model is "ensemble").  Parameters that start with *__name* will update the weight of the *name* model in the collection.  This allows the weighting scheme of the ensemble to be tuned with all other parameters using any scikit-learn tuning package.

## Current Version is v0.04

This package is currently in the beginning stages but future work is planned.

## Installation

	pip install ensemblizer

## Usage
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from ensemblizer import ModelCollection, CatEnsemble

np.random.seed(0)
x_train = np.random.randint(0, 10, (80, 3))
x_test = np.random.randint(0, 10, (20, 3))
y_train = np.random.randint(0, 2, 80)
y_test = np.random.randint(0, 2, 20)
	
models = ModelCollection([('log', LogisticRegression(random_state=0)),('nb', MultinomialNB())])
test.fit(x_train, y_train)
	
ensemble = CatEnsemble(test, KNeighborsClassifier())
ensemble.fit(x_train, y_train)
test_preds = ensemble.predict(x_test)
print(f"Accuracy on test set is {accuracy_score(y_test, test_preds)}"
	
#change the C param of the 'log' model to 15, the alpha param of the 'nb' model to 1,
#the n_neighbors param of the ensemble model to 10, and the weight of the 'log' model to 3  
ens.set_params('log__C': 15, 'nb__alpha': 1, 'ensemble__n_neighbors': 10, '__log': 3})
ens.fit(x_train, y_train)
test_preds = ensemble.predict(x_test)
print(f"Accuracy on test set is {accuracy_score(y_test, test_preds)}"
```

## Known Bugs

A major known bug right now is that CatEnsemble only works with scikit-learn's GridSearchCV and RandomizedSearchCV with deep_train=True.  This is due to the way that scikit-learn copies estimators before resetting parameters for each parameter combination.  This causes the models stored in the ModelCollection object to reset to untrained status.  I am currently working on refactoring the class to be completely compatible.  It is currently compatible with all tuneRs.
	
## Future Plans

The next step is to create an ensemble regressor model.

## License

Lol

## Known Bugs

A major known bug right now is that CatEnsemble only works with scikit-learn's GridSearchCV and RandomizedSearchCV with deep_train=True.  This is due to the way that scikit-learn copies estimators before resetting parameters for each parameter combination.  This causes the models stored in the ModelCollection object to reset to untrained status.  I am currently working on refactoring the class to be completely compatible.  It is currently compatible with all tuneRs.
	
