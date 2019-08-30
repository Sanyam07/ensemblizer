# Ensemblizer

**ensemblizer** is a small package for various ensemble methods

**ModelCollection** is a simple way to aggregate models so that they can be trained together.

**CatEnsemble** takes a **ModelCollection** object (or the same input as one) with an additional ensemble model.  The ensemble model will train on the predicted probabilities (or just predictions) of the **ModelCollection** and make a separate prediction.  This ensemble method allows for different weighting schemes of the outputs of each model as well as stacking the original data with the predictions when training the ensemble model.  Additionally, this ensemble class allows you to pretrain the **ModelCollection** object prior to training the ensemble model or training the **ModelCollection** object *with* the ensemble model.  This allows for more efficient hyperparameter tuning (albeit much longer training and tuning times).

Hyperparameters can be trained like any other scikit-learn-esque model.  When setting parameters for the ensemble model, parameters that begin with *name__* will set the hyperparameter of the *name* model in the **ModelCollection** object (similar to working with pipeline hyperparameter in scikit-learn).  Parameters that don't start with a *name__* denote hyperparameters for the ensemble model itself.

## Current Version is v0.01

This package is currently in the beginning stages but future work is planned.

## Usage

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
	
	models = ModelCollection(\[('log', LogisticRegression(random_state=0)),('nb', MultinomialNB())\])
	test.fit(x_train, y_train)
	
	ensemble = CatEnsemble(test, KNeighborsClassifier())
	ensemble.fit(x_train, y_train)
	test_preds = ensemble.predict(x_test)
	print(f"Accuracy on test set is {accuracy_score(y_test, test_preds)}"
	
	ens.set_params(\*\*{'log__C': 15, 'nb__alpha': 1, 'n_neighbors': 10})
	ens.fit(x_train, y_train)
	test_preds = ensemble.predict(x_test)
	print(f"Accuracy on test set is {accuracy_score(y_test, test_preds)}"
	
## Future Plans

The next step is to create an ensemble regressor model.  

## License

Lol
