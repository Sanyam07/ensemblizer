class ModelCollection:

    def __init__(self, model_list):
        from tqdm.auto import tqdm
        self.tqdm = tqdm
        import numpy as np
        self.np = np
        self.model_list = []
        self.name_list = []
        for model in model_list:
            self.name_list.append(model[0])
            self.model_list.append(model[1])
        self.num_models = len(model_list)

    def __get_X(self, X, key):
        if type(X)==list:
            return X[key]
        else:
            return X

    def concat(self, new_collection):
        if type(new_collection)==list:
            try:
                new_collection = ModelCollection(new_collection)
            except:
                raise ValueError("Incorrectly formatted list.")
        self.model_list = self.model_list + new_collection.model_list
        self.name_list = self.name_list + new_collection.name_list
        self.num_models = self.num_models + new_collection.num_models

    def fit(self, X, y, names=None, verbose=False):
        if names is None:
            names = self.name_list
        for key, name in self.tqdm(enumerate(names), disable=(not verbose)):
            index = self.name_list.index(name)
            self.model_list[index] = self.model_list[index].fit(self.__get_X(X, key), y)
        return self.model_list

    def predict(self, X, verbose=False):
        for model_col in self.tqdm(range(self.num_models), disable=(not verbose)):
            if model_col == 0:
                predict_array = self.model_list[model_col].predict(self.__get_X(X, model_col))
            else:
                temp_array = self.model_list[model_col].predict(self.__get_X(X, model_col))
                predict_array = self.np.vstack((predict_array, temp_array))
        return self.np.transpose(predict_array)

    def predict_proba(self, X, shorten_array=True, verbose=False):
        if shorten_array:
            slice = 1
        else:
            slice = 0
        for model_col in self.tqdm(range(self.num_models), disable=(not verbose)):
            if model_col == 0:
                prob_array = self.model_list[model_col].predict_proba(self.__get_X(X, model_col))[:,slice:]
            else:
                temp_array = self.model_list[model_col].predict_proba(self.__get_X(X, model_col))[:,slice:]
                prob_array = self.np.hstack((prob_array, temp_array))
        return prob_array

    def __sort_params(self, param_dict, skip_unknown=False):
        param_list = [dict()]*self.num_models
        for key, value in param_dict.items():
            model_name = key.split('__')[0]
            param_name = key.split('__')[1]
            try:
                name_index = self.name_list.index(model_name)
                param_list[name_index][param_name] = value
            except:
                if not skip_unknown:
                    raise ValueError(f"{model_name} not a valid model name.")
        return param_list

    def set_params(self, params, skip_unknown=False):
        if type(params)==dict:
            params = self.__sort_params(params, skip_unknown=skip_unknown)
        for index, param_dict in enumerate(params):
            if bool(param_dict):
                self.model_list[index].set_params(param_dict)
        return self.model_list

class CatEnsemble:

    def __init__(self, ):
'''
class PretrainedCatEnsemble:
    def __init__(self, model_list, ensemble_model=None, weights=None, include_data=False, use_probs=True):
        if ensemble_model is None:
            from sklearn.linear_model import LogisticRegression
            self.ensemble_model = LogisticRegression()
        else:
            self.ensemble_model = ensemble_model
        self.num_models = len(model_list)
        self.model_list = model_list
        self.num_labels = 1
        self.use_probs = use_probs
        self.optimal_params = dict()
        self.stored_probs = None
        self.stored_y = None
        self.stored_test_probs = None
        self.stored_test_y = None
        self.stored_test = False
        self.include_data = include_data
        self.stored_X_train = None
        self.stored_X_test = None
        self.base_weights = np.array([1.0] * self.num_models)
        self.max_weights = np.array([1.0] * self.num_models)
        if weights is None:
            self.weights = np.array([1.0] * self.num_models)
        else:
            self.weights = np.array(weights)

    def set_data(self, X_train=None, y_train=None, X_test=None, y_test=None, weighted=False, include_data=None,
                 verbose=False):
        if include_data is not None:
            self.include_data = include_data
        if X_train is not None:
            self.stored_probs = self.__prob_array(X_train, weighted=weighted, verbose=verbose)
            if self.include_data:
                self.stored_X_train = X_train
        if X_test is not None:
            self.stored_test = True
            self.stored_test_probs = self.__prob_array(X_test, weighted=weighted, verbose=verbose)
            if self.include_data:
                self.stored_X_test = X_test
        if y_test is not None:
            self.stored_test_y = y_test
        if y_train is not None:
            self.stored_y = y_train

    def set_weights(self, weights="base"):
        if weights == "base":
            self.weights = self.base_weights
        elif weights == "max":
            self.weights = self.max_weights
        else:
            self.weights = weights

    def change_ensemble_method(self, ensemble_model, weighted=False):
        self.ensemble_model = ensemble_model
        if weighted:
            X = self.stored_probs * self.weights
        else:
            X = self.stored_probs
        self.ensemble_model.fit(X, self.stored_y)
        return self.ensemble_model.predict(X)

    def tune_hyperparameters(self, hyperparameters, num_samples=5, method="bayes", n_iter=60,
                             n_jobs=-1, sample_size=0.2, test_size=0.3, verbose=0, random_state=None, train_final=True,
                             return_accuracy=False, return_search=False, weighted=False):
        if method == "bayes":
            grid = BayesSearchCV(self.ensemble_model, hyperparameters, cv=num_samples, n_iter=n_iter,
                                 verbose=verbose, n_jobs=n_jobs, random_state=random_state)
        elif method == "grid":
            grid = GridSearchCV(self.ensemble_model, hyperparameters, cv=num_samples,
                                verbose=verbose, n_jobs=n_jobs)
        elif method == "random":
            grid = RandomizedSearchCV(self.ensemble_model, hyperparameters, cv=num_samples, n_iter=n_iter,
                                      verbose=verbose, n_jobs=n_jobs, random_state=random_state)
        elif method == "random-resample":
            grid = RandomSearchResample(self.ensemble_model, params=hyperparameters, num_iter=n_iter,
                                        sample_size=sample_size,
                                        num_samples=num_samples, test_size=test_size, random_state=random_state)
        X = self.__prob_array(X="train", weighted=weighted)
        grid.fit(X, self.stored_y)
        if train_final:
            model = self.ensemble_model.set_params(**grid.best_params_)
            model.fit(X, self.stored_y)
        self.optimal_params = grid.best_params_
        self.ensemble_model.set_params(**self.optimal_params)
        if return_accuracy:
            if return_search:
                return grid.best_params_, grid.best_score_, grid
            else:
                return grid.best_params_, grid.best_score_
        else:
            if return_search:
                return grid.best_params_, grid
            else:
                return grid.best_params_

    def maximize_weights(self, weight_grid=None, sample_size=0.2, num_samples=20, set_weights=True,
                         method="random", n_iter=1500, return_acc=True, random_state=None, verbose=False):
        max_acc = 0.0
        max_acc_weights = []
        X = self.__prob_array(X="train")
        y = self.stored_y
        if weight_grid is None:
            simple_grid = np.arange(start=0.0, stop=1.01, step=0.05)
            weight_grid = [simple_grid for _ in range(self.num_models)]
        weight_grid = itertools.product(*weight_grid)
        if method == "random":
            random.seed(random_state)
            weight_grid = random.sample(list(weight_grid), n_iter)
        for weights in tqdm_notebook(weight_grid, disable=(not verbose)):
            acc = accuracy_score(y, self.mean(X=X, weights=list(weights), need_prob_array=False))
            if acc > max_acc:
                max_acc = acc
                max_acc_weights = weights
        self.max_weights = weights
        if set_weights:
            self.weights = max_acc_weights
        if return_acc:
            return self.max_weights, max_acc
        else:
            return self.max_weights

    def mean(self, X="train", weights="base", need_prob_array=True, probs=False, verbose=False):
        self.set_weights(weights)
        if need_prob_array:
            og_probs = self.__prob_array(X, weighted=True, verbose=verbose)
        else:
            og_probs = X * self.weights
        total_weight = np.sum(self.weights)
        if total_weight == 0:
            return np.array([0.0] * len(og_probs))
        weighted_probs = []
        for row in og_probs:
            val = np.sum(row) / total_weight
            if not probs:
                val = int(round(val))
            weighted_probs.append(val)
        return np.array(weighted_probs)

    def fit(self, X=None, y=None, weighted=False, verbose=False, return_predictions=False):
        if X is None:
            X_data = self.stored_probs
            y = self.stored_y
            if self.include_data:
                X = self.stored_X_train
        else:
            self.num_labels = len(y.unique()) - 1
            X_data = self.__prob_array(X, weighted=weighted, verbose=verbose)
            self.stored_probs = X_data
            self.stored_y = y
            if self.include_data:
                self.stored_X_train = X
        if self.include_data:
            X_data = np.concatenate((X_data, X), axis=1)
        self.ensemble_model.fit(X_data, y)
        if return_predictions:
            return self.ensemble_model.predict(X_data)

    def predict(self, X, verbose=False, weighted=False):
        X_probs = self.__prob_array(X, verbose=verbose, weighted=weighted)
        if X == "train" and self.include_data:
            X = self.stored_X_train
        elif X == "test" and self.include_data:
            X = self.stored_X_test
        if self.include_data:
            X_data = np.concatenate((X_probs, X), axis=1)
        return self.ensemble_model.predict(X_data)

    def predict_proba(self, X, weighted=False):
        X_probs = self.__prob_array(X, weighted=weighted)
        if X == "train" and self.include_data:
            X = self.stored_X_train
        elif X == "test" and self.include_data:
            X = self.stored_X_test
        if self.include_data:
            X_data = np.concatenate((X_probs, X), axis=1)
        return self.ensemble_model.predict_proba(X_data)

    def mode_predict(self, X="train"):
        X = self.__prob_array(X)
        X = X.round().astype(int)
        pred = stats.mode(X, axis=1)[0]
        return pred

    def __prob_array(self, X, weighted=False, verbose=False):
        if weighted:
            weights = self.weights
        else:
            weights = 1.0
        if X == "train":
            return self.stored_probs * weights
        elif X == "test":
            return self.stored_test_probs * weights
        num_rows = len(self.__get_features(X, 0))
        prob_matrix = np.zeros((num_rows, self.num_models * self.num_labels), dtype='float32')
        for index in tqdm_notebook(range(self.num_models), disable=(not verbose)):
            if self.num_labels > 1:
                prob_matrix[:, index:index + self.num_labels] = self.__predict(self.model_list[index],
                                                                               self.__get_features(X, index))
            else:
                prob_matrix[:, index] = self.__predict(self.model_list[index], self.__get_features(X, index))
        return prob_matrix * weights

    def __predict(self, model, X):
        if not self.use_probs:
            return model.predict(X)
        try:
            multi_channel = model.predict_proba(X)
            if self.num_labels == 1:
                return np.reshape(multi_channel[:, 1:], len(X))
            else:
                return multi_channel[:, 1:]
        except:
            return model.predict(X)

    def __get_features(self, X, index=None):
        if type(X) == list:
            return X[index]
        else:
            return X
'''