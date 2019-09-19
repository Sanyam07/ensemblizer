from sklearn.base import BaseEstimator, ClassifierMixin


class ModelCollection:

    def __init__(self, models, **model_params):
        '''

        Class to allow collections of models to be easily fit together and predict outcome of each model

        :param model_list: List of tuples of form (name, model).  Similar to the setup for a scitkit-learn pipeline
        '''
        from tqdm.auto import tqdm
        self.tqdm = tqdm
        import numpy as np
        self.np = np
        self.models = models
        self.__set_params(models)
        self.set_params(**model_params)

    def __set_params(self, models):
        if models is None:
            models = self.models
        self.models = models
        self.model_list = []
        self.name_list = []
        for model in models:
            self.name_list.append(model[0])
            self.model_list.append(model[1])
        self.num_models = len(models)

    def __len__(self):
        return self.num_models

    def model_index(self, model_name):
        '''
        Returns index of model given model name
        :param model_name: name of model
        :return: index of model in self.model_list
        '''
        try:
            return self.name_list.index(model_name)
        except:
            raise ValueError("Model name not found in model list.")

    def __get_X(self, X, key):
        '''
        Workaround to allow different models to be trained on different sets concurrently

        :param X: Either a single dataset or list of datasets
        :param key: Index of dataset in list
        :return: returns X if X is not a list, returns the key-th dataset if X is a list of datasets
        '''
        if type(X) == list:
            return X[key]
        else:
            return X

    def __add__(self, new_collection):
        '''

        :param new_collection: New collection to concatenate with current one
        :return:
        '''
        if type(new_collection) == list:
            try:
                new_collection = ModelCollection(new_collection)
            except:
                raise ValueError("Incorrectly formatted list.")
        self.model_list = self.model_list + new_collection.model_list
        self.name_list = self.name_list + new_collection.name_list
        self.num_models = self.num_models + new_collection.num_models

    def fit(self, X, y, names=None, verbose=False):
        '''
        Fits each model

        :param X: Feature set or list of feature sets
        :param y: Labels
        :param names: names of models to train.  If None, trains all models
        :param verbose: True to return tqdm progress bar
        :return: self
        '''
        temp_list = []
        if names is None:
            names = self.name_list
        for key, name in self.tqdm(enumerate(names), disable=(not verbose)):
            index = self.name_list.index(name)
            temp_model = self.model_list[key].fit(self.__get_X(X, key), y)
            temp_list.append(temp_model)
        self.model_list = temp_list
        return self

    def predict(self, X, verbose=False):
        '''
        Create a prediction array

        :param X: Features
        :param verbose: True to return tqdm progress bar
        :return: returns an array of predictions, where each column represents the prediction of it's corresponding model
        '''
        for model_col in self.tqdm(range(self.num_models), disable=(not verbose)):
            if model_col == 0:
                predict_array = self.model_list[model_col].predict(self.__get_X(X, model_col))
            else:
                temp_array = self.model_list[model_col].predict(self.__get_X(X, model_col))
                predict_array = self.np.vstack((predict_array, temp_array))
        return self.np.transpose(predict_array)

    def predict_proba(self, X, shorten_array=True, verbose=False):
        '''
        Create a probability prediction array

        :param X: Features
        :param verbose: True to return tqdm progress bar
        :return: returns an array of prediction probabilities, where each column represents the prediction of it's corresponding model
        '''
        if shorten_array:
            slice = 1
        else:
            slice = 0
        for model_col in self.tqdm(range(self.num_models), disable=(not verbose)):
            if model_col == 0:
                try:
                    prob_array = self.model_list[model_col].predict_proba(self.__get_X(X, model_col))[:, slice:]
                except:
                    prob_array = self.model_list[model_col].predict(self.__get_X(X, model_col))
            else:
                try:
                    temp_array = self.model_list[model_col].predict_proba(self.__get_X(X, model_col))[:, slice:]
                except:
                    temp_array = self.model_list[model_col].predict(self.__get_X(X, model_col))
                prob_array = self.np.hstack((prob_array, temp_array))
        return prob_array

    def __sort_params(self, param_dict):
        '''
        Sorts overall parameters into individual model parameters for each model

        :param param_dict:  Combined parameters
        :return: a list of parameters, where the index of a paramater dictionary corresponds to the index of that model
        in self.model_list
        '''
        param_list = [dict() for _ in range(self.num_models)]
        for key, value in param_dict.items():
            split = key.split('__')
            if [key] == split:
                raise ValueError(f"{key} is not a valid hyperparameter.")
            elif [key] != split:
                model_name = split[0]
                param_name = "__".join(split[1:])
                try:
                    name_index = self.name_list.index(model_name)
                    param_list[name_index][param_name] = value
                except:
                    raise ValueError(f"{model_name} not a valid model name.")
        return param_list

    def set_params(self, models=None, **params):
        '''
        Set parameters for collection

        :param params: Parameter dictionary
        :return: Self
        '''
        self.__set_params(models)
        if params:
            params = self.__sort_params(params)
        for index in range(len(params)):
            if bool(params[index]):
                self.model_list[index].set_params(**params[index])
        return self

    def get_params(self, deep=True):
        '''

        :return: parameters for collection
        '''
        param_dict = {'models': self.models}
        for index in range(len(self.model_list)):
            temp_params = self.model_list[index].get_params(deep=deep)
            temp_dict = dict()
            for key, value in temp_params.items():
                param_name = self.name_list[index] + "__" + key
                temp_dict[param_name] = value
            param_dict.update(temp_dict)
        return param_dict


class CatEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, models, ensemble_model, weights=None, name="ensemble", stack_data=False, use_probs=True,
                 deep_train=False, sparse_stack=False, metric=None, stack_transformer=None, **model_params):
        import numpy as np
        self.np = np
        from scipy import stats
        self.stats = stats
        if type(models) == list:
            self.models = ModelCollection(models)
        else:
            self.models = models
        if weights is None:
            self.weights = [1.0] * self.models.num_models
        else:
            self.weights = weights
        self.sparse_stack = sparse_stack
        from scipy import sparse
        self.sparse_hstack = sparse.hstack
        self.name = name
        if metric:
            self.metric = metric
        else:
            from sklearn.metrics import accuracy_score
            self.metric = accuracy_score
        self.ensemble_model = ensemble_model
        self.stack_data = stack_data
        if self.stack_data == True and self.stack_transformer is None:
            self.stack_transformer = lambda x: x
        self.use_probs = use_probs
        self._estimator_type = "classifier"
        self.deep_train = deep_train
        self.stack_transformer = stack_transformer
        if self.stack_transformer is not None:
            self.stack_data = True
        if ensemble_model == "mean" or ensemble_model == "mode":
            if self.stack_data:
                raise ValueError("Cannot stack data using 'mean' or 'mode'")
        self.set_params(**model_params)

    def change_ensemble_params(self, models=None, ensemble_model=None, weights=None, stack_data=None, use_probs=None,
                               deep_train=None, sparse_stack=None, metric=None, stack_transformer=None):
        if models is not None:
            self.models = models
        if ensemble_model is not None:
            self.ensemble_model = ensemble_model
        if weights is not None:
            self.weights = weights
        if stack_data is not None:
            self.stack_data = stack_data
            if stack_data == True and stack_transformer is None:
                self.stack_transformer = lambda x: x
        if use_probs is not None:
            self.use_probs = use_probs
        if deep_train is not None:
            self.deep_train = deep_train
        if sparse_stack is not None:
            self.sparse_stack = sparse_stack
        if metric is not None:
            self.metric = metric
        if stack_transformer is not None:
            self.stack_transformer = stack_transformer
            self.stacked_data = True

    def __mean(self, X, return_probs=False):
        #
        # currently only set up to work with binary classification problems
        #
        total_weights = self.np.sum(self.weights)
        X = self.__base_array(X)
        pred_probs = self.np.sum(X, axis=1) / total_weights
        if return_probs:
            return pred_probs
        else:
            return self.np.round(pred_probs).astype(int)

    def __mode(self, X, return_probs=False):
        #
        # currently only set up to work with binary classification problems
        #
        base = self.models.predict(X)
        if return_probs:
            return self.np.mean(base, axis=1)
        else:
            return self.stats.mode(base, axis=1)[0].flatten()

    def fit(self, X, y, return_preds=False):
        if self.deep_train:
            self.models.fit(X, y)
        X_base = self.__base_array(X)
        if self.ensemble_model == "mean":
            if return_preds:
                return self.__mean(X)
            else:
                return None
        elif self.ensemble_model == "mode":
            if return_preds:
                return self.__mode(X)
            else:
                return None
        self.ensemble_model.fit(X_base, y)
        self.estimator_ = self
        if return_preds:
            return self.ensemble_model.predict(X_base)
        else:
            return self

    def __base_array(self, X):
        if self.use_probs:
            base_array = self.models.predict_proba(X)
        else:
            base_array = self.models.predict(X)
        base_array = base_array * self.weights
        if self.stack_data:
            stacked = self.stack_transformer.transform(X)
            if self.sparse_stack:
                base_array = self.sparse_hstack((base_array, stacked))
            else:
                base_array = self.np.hstack((base_array, stacked))
        return base_array

    def set_params(self, models=None, ensemble_model=None, weights=None, stack_data=None,
                   use_probs=None, deep_train=None, sparse_stack=None, metric=None, stack_transformer=None, **params):
        self.change_ensemble_params(models, ensemble_model, weights, stack_data, use_probs, deep_train,
                                    sparse_stack, metric, stack_transformer)
        collection_dict = dict()
        ensemble_dict = dict()
        for key, value in params.items():
            split = key.split("__")
            if split[0] == self.name:
                ensemble_dict["__".join(split[1:])] = value
            else:
                if split[0] == "":
                    self.weights[self.models.model_index(split[1])] = value
                else:
                    param_name = "__".join(split)
                    collection_dict[param_name] = value
        self.models.set_params(**collection_dict)
        if (self.ensemble_model != "mean") and (self.ensemble_model != "mode"):
            self.ensemble_model.set_params(**ensemble_dict)
        return self

    def get_params(self, deep=True):
        param_dict = {'models': self.models, 'ensemble_model': self.ensemble_model, 'weights': self.weights,
                      'stack_data': self.stack_data, 'use_probs': self.use_probs, 'deep_train': self.deep_train,
                      'sparse_stack': self.sparse_stack, 'metric': self.metric,
                      'stack_transformer': self.stack_transformer}
        temp_params = self.ensemble_model.get_params(deep=deep)
        for key, value in temp_params.items():
            param_name = self.name + "__" + key
            param_dict[param_name] = value
        for index in range(len(self.weights)):
            name = "__" + self.models.name_list[index]
            param_dict[name] = self.weights[index]
        param_dict.update(self.models.get_params(deep=deep))
        return param_dict

    def predict(self, X):
        if self.ensemble_model == "mean":
            return self.__mean(X)
        elif self.ensemble_model == "mode":
            return self.__mode(X)
        X_base = self.__base_array(X)
        return self.ensemble_model.predict(X_base)

    def score(self, X, y):
        return self.metric(self.predict(X), y)

    def predict_proba(self, X):
        if self.ensemble_model == "mean":
            return self.__mean(X, return_probs=True)
        elif self.ensemble_model == "mode":
            return self.__mode(X, return_probs=True)
        X_base = self.__base_array(X)
        return self.predict_proba(X_base)