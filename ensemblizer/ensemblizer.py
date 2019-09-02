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

    def model_index(self, model_name):
        try:
            return self.name_list.index(model_name)
        except:
            raise ValueError("Model name not found in model list.")

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
                try:
                    prob_array = self.model_list[model_col].predict_proba(self.__get_X(X, model_col))[:,slice:]
                except:
                    prob_array = self.model_list[model_col].predict(self.__get_X(X, model_col))
            else:
                try:
                    temp_array = self.model_list[model_col].predict_proba(self.__get_X(X, model_col))[:,slice:]
                except:
                    temp_array = self.model_list[model_col].predict(self.__get_X(X, model_col))
                prob_array = self.np.hstack((prob_array, temp_array))
        return prob_array

    def __sort_params(self, param_dict, skip_unknown=False):
        param_list = [dict() for _ in range(self.num_models)]
        for key, value in param_dict.items():
            split = key.split('__')
            if [key]==split and not skip_unknown:
                raise ValueError(f"{key} is not a valid hyperparameter.")
            elif [key]!=split:
                model_name = split[0]
                param_name = "__".join(split[1:])
                try:
                    name_index = self.name_list.index(model_name)
                    param_list[name_index][param_name] = value
                except:
                    if not skip_unknown:
                        raise ValueError(f"{model_name} not a valid model name.")
        return param_list

    def set_params(self, skip_unknown=False, **params):
        if type(params)==dict:
            params = self.__sort_params(params, skip_unknown=skip_unknown)
        for index in range(len(params)):
            if bool(params[index]):
                self.model_list[index].set_params(**params[index])
        return self.model_list

class CatEnsemble:

    def __init__(self, models, ensemble_model, weights=None, name="ensemble", stack_data=False, use_probs=True,
                 train_collection=False, sparse_data=False, stack_transformer=None):
        import numpy as np
        self.np = np
        from scipy import stats
        self.stats = stats
        if type(models)==list:
            self.models = ModelCollection(models)
        else:
            self.models = models
        if weights is None:
            self.weights = [1.0]*self.models.num_models
        else:
            self.weights = weights
        self.sparse = sparse_data
        from scipy import sparse
        self.sparse_hstack = sparse.hstack
        self.name = name
        self.ensemble_model = ensemble_model
        self.stack_data = stack_data
        if self.stack_data==True and self.stack_transformer is None:
            self.stack_transformer = lambda x: x
        self.use_probs = use_probs
        self.ensemble_params = dict()
        self.train_collection = train_collection
        self.stack_transformer = stack_transformer
        if self.stack_transformer is not None:
            self.stack_data = True
        if ensemble_model == "mean" or ensemble_model == "mode":
            if self.stack_data:
                raise ValueError("Cannot stack data using 'mean' or 'mode'")

    def set_ensemble_params(self, models=None, ensemble_model=None, weights=None, stack_data=None, use_probs=None,
                            train_collection=None, sparse_data=None, reset_stored=False, stack_transformer=None):
        if models is not None:
            self.models = models
        if ensemble_model is not None:
            self.ensemble_model = ensemble_model
        if weights is not None:
            self.weights = weights
        if stack_data is not None:
            self.stack_data = stack_data
            if stack_data==True and stack_transformer is None:
                self.stack_transformer = lambda x: x
        if use_probs is not None:
            self.use_probs = use_probs
        if train_collection is not None:
            self.train_collection = train_collection
        if sparse_data is not None:
            self.sparse = sparse_data
        if stack_transformer is not None:
            self.stack_transformer = stack_transformer
            self.stacked_data = True


    def __mean(self, X, return_probs=False):
        #
        # currently only set up to work with binary classification problems
        #
        total_weights = self.np.sum(self.weights)
        X = self.__base_array(X)
        pred_probs = self.np.sum(X, axis=1)/total_weights
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
        if return_preds:
            return self.ensemble_model.predict(X_base)
        else:
            return self

    def __base_array(self, X):
        if self.use_probs:
            base_array = self.models.predict_proba(X)
        else:
            base_array = self.models.predict(X)
        base_array = base_array*self.weights
        if self.stack_data:
            stacked = self.stack_transformer.transform(X)
            if self.sparse:
                base_array = self.sparse_hstack((base_array, stacked))
            else:
                base_array = self.np.hstack((base_array, stacked))
        return base_array

    def set_params(self, skip_unknown=False, **params):
        collection_dict = dict()
        ensemble_dict = dict()
        for key, value in params.items():
            split = key.split("__")
            if split[0]==self.name:
                ensemble_dict["__".join(split[1:])] = value
            else:
                if split[0]=="":
                    self.weights[self.models.model_index(key.split('__')[1])] = value
                else:
                    param_name = "__".join(split)
                    collection_dict[param_name] = value
        self.models.set_params(skip_unknown=skip_unknown, **collection_dict)
        if (self.ensemble_model != "mean") and (self.ensemble_model != "mode"):
            self.ensemble_model.set_params(**ensemble_dict)

    def predict(self, X):
        if self.ensemble_model == "mean":
            return self.__mean(X)
        elif self.ensemble_model == "mode":
            return self.__mode(X)
        X_base = self.__base_array(X)
        return self.ensemble_model.predict(X_base)

    def predict_proba(self, X):
        if self.ensemble_model == "mean":
            return self.__mean(X, return_probs=True)
        elif self.ensemble_model == "mode":
            return self.__mode(X, return_probs=True)
        X_base = self.__base_array(X)
        return self.predict_proba(X_base)