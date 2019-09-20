import unittest

class modelTest(unittest.TestCase):


    def setUp(self):
        import numpy as np
        self.np = np
        np.random.seed(0)
        self.x_train = np.random.randint(0, 10, (80, 3))
        self.x_test = np.random.randint(0, 10, (20, 3))
        self.y_train = np.random.randint(0, 2, 80)
        self.y_test = np.random.randint(0, 2, 20)

    def total_misclass(self, actual, preds):
        return self.np.abs(self.np.sum(actual - preds))

    def sum_probs(self, probs):
        return int(self.np.sum(probs) * 10)

    def test_predict(self):
        from ensemblizer.ensemblizer import ModelCollection
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        test = ModelCollection([('log', LogisticRegression(random_state=0)), ('nb', MultinomialNB())])
        test.fit(self.x_train, self.y_train)
        train_preds = test.predict(self.x_train)
        self.assertEqual(self.total_misclass(train_preds[:, 0], self.y_train), 10)  # should be 10
        self.assertEqual(self.total_misclass(train_preds[:, 1], self.y_train), 8)  # should be 8

        test_preds = test.predict(self.x_test)
        self.assertEqual(self.total_misclass(test_preds[:, 0], self.y_test), 2)  # should be 2
        self.assertEqual(self.total_misclass(test_preds[:, 1], self.y_test), 2)  # should be 2

    def test_predict_proba_and_set_params(self):
        from ensemblizer.ensemblizer import ModelCollection
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        test = ModelCollection([('log', LogisticRegression(random_state=0)), ('nb', MultinomialNB())])
        test.set_params(**{'log__C': 10, 'nb__alpha': 2})
        test.fit(self.x_train, self.y_train)
        train_probs= test.predict_proba(self.x_train)
        self.assertEqual(self.sum_probs(train_probs[:,0]), 359)  # should be 359
        self.assertEqual(self.sum_probs(train_probs[:,1]), 359)  # should be 359

        test_probs= test.predict_proba(self.x_test)
        self.assertEqual(self.sum_probs(test_probs[:,0]), 89)  # should be 89
        self.assertEqual(self.sum_probs(test_probs[:,1]), 89)  # should be 89

class CatEnsembleTest(unittest.TestCase):


    def setUp(self):
        import numpy as np
        from sklearn.neighbors import KNeighborsClassifier
        from ensemblizer.ensemblizer import ModelCollection, CatEnsemble
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        self.np = np
        np.random.seed(0)
        self.x_train = np.random.randint(0, 10, (80, 3))
        self.x_test = np.random.randint(0, 10, (20, 3))
        self.y_train = np.random.randint(0, 2, 80)
        self.y_test = np.random.randint(0, 2, 20)
        self.test = ModelCollection([('log', LogisticRegression(random_state=0)), ('nb', MultinomialNB())])
        self.test.fit(self.x_train, self.y_train)
        self.ens = CatEnsemble(self.test, KNeighborsClassifier())

    def total_misclass(self, actual, preds):
        return self.np.abs(self.np.sum(actual - preds))

    def sum_probs(self, probs):
        return int(self.np.sum(probs) * 10)

    def test_simple(self):
        self.ens.fit(self.x_train, self.y_train)
        ens_train_preds = self.ens.predict(self.x_train)
        ens_test_preds = self.ens.predict(self.x_test)
        self.assertEqual(self.total_misclass(ens_train_preds, self.y_train), 7)  # should be 7
        self.assertEqual(self.total_misclass(ens_test_preds, self.y_test), 1)  # should be 1

    def test_change_params_simple(self):
        self.ens.set_params(**{'log__C': 15, 'nb__alpha': 1, 'ensemble__n_neighbors': 10})
        self.ens.fit(self.x_train, self.y_train)
        ens_train_preds = self.ens.predict(self.x_train)
        ens_test_preds = self.ens.predict(self.x_test)
        self.assertEqual(self.total_misclass(ens_train_preds, self.y_train), 13)  # should be 13
        self.assertEqual(self.total_misclass(ens_test_preds, self.y_test), 0)  # should be 0

    def test_change_collection_change_params(self):
        self.ens.set_params(deep_train=True)
        self.ens.fit(self.x_train, self.y_train)
        ens_train_preds = self.ens.predict(self.x_train)
        ens_test_preds = self.ens.predict(self.x_test)
        self.assertEqual(self.total_misclass(ens_train_preds, self.y_train), 7)  # should be 7
        self.assertEqual(self.total_misclass(ens_test_preds, self.y_test), 1)  # should be 1

        self.ens.set_params(**{'log__C': 1, 'nb__alpha': 1, 'ensemble__n_neighbors': 1})
        self.ens.fit(self.x_train, self.y_train)
        ens_train_preds = self.ens.predict(self.x_train)
        ens_test_preds = self.ens.predict(self.x_test)
        self.assertEqual(self.total_misclass(ens_train_preds, self.y_train), 2)  # should be 2
        self.assertEqual(self.total_misclass(ens_test_preds, self.y_test), 4)  # should be 4

    def test_shallow_grid(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.exceptions import NotFittedError
        self.ens.set_params(deep_train=False)
        self.GridSearchCV = GridSearchCV
        params = {'ensemble__n_neighbors': [2, 3, 5]}
        search = self.GridSearchCV(self.ens, params)
        try:
            search.fit(self.x_train, self.y_train)
        except NotFittedError:
            self.fail("Not scikit-learn shallow search compatible")

    def test_deep_grid(self):
        from sklearn.exceptions import NotFittedError
        self.ens.set_params(deep_train=True)
        params = {'ensemble__n_neighbors': [2, 3, 5],
                  'log__C': [1, 10, 100]}
        search = self.GridSearchCV(self.ens, params)
        try:
            search.fit(self.x_train, self.y_train)
        except NotFittedError:
            self.fail("Not scikit-learn deep search compatible")

if __name__=='__main__':
    unittest.main()
