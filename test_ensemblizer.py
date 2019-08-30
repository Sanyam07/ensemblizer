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

    def test_predict_proba(self):
        from ensemblizer.ensemblizer import ModelCollection
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        test = ModelCollection([('log', LogisticRegression(random_state=0)), ('nb', MultinomialNB())])
        test.fit(self.x_train, self.y_train)
        train_probs= test.predict_proba(self.x_train)
        self.assertEqual(self.sum_probs(train_probs[:,0]), 357)  # should be 357
        self.assertEqual(self.sum_probs(train_probs[:,1]), 359)  # should be 359

        test_probs= test.predict_proba(self.x_test)
        self.assertEqual(self.sum_probs(test_probs[:,0]), 89)  # should be 89
        self.assertEqual(self.sum_probs(test_probs[:,1]), 89)  # should be 89

if __name__=='__main__':
    unittest.main()
