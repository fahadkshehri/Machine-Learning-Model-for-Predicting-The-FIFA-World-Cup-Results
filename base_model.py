import abc

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict

# Author: Fahad Alshehri
# CSS 490 Machine Learning
# Date 8/17/2018

class BaseModel:

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def train(self, features, targets):
        print('%s training...' % self.__class__.__name__)

    @abc.abstractmethod
    def predict(self, features):
        print('%s predict...' % self.__class__.__name__)

    @abc.abstractmethod
    def predict_prob(self, features):
        print('%s predict_prob...' % self.__class__.__name__)

    @abc.abstractmethod
    def predict_log_prob(self, features):
        print('%s predict_prob...' % self.__class__.__name__)

    @abc.abstractmethod
    def accuracy_score(self, features, targets):
        print('%s accuracy_score...' % self.__class__.__name__)

    def cross_val_score(self, features, targets):
        print('%s cross_val_score...' % self.__class__.__name__)
        scores = cross_val_score(self.model, X=features, y=targets)
        print('cross_val_score results:\n %s' % scores)
        return scores

    def cross_val_predict(self, features, targets):
        print('%s cross_val_predict...' % self.__class__.__name__)
        scores = cross_val_predict(self.model, X=features, y=targets)
        print('cross_val_predict results:\n%s' % scores)
        return scores

    def metrics_mse(self, features, targets):
        print('%s metrics_mse...' % self.__class__.__name__)
        targets_pred = self.predict(features)
        result = mean_squared_error(y_true=targets, y_pred=targets_pred)
        return result

    def metrics_mae(self, features, targets):
        print('%s metrics_mae...' % self.__class__.__name__)
        targets_pred = self.predict(features)
        result = mean_absolute_error(y_true=targets, y_pred=targets_pred)
        return result

    def save_model(self, path):
        print('%s save_model...' % self.__class__.__name__)
        joblib.dump(self.model, path)

    def load_model(self, path):
        print('%s load_model...' % self.__class__.__name__)
        self.model = joblib.load(path)