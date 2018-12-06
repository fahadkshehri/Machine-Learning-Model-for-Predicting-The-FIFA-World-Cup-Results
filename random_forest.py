from sklearn.ensemble import RandomForestRegressor

from models.base_model import BaseModel
from preprocess import read_csv
import numpy as np
import pandas as pd

# Author: Fahad Alshehri
# CSS 490 Machine Learning
# Date 8/17/2018

class RandomForestModel(BaseModel):
    """RandomForest classifier."""

    def __init__(self,
                 n_estimators=100,
                 max_features='sqrt',
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 bootstrap=True,
                 oob_score=True,
                 verbose=1):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            verbose=verbose)

        super().__init__(model)

    def predict(self, features):
        super().predict(features)
        labels = self.model.predict(features)
        return labels

    def predict_prob(self, features):
        super().predict_prob(features)
        probs = self.model.predict_proba(features)
        return probs

    def predict_log_prob(self, features):
        super().predict_log_prob(features)
        probs = self.model.predict_log_proba(features)
        return probs

    def predict_individuals(self, features):
        print('%s predict_individuals.' % self.__class__.__name__)
        y_hats = []
        for e in self.model.estimators_:
            y_hats.append(e.predict(features))
        return np.array(y_hats)

    def train(self, features, targets):
        super().train(features, targets)
        self.model.fit(X=features, y=targets)

    def accuracy_score(self, features, targets):
        super().accuracy_score(features, targets)
        score = self.model.score(features, targets, self.model.class_weight)
        return score

    def abs_errors(self, features, targets):
        targets_pred = self.predict(features)
        result = abs(targets_pred - targets)
        return result


if __name__ == '__main__':
    train_df = read_csv('../data/train_dataset.csv')
    train_y = train_df.outcome
    train_x = train_df.drop(columns=['outcome', 'date', 'team1', 'team2'])

    test_df = read_csv('../data/matches_to_predict.csv')
    test_y = test_df.outcome
    test_x = test_df.drop(columns=['outcome', 'date', 'team1', 'team2'])

    model = RandomForestModel(n_estimators=500, min_samples_split=5)
    model.train(train_x, train_y)
    model.save_model('../ckpts/random_forest_model.ckpt')

    # model = RandomForestModel(n_estimators=100)
    # model.load_model('../ckpts/random_forest_model.ckpt')

    # score = model.accuracy_score(train_x, train_y)
    mse = model.metrics_mse(test_x, test_y)
    mae = model.metrics_mae(test_x, test_y)
    score = model.accuracy_score(test_x, test_y)

    print('MSE: %s' % mse)
    print('MAE: %s' % mae)
    print('Score: %s' % score)

    # Predict test dataset.
    individual_res = model.predict_individuals(test_x)

    outcome_pred = individual_res.mean(axis=0)
    outcome_std = individual_res.std(axis=0)

    pred_stats = pd.DataFrame({'outcome': outcome_pred, 'sd': outcome_std})
    pred_stats = pd.concat([test_df[['team1', 'team2']], pred_stats], axis=1)
    pred_stats.to_csv('../data/wc2018staticPredictions.csv', index=False)

    # abs_errors = model.abs_errors(test_x, test_y)
    # plot_scatters(abs_errors, 'index', 'abs_errors', 'Absolute errors')