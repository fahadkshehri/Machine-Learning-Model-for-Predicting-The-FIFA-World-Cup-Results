import os

import pandas as pd

from models.random_forest import RandomForestModel
from preprocess import read_csv


# Author: Fahad Alshehri
# CSS 490 Machine Learning
# Date 8/17/2018
if __name__ == '__main__':
    ckpt_dir = os.path.join(os.getcwd(), 'data', 'ckpts')
    input_dir = os.path.join(os.getcwd(), 'data', 'input')
    preprocess_dir = os.path.join(os.getcwd(), 'data', 'preprocess')
    simulate_dir = os.path.join(os.getcwd(), 'data', 'simulate')

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)
    if not os.path.exists(simulate_dir):
        os.makedirs(simulate_dir)

    train_df = read_csv(os.path.join(preprocess_dir, 'train_dataset.csv'))
    train_y = train_df.outcome
    train_x = train_df.drop(columns=['outcome', 'date', 'team1', 'team2'])

    test_df = read_csv(os.path.join(preprocess_dir, 'matches_to_predict.csv'))
    test_y = test_df.outcome
    test_x = test_df.drop(columns=['outcome', 'date', 'team1', 'team2'])

    model = RandomForestModel(n_estimators=500, min_samples_split=5)
    model.train(train_x, train_y)
    model.save_model(os.path.join(ckpt_dir, 'random_forest_model.ckpt'))

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
    pred_stats_path = os.path.join(simulate_dir, 'wc2018staticPredictions.csv')
    pred_stats.to_csv(pred_stats_path, index=False)
    print('Prediction results saved to `%s`' % pred_stats_path)
