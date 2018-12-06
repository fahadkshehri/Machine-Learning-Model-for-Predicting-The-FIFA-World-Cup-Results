import os
import random
import warnings
from itertools import product

import pandas as pd

# Author: Fahad Alshehri
# CSS 490 Machine Learning
# Date 8/17/2018

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_csv(path, strip_values=False):
    if os.path.exists(path):
        file = open(path, mode='r', encoding='utf-8')
        if strip_values:
            df = pd.read_csv(file, engine='python', delimiter=' *, *')  # 消除空格
        else:
            df = pd.read_csv(file)
        return df
    else:
        raise FileNotFoundError('File %s not found.' % path)


def clean_up(df: pd.DataFrame):
    """Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
       'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
       'team2Score', 'team1PenScore', 'team2PenScore']
      """
    print('# Do `clean_up`.')
    # Convert timestamp to datetime
    df['date'] = df.date.astype('str')
    df['date'] = pd.to_datetime(df.date)
    # Eliminate duplicates
    df = df.drop_duplicates(subset=['date', 'team1', 'team2'])
    # Reset index
    df = df.reset_index(drop=True)
    # Add `match_id` column.
    df['match_id'] = pd.Series(data=[int(i) for i in range(1, df.shape[0] + 1)])

    # Mapping team code to new.
    mapping = {"FRG": "GER", "TCH": "CZE", "URS": "RUS", "SCG": "SRB", "ZAI": "COD"}

    def mapping_filename(row):
        if row['team1'] in mapping:
            row['team1'] = mapping[row['team1']]
        if row['team2'] in mapping:
            row['team2'] = mapping[row['team2']]

        return row

    df = df.apply(lambda row: mapping_filename(row), axis='columns')

    return df


def exchange_team_order(data: pd.DataFrame):
    """Exchange team order to avoid difference bias.

    The Mean value for the goal differential is greater than 0.6, which may
    present a problem later on when training the model.It may capture this bias
    team1 is better than team2, which is something we'd rather avoid, especially
    since the World Cup final tournament is played in a single country.
    So we should get rid of that by simply ran# Domizing the order in which teams are
    listed for any one match.

    Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
       'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
       'team2Score', 'team1PenScore', 'team2PenScore']
    """
    print('# Do `exchange_team_order`.')

    def exchange(row):
        p = random.random()
        if p >= 0.5:
            # Exchange
            row['team1'], row['team2'] = row['team2'], row['team1']
            row['team1Text'], row['team2Text'] = row['team2Text'], row['team1Text']
            row['team1Score'], row['team2Score'] = row['team2Score'], row['team1Score']
            row['team1PenScore'], row['team2PenScore'] = row['team2PenScore'], row['team1PenScore']
            row['resText'] = ''
        return row

    random.seed(4342)
    # Apply function.
    return data.apply(lambda row: exchange(row), axis='columns')


def add_additional_features(data: pd.DataFrame):
    """Add some more features to DataFrame

     Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
       'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
       'team2Score', 'team1PenScore', 'team2PenScore']
    """
    print('# Do `add_additional_features`.')

    def add_feature(row):
        # is the game played in a neutral venue
        row['team1Home'] = True if row.team1Text in str(row.venue) else False
        row['team2Home'] = True if row.team2Text in str(row.venue) else False
        row['neutralVenue'] = not (row.team1Home or row.team2Home)
        row['friendly'] = True if 'Friendly' in row.CupName else False
        row['qualifier'] = True if 'qual' in row.CupName else False
        row['finaltourn'] = True if 'final' in row.CupName or 'Confederations Cup' in row.CupName else False

        return row

    new_data = pd.concat([data, pd.DataFrame(columns=['team1Home', 'team2Home', 'neutralVenue',
                                                      'friendly', 'qualifier', 'finaltourn'])],
                         ignore_index=True)

    return new_data.apply(lambda row: add_feature(row), axis='columns')


def drop_friendly_match(data: pd.DataFrame):
    """Add some more features to DataFrame

    Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
    'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
    'team2Score', 'team1PenScore', 'team2PenScore','team1Home',
    'team2Home', 'neutralVenue', 'friendly', 'qualifier', 'finaltourn']
    """
    return data[data.friendly == False]


def make_team_perf_dataset(data: pd.DataFrame, filter_outlier=False):
    """
    Eliminate friendly matches from the dataset and make team performance dataset.

    The main objective for a team playing a friendly is not to win it, but to evaluate
    its own players and tactics.For this reason it's not uncommon for friendlies to allow
    an unlimited number of substitutions, and for a team to roll out its entire squad
    during a friendly game.

    Take each observation in `matches` - which has the form *"team1 vs team2"*
    and produce two separate observations of the form *"team1 played against team2"*
    and *"team2 played against team1"* respectively.

    Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
    'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
    'team2Score', 'team1PenScore', 'team2PenScore','team1Home',
    'team2Home', 'neutralVenue', 'friendly', 'qualifier', 'finaltourn']
    """

    print('# Do `make_team_perf_dataset`.')

    rows = []
    for index, row in data.iterrows():
        if row.friendly:
            # only use official matches (no friendlies)
            continue

        goal_diff = row.team1Score - row.team2Score

        # Restrict gd to [-7, 7]
        if filter_outlier:
            goal_diff = -7 if goal_diff < -7 else goal_diff
            goal_diff = 7 if goal_diff > 7 else goal_diff

        # team1 against team2
        rows.append({'match_id': row.match_id, 'date': row.date, 'name': row.team1, 'opponentName': row.team2,
                     'homeVenue': row.team1Home, 'neutralVenue': row.neutralVenue, 'gs': row.team1Score,
                     'ga': row.team2Score, 'gd': goal_diff, 'win': row.team1Score > row.team2Score,
                     'loss': row.team1Score < row.team2Score, 'draw': row.team1Score == row.team2Score,
                     'friendly': row.friendly, 'qualifier': row.qualifier, 'finaltourn': row.finaltourn})
        # team2 against team1
        rows.append({'match_id': row.match_id, 'date': row.date, 'name': row.team2, 'opponentName': row.team1,
                     'homeVenue': row.team2Home, 'neutralVenue': row.neutralVenue, 'gs': row.team2Score,
                     'ga': row.team1Score, 'gd': -goal_diff, 'win': row.team2Score > row.team1Score,
                     'loss': row.team2Score < row.team1Score, 'draw': row.team2Score == row.team1Score,
                     'friendly': row.friendly, 'qualifier': row.qualifier, 'finaltourn': row.finaltourn})

    return pd.DataFrame(rows)


def make_score_freq_dataset(data: pd.DataFrame):
    """Investigate what is the occurence frequency for match scores.

    Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
       'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
       'team2Score', 'team1PenScore', 'team2PenScore','team1Home',
       'team2Home', 'neutralVenue', 'friendly', 'qualifier', 'finaltourn']
    """
    print('# Do `make_score_freq_dataset`.')

    groups = data.groupby(['team1Score', 'team2Score'])

    new_rows = []
    for index, group in groups:
        team1_score, team2_score = int(group.team1Score.iloc[0]), int(group.team2Score.iloc[0])
        new_rows.append({'team1Score': team1_score,
                         'team2Score': team2_score,
                         'count': len(group),
                         'freq': len(group) / len(data),
                         'scoreText': str(team1_score) + '-' + str(team2_score)})

    new_data = pd.DataFrame(new_rows)
    new_data.sort_index(by='count', ascending=False, inplace=True)

    return new_data


def make_sum_per_match_dataset(data: pd.DataFrame):
    """Investigate what is the occurence frequency for match scores.

    Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
       'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
       'team2Score', 'team1PenScore', 'team2PenScore','team1Home',
       'team2Home', 'neutralVenue', 'friendly', 'qualifier', 'finaltourn']
    """
    print('# Do `make_sum_per_match_dataset`.')

    groups = data.groupby([data.team1Score + data.team2Score])

    new_rows = []
    for index, group in groups:
        team1_score, team2_score = int(group.team1Score.iloc[0]), int(group.team2Score.iloc[0])
        new_rows.append({'goalSum': team1_score + team2_score,
                         'count': len(group),
                         'freq': len(group) / len(data)})

    new_data = pd.DataFrame(new_rows)
    new_data.sort_index(by='count', ascending=False, inplace=True)

    return new_data


def make_diff_per_match_dataset(data: pd.DataFrame):
    """Make score difference dataset.

    Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
    'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
    'team2Score', 'team1PenScore', 'team2PenScore','team1Home',
    'team2Home', 'neutralVenue', 'friendly', 'qualifier', 'finaltourn']
    """

    print('# Do `make_diff_per_match_dataset`.')

    groups = data.groupby([data.team1Score - data.team2Score])

    new_rows = []
    for index, group in groups:
        team1_score, team2_score = int(group.team1Score.iloc[0]), int(group.team2Score.iloc[0])
        new_rows.append({'goalDiff': team1_score - team2_score,
                         'count': len(group),
                         'freq': len(group) / len(data)})

    new_data = pd.DataFrame(new_rows)
    new_data.sort_index(by='count', ascending=False, inplace=True)

    return new_data


def make_opponent_dataset(team_perf: pd.DataFrame, team_info: pd.DataFrame):
    """
    Take team confederation into account.

    Add two columns: 'opponentName' and 'opponentConfederationCoefficient'.
    'opponentConfederationCoefficient' is  adjustment coefficients to each conference,
    in a similar way to how FIFA's World ranking algorithm accounts for regional strength.

    team_perf data structure:
    ['match_id', 'date', 'name', 'opponentName', 'homeVenue', 'neutralVenue',
    'gs', 'ga', 'gd', 'win', 'loss', 'draw', 'friendly', 'qualifier', 'finaltourn']

    team_info data structure:
    ['confederation', 'name', 'fifa_code', 'ioc_code']
    """

    print('# Do `make_opponent_dataset`.')

    coefficient_df = pd.DataFrame({"confederation": ["UEFA", "CONMEBOL", "CONCACAF", "AFC", "CAF", "OFC"],
                                   "adjust": [0.99, 1.00, 0.85, 0.85, 0.85, 0.85]})

    # not_defined = team_info.fifa_code == '---'
    # team_info.loc[not_defined, 'fifa_code'] = team_info.loc[not_defined, 'ioc_code']
    # team_info.drop_duplicates(subset=['fifa_code'], inplace=True)
    team_info = team_info.merge(coefficient_df, how='left', on='confederation')

    merged_df = team_perf.merge(team_info, left_on=['opponentName'], right_on=['fifa_code'],
                                how='left', suffixes=['', '_y'])

    merged_df = merged_df.drop(columns=['name_y', 'fifa_code', 'ioc_code'])

    merged_df.rename(columns={'adjust': 'opponentCC'}, inplace=True)

    # Eliminate NaN.
    merged_df.loc[merged_df['opponentCC'].isna(), 'opponentCC'] = 1.0
    merged_df = merged_df[['match_id', 'date', 'name', 'opponentName', 'opponentCC',
                           'homeVenue', 'neutralVenue', 'gs', 'ga', 'gd', 'win', 'loss', 'draw',
                           'friendly', 'qualifier', 'finaltourn']]

    return merged_df


def make_team_features_dataset(opponent_info_df: pd.DataFrame):
    """
    Feature Engineering
    Now, let's calculate some lag features for each team which is about to play a game.
    We'll look at the previous N games a team has played, up to the game in question,
    and we'll calculate the percentage of wins, draws, losses, as well as the goal differential,
    per game, for those past N games.

    For example, taking N=10:

    last10games_w_per = (number of wins in the past 10 games) / 10
    last10games_d_per = (number of draws in the past 10 games) / 10
    last10games_l_per = (number of losses in the past 10 games) / 10
    last10games_gd_per = (goals scored - goals conceeded in the past 10 games) / 10

    We'll use three different values for N (10, 30 and 50) to capture short-, medium-, and long-term form.
    We'll calculate those values for every team and every game in our dataset.

    To model the strength of opposition faced, we'll use the same technique with respect to the
    opponentConfederationCoefficient values we introduced earlier.

    `opponent_info` data structure.
    ['match_id', 'date', 'name', 'opponentName', 'homeVenue', 'neutralVenue',
    'gs', 'ga', 'gd', 'win', 'loss', 'draw', 'friendly', 'qualifier', 'finaltourn'
    'confederation', 'name_y', 'fifa_code', 'ioc_code'，'opponentCC']
    """

    print('# Do `make_team_features_dataset`.')

    opponent_info_df.sort_values(by='date', inplace=True)

    # Last 10 games info.
    print('#    Computing last 10 games info...')
    last_10_per = opponent_info_df.groupby('name').rolling(10, min_periods=1)[
        'win', 'loss', 'draw', 'gd', 'opponentCC'].sum().reset_index(level=0)
    last_10_per.loc[:, ['win', 'loss', 'draw', 'gd', 'opponentCC']] /= 10
    last_10_per.rename(columns={'win': 'last10win_per', 'loss': 'last10loss_per',
                                'draw': 'last10draw_per', 'gd': 'last10gd_per',
                                'opponentCC': 'last10_oppCC_per'}, inplace=True)
    last_10_per.pop('name')
    opponent_info_df = pd.concat([opponent_info_df, last_10_per], axis=1)

    # Last 30 games info.
    print('#    Computing last 30 games info...')
    last_30_per = opponent_info_df.groupby('name').rolling(30, min_periods=1)[
        'win', 'loss', 'draw', 'gd', 'opponentCC'].sum().reset_index(level=0)
    last_30_per.loc[:, ['win', 'loss', 'draw', 'gd', 'opponentCC']] /= 30
    last_30_per.rename(columns={'win': 'last30win_per', 'loss': 'last30loss_per',
                                'draw': 'last30draw_per', 'gd': 'last30gd_per',
                                'opponentCC': 'last30_oppCC_per'}, inplace=True)
    last_30_per.pop('name')
    opponent_info_df = pd.concat([opponent_info_df, last_30_per], axis=1)

    # Last 50 games info.
    print('#    Computing last 50 games info...')
    last_50_per = opponent_info_df.groupby('name').rolling(50, min_periods=1)[
        'win', 'loss', 'draw', 'gd', 'opponentCC'].sum().reset_index(level=0)
    last_50_per.loc[:, ['win', 'loss', 'draw', 'gd', 'opponentCC']] /= 50
    last_50_per.rename(columns={'win': 'last50win_per', 'loss': 'last50loss_per',
                                'draw': 'last50draw_per', 'gd': 'last50gd_per',
                                'opponentCC': 'last50_oppCC_per'}, inplace=True)
    last_50_per.pop('name')
    opponent_info_df = pd.concat([opponent_info_df, last_50_per], axis=1)

    opponent_info_df = opponent_info_df[[
        'match_id', 'date', 'name', 'opponentName', 'gs', 'ga',
        'win', 'last10win_per', 'last30win_per', 'last50win_per',
        'loss', 'last10loss_per', 'last30loss_per', 'last50loss_per',
        'draw', 'last10draw_per', 'last30draw_per', 'last50draw_per',
        'gd', 'last10gd_per', 'last30gd_per', 'last50gd_per',
        'opponentCC', 'last10_oppCC_per', 'last30_oppCC_per', 'last50_oppCC_per']]

    return opponent_info_df


def make_matches_features_dataset(
        matches_no_friendly: pd.DataFrame, team_features: pd.DataFrame):
    """
    Now that we have built a series of team-specific features,
    we need to fold them back into match-specific features.
    We will then have a set of features for both teams about to face each other.

    `matches_features` data structure:
     Data structure:
    ['date', 'team1', 'team1Text', 'team2', 'team2Text', 'resText',
    'statText', 'venue', 'IdCupSeason', 'CupName', 'team1Score',
    'team2Score', 'team1PenScore', 'team2PenScore','team1Home',
    'team2Home', 'neutralVenue', 'friendly', 'qualifier', 'finaltourn']

    `team_features` data structure.
    ['date', 'draw', 'finaltourn', 'friendly', 'ga', 'gd', 'gs', 'homeVenue',
    'loss', 'match_id', 'name', 'neutralVenue', 'opponentName',
    'qualifier', 'win', 'confederation', 'name_y', 'fifa_code', 'ioc_code',
    'opponentCC', 'last10win_per', 'last10loss_per', 'last10draw_per',
    'last10gd_per', 'last10_oppCC_per', 'last30win_per', 'last30loss_per',
    'last30draw_per', 'last30gd_per', 'last30_oppCC_per', 'last50win_per',
    'last50loss_per', 'last50draw_per', 'last50gd_per', 'last50_oppCC_per']
    """
    print('# Do `make_matches_features_dataset`.')

    matches_no_friendly = matches_no_friendly[['date', 'match_id', 'team1', 'team2', 'team1Home', 'team2Home',
                                               'neutralVenue', 'friendly', 'qualifier', 'finaltourn']]
    team_features = team_features[['match_id', 'name', 'gd', 'last10win_per', 'last10loss_per', 'last10draw_per',
                                   'last10gd_per', 'last10_oppCC_per', 'last30win_per', 'last30loss_per',
                                   'last30draw_per', 'last30gd_per', 'last30_oppCC_per', 'last50win_per',
                                   'last50loss_per', 'last50draw_per', 'last50gd_per', 'last50_oppCC_per']]

    matches_features = matches_no_friendly.merge(
        team_features, left_on=['match_id', 'team1'], right_on=['match_id', 'name'],
        how='left', suffixes=('', '_y'))

    matches_features = matches_features.merge(
        team_features, left_on=['match_id', 'team2'], right_on=['match_id', 'name'],
        how='left', suffixes=('.t1', '.t2'))

    matches_features['outcome'] = matches_features['gd.t1']

    # Drop some columns.
    matches_features.drop(columns=['name.t1', 'name.t2', 'match_id', 'gd.t1', 'gd.t2'],
                          axis=1, inplace=True)

    return matches_features


def make_wc2018_dataset(
        matches_features: pd.DataFrame,
        team_features: pd.DataFrame,
        wc2018_qualified: pd.DataFrame):
    """
    Simulating the Tournament

    With a trained model at our disposal, we can now run tournament simulations on it.
    For example, let's take the qualified teams for the FIFA 2018 World Cup.

    `team_features` data structure.
    ['date', 'draw', 'finaltourn', 'friendly', 'ga', 'gd', 'gs', 'homeVenue',
    'loss', 'match_id', 'name', 'neutralVenue', 'opponentName',
    'qualifier', 'win', 'confederation', 'name_y', 'fifa_code', 'ioc_code',
    'opponentCC', 'last10win_per', 'last10loss_per', 'last10draw_per',
    'last10gd_per', 'last10_oppCC_per', 'last30win_per', 'last30loss_per',
    'last30draw_per', 'last30gd_per', 'last30_oppCC_per', 'last50win_per',
    'last50loss_per', 'last50draw_per', 'last50gd_per', 'last50_oppCC_per']
    """
    print('# Do `make_wc2018_dataset`.')

    def expand_grid(dictionary):
        return pd.DataFrame([row for row in product(*dictionary.values())],
                            columns=dictionary.keys())

    # Make all possible match pairs.
    to_predict = expand_grid(
        {'team1': wc2018_qualified.name.values, "team2": wc2018_qualified.name.values})
    to_predict = to_predict[to_predict.team1 < to_predict.team2]

    # Select the latest game stats.
    team_features = team_features.sort_values('date').groupby(by='name').last().reset_index()

    # Preprocess team_features.
    team_features = team_features[
        ['name', 'gd', 'last10win_per', 'last10loss_per', 'last10draw_per',
         'last10gd_per', 'last10_oppCC_per', 'last30win_per', 'last30loss_per',
         'last30draw_per', 'last30gd_per', 'last30_oppCC_per', 'last50win_per',
         'last50loss_per', 'last50draw_per', 'last50gd_per', 'last50_oppCC_per']]

    # Add historical info.
    to_predict = to_predict.merge(
        team_features, left_on=['team1'], right_on=['name'], how='left')
    to_predict = to_predict.merge(
        team_features, left_on=['team2'], right_on=['name'], how='left', suffixes=('.t1', '.t2'))
    to_predict['outcome'] = to_predict['gd.t1']
    to_predict = to_predict.drop(columns=['name.t1', 'name.t2', 'gd.t1', 'gd.t2'])

    length = len(to_predict)
    extra_features = pd.DataFrame(
        {'date': ['2018-06-14'] * length, 'team1Home': to_predict.team1 == 'RUS',
         'team2Home': to_predict.team2 == 'RUS',
         'neutralVenue': (to_predict.team1 != 'RUS') & (to_predict.team2 != 'RUS'),
         'friendly': [False] * length, 'qualifier': [False] * length, 'finaltourn': [True] * length})

    to_predict = pd.concat([extra_features, to_predict], axis=1)

    # Keep columns the same order as matches_features.
    to_predict = to_predict[matches_features.columns]

    return to_predict


if __name__ == '__main__':
    _input_dir = os.path.join(os.getcwd(), 'data', 'input')
    _preprocess_dir = os.path.join(os.getcwd(), 'data', 'preprocess')

    if not os.path.exists(_input_dir):
        os.makedirs(_input_dir)
    if not os.path.exists(_preprocess_dir):
        os.makedirs(_preprocess_dir)

    # Clean up
    _matches = read_csv(os.path.join(_input_dir, 'matches.csv'))
    _matches = clean_up(_matches)

    # Exchange team pos
    _matches_exchanged = exchange_team_order(_matches)

    # Add additional features.
    _matches_add_features = add_additional_features(_matches_exchanged)

    # Drop friendly matches.
    _matches_no_friendly = drop_friendly_match(_matches_add_features)
    _save_path = os.path.join(_preprocess_dir, 'matches_no_friendly.csv')
    _matches_no_friendly.to_csv(_save_path, index=False)

    # Make team performance dataset.
    _team_perf = make_team_perf_dataset(_matches_no_friendly, True)
    _save_path = os.path.join(_preprocess_dir, 'team_perf.csv')
    _team_perf.to_csv(_save_path, index=False)

    # Make score frequency dataset.
    _score_freq = make_score_freq_dataset(_matches_no_friendly)
    _save_path = os.path.join(_preprocess_dir, 'score_freq.csv')
    _score_freq.to_csv(_save_path, index=False)

    # Make score sum dataset.
    _goal_sum_per_match = make_sum_per_match_dataset(_matches_no_friendly)
    _save_path = os.path.join(_preprocess_dir, 'goal_sum_per_match.csv')
    _goal_sum_per_match.to_csv(_save_path, index=False)

    # Make score diff dataset.
    _goal_diff_per_match = make_diff_per_match_dataset(_matches_no_friendly)
    _save_path = os.path.join(_preprocess_dir, 'goal_diff_per_match.csv')
    _goal_diff_per_match.to_csv(_save_path, index=False)

    # Opponent dataset.
    _team_perf = read_csv(os.path.join(_preprocess_dir, 'team_perf.csv'), strip_values=True)
    _team_info = read_csv(os.path.join(_input_dir, 'teams.csv'), strip_values=True)
    _opponent_info = make_opponent_dataset(_team_perf, _team_info)
    _save_path = os.path.join(_preprocess_dir, 'opponent_info.csv')
    _opponent_info.to_csv(_save_path, index=False)

    # Team features dataset.
    _opponent_info = read_csv(os.path.join(_preprocess_dir, 'opponent_info.csv'))
    _team_features = make_team_features_dataset(_opponent_info)
    _save_path = os.path.join(_preprocess_dir, 'team_features.csv')
    _team_features.to_csv(_save_path, index=False)

    # Matches features.
    # matches_no_friendly = read_csv('../data/matches_no_friendly.csv')
    # team_features = read_csv('../data/team_features.csv')
    _matches_features = make_matches_features_dataset(_matches_no_friendly, _team_features)
    _save_path = os.path.join(_preprocess_dir, 'matches_features.csv')
    _matches_features.to_csv(_save_path, index=False)

    # Split dataset.
    _matches_no_friendly = read_csv(os.path.join(_preprocess_dir, 'matches_no_friendly.csv'))
    _matches_features = read_csv(os.path.join(_preprocess_dir, 'matches_features.csv'))
    _train_dataset = _matches_features[_matches_features.date < '2009-01-01']
    _test_dataset = _matches_features[(_matches_features.date >= '2009-01-01') &
                                      (_matches_features.date < '2015-01-01')]

    _save_path = os.path.join(_preprocess_dir, 'train_dataset.csv')
    _train_dataset.to_csv(_save_path, index=False)
    _save_path = os.path.join(_preprocess_dir, 'test_dataset.csv')
    _test_dataset.to_csv(_save_path, index=False)

    # Make final predict dataset.
    _matches_features = read_csv(os.path.join(_preprocess_dir, 'matches_features.csv'))
    _team_features = read_csv(os.path.join(_preprocess_dir, 'team_features.csv'))
    _wc2018_qualified = read_csv(os.path.join(_input_dir, 'wc2018_qualified.csv'))
    _matches_to_predict = make_wc2018_dataset(_matches_features, _team_features, _wc2018_qualified)
    _matches_to_predict.to_csv(os.path.join(_preprocess_dir, 'matches_to_predict.csv'), index=False)
