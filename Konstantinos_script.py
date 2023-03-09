import pandas as pd
import xgboost
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
import catboost as cb
from catboost import Pool
import numpy as np
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import optuna


def data_preprocessing(dfs):
    same_cols = ['client_id', 'customer_since_all',
                 'customer_since_bank', 'customer_birth_date',
                 'customer_gender',
                 'customer_postal_code', 'customer_occupation_code',
                 'customer_education']
    dfs_1_2 = pd.merge(dfs[0], dfs[1], on=same_cols, suffixes=('_first_month', None))
    merged_df = pd.merge(dfs_1_2, dfs[2], on=same_cols, suffixes=('_second_month', '_third_month'))
    #
    # merged_df['customer_birth_date'] = merged_df['customer_birth_date'].apply(lambda k: int(k.split('-')[0]))
    #
    # merged_df['customer_since_all'] = merged_df['customer_since_all'].fillna('1800-01')
    # merged_df['customer_since_all'] = merged_df['customer_since_all'].apply(lambda k: int(k.split('-')[0]))
    #
    # merged_df['customer_since_bank'] = merged_df['customer_since_bank'].fillna('1800-01')
    # merged_df['customer_since_bank'] = merged_df['customer_since_bank'].apply(lambda k: int(k.split('-')[0]))

    # merged_df['customer_occupation_code'] = merged_df['customer_occupation_code'].fillna(10.0)

    # merged_df = merged_df.dropna(axis=1)
    merged_df = merged_df.drop('client_id', axis=1)

    return merged_df


folder = 'data'
t_format = '.csv'
train_filenames = ['train_month_1', 'train_month_2', 'train_month_3_with_target']
test_filenames = ['test_month_1', 'test_month_2', 'test_month_3']

train_dfs = [pd.read_csv(Path(folder, filename + t_format)) for filename in train_filenames]
test_dfs = [pd.read_csv(Path(folder, filename + t_format)) for filename in test_filenames]

train_df = data_preprocessing(train_dfs)
train_X_df = train_df.drop('target', axis=1)
train_y_df = train_df['target']
test_X_df = data_preprocessing(test_dfs)

train_rows_nan_df = train_df.isnull().sum()
test_rows_nan_df = test_X_df.isnull().sum()

cat_feats = list(train_rows_nan_df[train_rows_nan_df != 0].index)
cat_feats.append("customer_birth_date")
train_X_df[cat_feats] = train_X_df[cat_feats].astype('str')
test_X_df[cat_feats] = test_X_df[cat_feats].astype('str')

X_train, X_val, y_train, y_val = train_test_split(train_X_df, train_y_df, test_size=0.2, random_state=0)


def objective(trial):
    param = {
        "objective": "Logloss",
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 30, 50),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "used_ram_limit": "8gb",
    }

    # if param["bootstrap_type"] == "Bayesian":
    #     param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    # elif param["bootstrap_type"] == "Bernoulli":
    #     param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    gbm = cb.CatBoostClassifier(**param)

    gbm.fit(X_train, y_train, cat_features=cat_feats,
            eval_set=[(X_val, y_val)],
            verbose=0,
            early_stopping_rounds=100)

    preds = gbm.predict(X_val)
    pred_labels = np.rint(preds)
    accuracy = average_precision_score(y_val, pred_labels)
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, timeout=2 * 60 * 60)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("{}: {}".format(key, value))

optimized_classifier = cb.CatBoostClassifier(objective="Logloss",
                                             depth=study.best_params['depth'],
                                             scale_pos_weight=study.best_params['scale_pos_weight'],
                                             colsample_bylevel=study.best_params['colsample_bylevel'],
                                             boosting_type=study.best_params["boosting_type"],
                                             bootstrap_type=study.best_params["bootstrap_type"])

optimized_classifier.fit(X_train, y_train, cat_features=cat_feats,
                         eval_set=[(X_val, y_val)],
                         verbose=0,
                         early_stopping_rounds=100)

cd_preds_proba = optimized_classifier.predict_proba(test_X_df)
preds_df = pd.DataFrame()
preds_df['ID'] = test_dfs[0]['client_id']
preds_df['PROB'] = cd_preds_proba[:, 1]
preds_df.to_csv('preds.csv', sep=',', index=False)