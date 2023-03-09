import pandas as pd
import xgboost
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def data_preprocessing(dfs):
    same_cols = ['client_id', 'customer_since_all',
                 'customer_since_bank', 'customer_birth_date',
                 'customer_gender',
                 'customer_postal_code', 'customer_occupation_code',
                 'customer_education']
    dfs_1_2 = pd.merge(dfs[0], dfs[1], on=same_cols, suffixes=('_first_month', None))
    merged_df = pd.merge(dfs_1_2, dfs[2], on=same_cols, suffixes=('_second_month', '_third_month'))

    merged_df['customer_birth_date'] = merged_df['customer_birth_date'].apply(lambda k: int(k.split('-')[0]))

    merged_df['customer_since_all'] = merged_df['customer_since_all'].fillna('1800-01')
    merged_df['customer_since_all'] = merged_df['customer_since_all'].apply(lambda k: int(k.split('-')[0]))

    merged_df['customer_since_bank'] = merged_df['customer_since_bank'].fillna('1800-01')
    merged_df['customer_since_bank'] = merged_df['customer_since_bank'].apply(lambda k: int(k.split('-')[0]))

    merged_df['customer_occupation_code'] = merged_df['customer_occupation_code'].fillna(10.0)

    merged_df = merged_df.dropna(axis=1)
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

params = {
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0.5, 1, 2, 5, 10, 20],
        'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [10, 20, 50, 100, 150, 200, 250, 300, 350],
        'max_depth': [2, 3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.02, 0.05, 0.10, 0.15],
        'scale_pos_weight': [30, 33, 34, 36, 40]
        }

folds = 4
param_comb = 1000

xgb_model = xgboost.XGBClassifier(objective='binary:logistic', eval_metric='auc',
                                  use_label_encoder=False)
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(xgb_model, param_distributions=params,
                                   n_iter=param_comb, scoring='roc_auc',
                                   n_jobs=4, cv=skf.split(train_X_df, train_y_df),
                                   verbose=3, random_state=42)

random_search.fit(train_X_df, train_y_df)

xgb_preds_proba = random_search.predict_proba(test_X_df)

preds_df = pd.DataFrame()
preds_df['ID'] = test_dfs[0]['client_id']
preds_df['PROB'] = xgb_preds_proba[:, 1]
preds_df.to_csv('preds.csv', sep=',', index=False)

print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best score:')
print(random_search.best_score_)
print('\n Best hyperparameters:')
print(random_search.best_params_)

explainer = shap.Explainer(random_search.best_estimator_)
shap_values = explainer(train_X_df)
plt.figure()
shap.plots.bar(shap_values)
plt.tight_layout()