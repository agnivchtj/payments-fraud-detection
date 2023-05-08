# model.py

from catboost import CatBoostClassifier
from joblib import dump


def load():
    # CatBoost
    clf_cat = CatBoostClassifier(verbose=False)
    dump(clf_cat, 'catboost_default.joblib')


if __name__ == '__main__':
    load()
