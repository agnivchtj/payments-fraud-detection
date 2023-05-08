# model.py

from lightgbm import LGBMClassifier
from joblib import dump


def load():
    # LightGBM classifier
    clf_lgbm = LGBMClassifier(verbose=0)
    dump(clf_lgbm, 'lightgbm_default.joblib')


if __name__ == '__main__':
    load()
