# model.py

from xgboost import XGBClassifier
from joblib import dump


def load():
    clf_xgb = XGBClassifier(verbose=0)
    dump(clf_xgb, 'xgboost_default.joblib')


if __name__ == '__main__':
    load()
