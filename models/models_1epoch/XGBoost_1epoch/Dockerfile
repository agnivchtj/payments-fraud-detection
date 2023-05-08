FROM jupyter/scipy-notebook

RUN pip install joblib
RUN pip install xgboost

COPY model.py ./model.py

RUN python3 model.py