import numpy as np
# from attention_forest.model import *
from attention_forest import TaskType, ForestKind, EAFParams, EpsAttentionForest
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1


def run_example():
    print("Attention Forest:")
    X, y = make_friedman1(random_state=12345)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12345)

    task = TaskType.REGRESSION
    model = EpsAttentionForest(EAFParams(
        kind=ForestKind.RANDOM,
        task=task,
        forest=dict(
            n_estimators=100,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=12345,
        ),
        eps=0.5,
        tau=1.0,
    ))

    model.fit(X_train, y_train)
    if task == TaskType.REGRESSION:
        metric = r2_score
        forest_predict = lambda x: model.forest.predict(x)
    else:
        metric = lambda a, b: roc_auc_score(a, b[:, 1])
        forest_predict = lambda x: model.forest.predict_proba(x)
    print("Forest score (train):", metric(y_train, forest_predict(X_train)))
    print("Forest score (test):", metric(y_test, forest_predict(X_test)))

    print("Before opt score (train):", metric(y_train, model.predict(X_train)))
    print("Before opt score (test):", metric(y_test, model.predict(X_test)))

    model.optimize_weights(X_train, y_train)
    print("After opt score (train):", metric(y_train, model.predict(X_train)))
    print("After opt score (test):", metric(y_test, model.predict(X_test)))


if __name__ == "__main__":
    run_example()
