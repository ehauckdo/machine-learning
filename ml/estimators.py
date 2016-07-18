from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import cross_validation

def perform_logistic_regression(train_set, train_target, test_set, predictors):
    alg = LogisticRegression(random_state=1)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    scores = cross_validation.cross_val_score(alg, train_set[predictors], train_target, cv=3)
    print("LR scores: "+str(scores.mean()))
    return predictions

def perform_random_forest(train_set, train_target, test_set, predictors, estimators=178, splits=6, leafs=4):
    alg = RandomForestClassifier(random_state=1, n_estimators=estimators, min_samples_split=splits, min_samples_leaf=leafs)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    scores = cross_validation.cross_val_score(alg, train_set[predictors], train_target, cv=3)
    print("RF scores: "+str(scores.mean()))
    return predictions;

def perform_gradient_boosting(train_set, train_target, test_set, predictors, estimators=30, depth=3):
    alg = GradientBoostingClassifier(random_state=1, n_estimators=estimators, max_depth=depth)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    scores = cross_validation.cross_val_score(alg, train_set[predictors], train_target, cv=3)
    print("GB scores: "+str(scores.mean()))
    return predictions;

def perform_SVM(train_set, train_target, test_set, predictors):
    alg = SVC()
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    scores = cross_validation.cross_val_score(alg, train_set[predictors], train_target, cv=3)
    print("SVM scores: "+str(scores.mean()))
    return predictions

