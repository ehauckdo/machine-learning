from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import numpy


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

def perform_svm(train_set, train_target, test_set, predictors):
    alg = SVC()
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    scores = cross_validation.cross_val_score(alg, train_set[predictors], train_target, cv=3)
    print("SVM scores: "+str(scores.mean()))
    return predictions

def perform_ensemble(algorithms, train_set, train_target, test_set):
    total_predictions = []
    for algorithm, predictor in algorithms:
        algorithm.fit(train_set[predictor], train_target)
        prediction = algorithm.predict_proba(test_set[predictor].astype(float))[:,1]
        total_predictions.append(prediction)

    # average scores 
    predictions = (total_predictions[0]*1  + total_predictions[1]*1)/ 2

    predictions[predictions <= .5] = 0
    predictions[predictions > .5] = 1
    predictions = predictions.astype(int)
    return predictions

# ensemble functions
def score_ensemble(algorithms, train_set, train_target):
    
    kf = KFold(train_set.shape[0], n_folds = 3, random_state=1)

    predictions = []
    for train, test in kf:

        full_test_predictions = []
        for alg, predictors in algorithms:

            fold_train_target = train_target.iloc[train]
            fold_train_predictors = train_set[predictors].iloc[train,:]

            alg.fit(fold_train_predictors, fold_train_target)

            probability_array = alg.predict_proba(train_set[predictors].iloc[test,:].astype(float))
            # select probabilty of belonging to "Survived = 1"
            test_predictions = probability_array[:,1] 
            full_test_predictions.append(test_predictions)
        
        test_predictions = (full_test_predictions[0]*1 + full_test_predictions[1]*1) / 2 
        test_predictions[test_predictions <= .5] = 0
        test_predictions[test_predictions > .5] = 1
        predictions.append(test_predictions)    

    # concatenate the three predicted test folders
    predictions = numpy.concatenate(predictions, axis=0)

    # Compute accuracy by comparing to the training data.
    accuracy = sum(predictions[predictions == train_target]) / len(predictions)
    print("Ensemble scores: "+str(accuracy))


# generate chart with correlation feature vs result
def feature_selection(dataframe, predictors, target):
    selector = SelectKBest(f_classif, k=5)
    selector.fit(dataframe[predictors], target)

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = numpy.log10(selector.pvalues_)

    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

