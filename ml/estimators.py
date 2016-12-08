from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import random
import matplotlib.pyplot as plt
import numpy

def score(alg, train_set, predictors, train_target, title="", cv=3):
    scores = cross_val_score(alg, train_set[predictors], train_target, cv=3)
    print("{0} scores: {1}".format(title, str(scores.mean())))
    print(scores)

def score_logloss(alg, train_set, predictors, train_target, title="", cv=3):
    scores = cross_val_score(alg, train_set[predictors], train_target, cv=3, scoring='log_loss')
    print("{0} log_loss scores: {1}".format(title, str(numpy.var(scores))))
    print(scores)

def perform_logistic_regression(train_set, train_target, test_set, predictors):
    alg = LogisticRegression(random_state=1)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    score(alg, train_set, predictors, train_target, "LR")
    return predictions

def perform_random_forest(train_set, train_target, test_set, predictors, estimators=178, splits=6, leafs=4):
    alg = RandomForestClassifier(random_state=1, n_estimators=estimators, min_samples_split=splits, min_samples_leaf=leafs)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    score(alg, train_set, predictors, train_target, "RF")
    return predictions;

def perform_gradient_boosting(train_set, train_target, test_set, predictors, estimators=30, depth=3):
    alg = GradientBoostingClassifier(random_state=1, n_estimators=estimators, max_depth=depth)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    score(alg, train_set, predictors, train_target, "GB")
    return predictions;

def perform_svm(train_set, train_target, test_set, predictors):
    alg = SVC()
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    score(alg, train_set, predictors, train_target, "SVM")
    return predictions

def perform_gaussianNB(train_set, train_target, test_set, predictors):
    alg = GaussianNB()
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    score(alg, train_set, predictors, train_target, "GaussianNB")
    return predictions

def perform_random_forest_proba(train_set, train_target, test_set, predictors, estimators=178, splits=6, leafs=4):
    alg = RandomForestClassifier(random_state=1, n_estimators=estimators, min_samples_split=splits, min_samples_leaf=leafs)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict_proba(test_set[predictors].astype(float))
    score_logloss(alg, train_set, predictors, train_target, "RF")
    return predictions

def perform_gradient_boosting_proba(train_set, train_target, test_set, predictors, estimators=30, depth=3):
    alg = GradientBoostingClassifier(random_state=1, n_estimators=estimators, max_depth=depth)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict_proba(test_set[predictors].astype(float))
    score_logloss(alg, train_set, predictors, train_target, "GB")
    return predictions;

def perform_gaussianNB_proba(train_set, train_target, test_set, predictors):
    alg = GaussianNB()
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict_proba(test_set[predictors].astype(float))
    score_logloss(alg, train_set, predictors, train_target, "GaussianNB")
    return predictions

def perform_random_forest_regressor(train_set, train_target, test_set, predictors, estimators=10, depth=None, splits=2):
    alg = RandomForestRegressor(random_state=1)
    alg.fit(train_set[predictors], train_target)
    
    #importances = alg.feature_importances_
    #print("Original ",numpy.argsort(importances))
    #indices = numpy.argsort(importances)[::-1]
    #print (" importances ",importances)
    #print (" indices ",indices)
    
    #for f in range(train_set.shape[1]-2):
    #    print("%2d) %-*s %f" % (f+1,30,predictors[indices[f]],
    #                                    importances[indices[f]]))

    predictions = alg.predict(test_set[predictors])
    return predictions;

def perform_gradient_boosting_regressor(train_set, train_target, test_set, predictors):
    alg = GradientBoostingRegressor(random_state=1, n_estimators=100, max_depth=2, learning_rate=0.15)
    alg.fit(train_set[predictors], train_target)
    predictions = alg.predict(test_set[predictors])
    return predictions;

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

# This function uses the new strategy for ensembling models
# Select only models which the CV improved the mean of previous CVs
def perform_ensemble_incremental(algorithms, train_set, train_target, test_set, intermediary_results=10, subset_min=0.7, subset_max=0.8):
    total_predictions = []
    intermediary_predictions = []
    total_scores = []

    def append_predictions():
        intermediary = numpy.mean(total_predictions, axis=0)
        intermediary[intermediary <= .5] = 0
        intermediary[intermediary > .5] = 1
        intermediary = intermediary.astype(int)
        intermediary_predictions.append([intermediary, len(total_predictions), numpy.mean(total_scores)])
    
    for algorithm, predictor in algorithms:
        subset_size = random.uniform(subset_min, subset_max)
        random_state = algorithm.get_params()["random_state"]
        train_set["Outcome"] = train_target
        subset = train_set.sample(frac=subset_size, random_state=random_state)
        subset_target = subset.pop("Outcome")
        score = numpy.sqrt(-cross_val_score(algorithm, subset[predictor], subset_target, scoring='neg_log_loss', cv=3, n_jobs=3))
        if not total_scores or numpy.mean(score) < numpy.mean(total_scores):
            total_scores.append(numpy.mean(score))
            print("{0}: Mean: {1} (New score: {2}, subset_size: {3})".format(len(total_predictions), numpy.mean(total_scores), numpy.mean(score), subset_size))
            algorithm.fit(subset[predictor], subset_target)
            prediction = algorithm.predict_proba(test_set[predictor].astype(float))[:,1]
            total_predictions.append(prediction)
            if len(total_predictions) % intermediary_results is 0:
                append_predictions()
                #intermediary = numpy.mean(total_predictions, axis=0)
                #intermediary[intermediary <= .5] = 0
                #intermediary[intermediary > .5] = 1
                #intermediary = intermediary.astype(int)
                #intermediary_predictions.append([intermediary, len(total_predictions), numpy.mean(total_scores)])

    append_predictions()
    return intermediary_predictions

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
    selector = SelectKBest(f_classif, k='all')
    selector.fit(dataframe[predictors], target)

    # Get the raw p-values for each feature, and transform from p-values into scores
    scores = numpy.log10(selector.pvalues_)

    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

def grid_search(estimator, dataframe, predictors, target):

    X_train, X_test, y_train, y_test = train_test_split(
    dataframe[predictors], dataframe[target], test_size=0.5, random_state=0)
    
    def scorer_classifier(estimator, X, y): 
    
        total = X.shape[0]
        results = estimator.predict(X)
    
        result = 0 
        for i in results:
            result += i
        return result/total

    def scorer_regression(estimator, X, y): 
    
        #print("\nX:")
        #print(X)
        total = X.shape[0]
        results = estimator.predict(X)
    
        #print("\nresults:")
        #print(results)
        result = 0 
        for i in results:
            if i > 0.5:
                result += 1
        return result/total

    svm_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
   
    tuned_parameters = {
        'max_depth': numpy.arange(3, 10),
        'n_estimators': numpy.arange(5, 100),
        'min_samples_split': numpy.arange(2, 8)
    }
 
    print("\n# Tuning hyper-parameters for scorer_regression")
    print()

    clf = GridSearchCV(estimator, tuned_parameters, n_jobs=3, cv=5)
                       #,scoring=scorer_regression) # grid search for svr
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print() 

