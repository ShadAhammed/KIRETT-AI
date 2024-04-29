from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

class Tuner:
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    def __init__(self,params, model,X_train,y_train):
        self.params= params
        self.model= model
        self.X_train= X_train
        self.y_train= y_train

    def GridSearch(self):
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.params, n_jobs=-1, cv=3, scoring='accuracy',error_score=0, verbose= 0)
        grid_result = grid_search.fit(self.X_train, self.y_train)
        print("Grid search result: Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        #for mean, stdev, param in zip(means, stds, params):
            #print("%f (%f) with: %r" % (mean, stdev, param))
        return grid_search.best_score_, grid_search.best_estimator_

    def RandomSearch(self):
        random_search = RandomizedSearchCV(self.model, param_distributions=self.params, scoring='accuracy',n_jobs=-1, cv=3, verbose= 0, random_state=1001)
        random_result = random_search.fit(self.X_train, self.y_train)
        print("Random search result: Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
        means = random_result.cv_results_['mean_test_score']
        stds = random_result.cv_results_['std_test_score']
        params = random_result.cv_results_['params']
        #for mean, stdev, param in zip(means, stds, params):
            #print("%f (%f) with: %r" % (mean, stdev, param))
        return random_search.best_score_, random_search.best_estimator_





