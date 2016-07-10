import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def tune_params(X_train, y_train, alg):
    from sklearn.grid_search import GridSearchCV

    # Random Forest
    if alg == 'rf':
        # current best
        # {'min_samples_split': 10, 'max_features': 0.5, 'n_estimators': 100, 'criterion': 'gini', 'min_samples_leaf': 5,
        #  'max_depth': 3.0}
        param_grid = [{'n_estimators': [150, 200, 300],
                       'criterion': ['gini', 'entropy'],
                       'min_samples_split': [2, 3, 5],
                       'min_samples_leaf': [3, 5, 8, 10],
                        'max_depth':[3.0, 5.0, 8.0, 10.0],
                        'max_features':[0.5, 0.8, 0.9]
                       }]
        gs = GridSearchCV(estimator=RandomForestClassifier(n_estimators=1000),
                          param_grid=param_grid,
                          scoring='accuracy',
                          verbose=3,
                          cv=5,
                          n_jobs=-1)
    elif alg == 'gb':
        #current best
        # {'n_estimators': 200, 'max_features': 'log2', 'min_samples_leaf': 10, 'min_samples_split': 5, 'max_depth': 2.0,
        #  'learning_rate': 0.2}
        param_grid = [{'n_estimators': [100, 150, 200],
                       'learning_rate': [0.1, 0.2, 0.3],
                       'min_samples_split': [3, 5, 8, 10],
                       'min_samples_leaf': [10, 12, 15],
                       'max_depth': [2.0, 3.0, 5.0, 8.0],
                       'max_features': ['auto', 'sqrt', 'log2']
                       }]
        gs = GridSearchCV(estimator=GradientBoostingClassifier(),
                          param_grid=param_grid,
                          scoring='accuracy',
                          verbose=1,
                          cv=5,
                          n_jobs=-1)

    gs.fit(X_train, y_train)
    print(gs.best_params_)


def create_submission(model, X_train, y_train, test, filename):
    print('training...')
    model.fit(X_train, y_train)

    print('predicting...')
    pred = model.predict(test)
    print('creating submission...')
    submission = pd.DataFrame({'PassengerId': test.index.values,
                               'Survived': pred})
    submission.to_csv(filename, index=False)

    print('file saved at ' + os.path.realpath('.') + filename)


def clean_data(data):
    from sklearn.preprocessing import LabelEncoder

    def parse_des(x):
        x = re.findall(r',\s\w+.', x)
        return (re.findall(r'\w+', str(x)))[0]

    sex_le = LabelEncoder()
    data['Sex'] = sex_le.fit_transform(data['Sex'])

    # assign missing Embarked column to 'S' as it has vast majority
    data.loc[data['Embarked'].isnull(), 'Embarked'] = 'S'

    # Determine the Age typical for each passenger class by Sex.
    # use the median instead of the mean because the Age
    # histogram seems to be right skewed.
    data['Age'] = data['Age'].groupby([data['Sex'], data['Pclass']]).apply(lambda x: x.fillna(x.median()))

    data['Fare'] = data['Fare'].groupby(data['Pclass']).apply(lambda x: x.fillna(x.median()))

    # new feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data.loc[data['FamilySize'] == 1, 'DiscretizedFSize'] = 'singleton'
    data.loc[(data['FamilySize'] > 1) & (data['FamilySize'] < 5), 'DiscretizedFSize'] = 'small'
    data.loc[data['FamilySize'] > 5, 'DiscretizedFSize'] = 'large'

    data['Title'] = data['Name'].apply(parse_des)
    data.loc[data['Title'] == 'Mlle', 'Title'] = 'Miss'
    data.loc[data['Title'] == 'Ms', 'Title'] = 'Miss'
    data.loc[data['Title'] == 'Mme', 'Title'] = 'Mrs'

    rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don',
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'the']
    data.loc[data['Title'].isin(rare_titles), 'Title'] = 'Other'

    # Transform Embarked, DiscretizedFSize and Title from a string to dummy variables
    data = pd.concat([data, pd.get_dummies(data['Embarked'], prefix='Embarked')], axis=1)
    data = pd.concat([data, pd.get_dummies(data['Title'], prefix='Title')], axis=1)
    data = data = pd.concat([data, pd.get_dummies(data['DiscretizedFSize'], prefix='DiscretizedFSize')], axis=1)

    # drop unused features
    data.drop(['Title', 'Name', 'Cabin', 'Ticket', 'Embarked', 'SibSp', 'Parch', 'DiscretizedFSize'], axis=1, inplace=True)

    print(data.columns.values)
    return data


def cv_report(X, y, model):
    from sklearn import cross_validation

    kfold = cross_validation.KFold(n=len(X), n_folds=10, random_state=1)
    scores = cross_validation.cross_val_score(
        model,
        X,
        y,
        # scoring='log_loss',
        cv=kfold,
    )
    print(scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


train_df = pd.read_csv('data/train.csv', index_col='PassengerId')
test_df = pd.read_csv('data/test.csv', index_col='PassengerId')

y = train_df['Survived']
train_df.drop('Survived', inplace=True, axis=1)

X = clean_data(train_df)
test = clean_data(test_df)

model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                               min_samples_split=10, criterion='gini',
                               max_depth=3.0, max_features=0.9)

# model = RandomForestClassifier(n_estimators=200)
# model.fit(X, y)
# features_imp = pd.DataFrame(model.feature_importances_, index=X.columns,
#                             columns=["importance"]).sort_values(['importance'], ascending=False)
# print('feature importance', features_imp)

cv_report(X, y, model)
# tune_params(X, y, 'rf')


# create_submission(model, X, y, test, 'data/submission.csv')




