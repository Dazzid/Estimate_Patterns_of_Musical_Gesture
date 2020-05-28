# spot check on raw data from the har dataset
from numpy import dstack
from numpy import vstack
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delimiter=',')
    return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['01_acc_x.csv', '02_acc_y.csv', '03_acc_z.csv']
    # body acceleration
    filenames += ['04_gyro_x.csv', '05_gyro_y.csv', '06_gyro_z.csv']
    # body gyroscope
    filenames += ['07_euler_x.csv', '08_euler_y.csv', '09_euler_z.csv']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '10_label.csv')
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train/', prefix + 'data/Gestures/Groups/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test/', prefix + 'data/Gestures/Groups/')
    print(testX.shape, testy.shape)
    # flatten X
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1] * trainX.shape[2]))
    testX = testX.reshape((testX.shape[0], testX.shape[1] * testX.shape[2]))
    # flatten y
    trainy, testy = trainy[:,0], testy[:,0]
    return trainX, trainy, testX, testy

# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
    n_estimators = 100
    # nonlinear models
    models['KNeighbors'] = KNeighborsClassifier(n_neighbors=3)
    models['Decision Tree'] = DecisionTreeClassifier(random_state=1, criterion='entropy', splitter='random')
    models['SVM'] = SVC(gamma='auto', kernel="linear", C=0.15, class_weight='balanced')
    models['Naive Bayes'] = GaussianNB()
    models['Gaussian Process'] = GaussianProcessClassifier(1.0 * RBF(1.33))
    # ensemble models
    models['Bagging'] = BaggingClassifier(n_estimators=n_estimators)
    models['Random Forest'] = RandomForestClassifier(n_estimators=n_estimators)
    models['Extra Trees'] = ExtraTreesClassifier(n_estimators=n_estimators)
    models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=n_estimators)
    print('Defined %d models' % len(models))
    return models

# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
    # fit the model
    model.fit(trainX, trainy)
    # make predictions
    yhat = model.predict(testX)
    # evaluate predictions
    accuracy = accuracy_score(testy, yhat)
    return accuracy * 100.0

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
    results = dict()
    for name, model in models.items():
        # evaluate the model
        results[name] = evaluate_model(trainX, trainy, testX, testy, model)
        # show process
        print('>%s: %.3f' % (name, results[name]))
    return results

# print and plot the results
def summarize_results(results, maximize=True):
    # create a list of (name, mean(scores)) tuples
    mean_scores = [(k,v) for k,v in results.items()]
    # sort tuples by mean score
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    # reverse for descending order (e.g. for accuracy)
    if maximize:
        mean_scores = list(reversed(mean_scores))
    print()
    for name, score in mean_scores:
        print('%s, classifier &%.3f' % (name, score))

# load dataset
trainX, trainy, testX, testy = load_dataset()
# get model list
models = define_models()
# evaluate models
results = evaluate_models(trainX, trainy, testX, testy, models)
# summarize results
summarize_results(results)