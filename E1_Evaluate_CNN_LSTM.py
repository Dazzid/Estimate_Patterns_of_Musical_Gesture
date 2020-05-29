# cnn_lstm
import numpy as np
import csv
import tensorflow as tf
from matplotlib import pyplot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os

# load a single file as a numpy array
def load_file(filepath):
    data = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)
    return np.array(data)

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group
    # load all 9 files as a single array
    filenames = ['01_acc_x.csv', '02_acc_y.csv', '03_acc_z.csv',
                 '04_gyro_x.csv', '05_gyro_y.csv', '06_gyro_z.csv',
                 '07_euler_x.csv', '08_euler_y.csv', '09_euler_z.csv']
    # load input data
    X = load_group(filenames, filepath).astype(np.float64)
    # load class output
    y = load_file(prefix + group + '10_label.csv').astype(np.int)
    return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train/', prefix + 'data/Gestures/Groups/')
    # load all test
    testX, testy = load_dataset_group('test/', prefix + 'data/Gestures/Groups/')
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = tf.keras.utils.to_categorical(trainy)
    testy = tf.keras.utils.to_categorical(testy)
    return trainX, trainy, testX, testy

def non_nan_average(x):
    # Computes the average of all elements that are not NaN in a rank 1 tensor
    nan_mask = tf.math.is_nan(x)
    x = tf.boolean_mask(x, tf.logical_not(nan_mask))
    return tf.keras.backend.mean(x)

def uar_accuracy(y_true, y_pred):
    c_mat = confusion_matrix(y_true, y_pred)
    print(c_mat)

    # These operations needed for image summary
    cf_mat1 = tf.cast(c_mat, tf.dtypes.float32)
    cf_mat1 = tf.expand_dims(cf_mat1, 2)
    cf_mat1 = tf.expand_dims(cf_mat1, 0)
    print(cf_mat1)

    diag = tf.linalg.tensor_diag_part(c_mat)
    # Calculate the total number of data examples for each class
    total_per_class = tf.reduce_sum(c_mat, axis=1)
    acc_per_class = diag / tf.maximum(1, total_per_class)
    uar = non_nan_average(acc_per_class)
    return uar

# fit and evaluate a model
def create_model(features, outpus,lSize):
    n_features, n_outputs = features, outpus
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 3, 50 #best so far is 5 30
    # define model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(lSize, 3, activation='relu'), input_shape=(None,n_length,n_features)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(lSize, 3, activation='relu')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.5)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D()))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    tf.keras.utils.plot_model(model, show_shapes=False, show_layer_names=True, to_file='figures/CNN_LSTM.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# summarize scores
def summarize_results(scores, params, saveit = False):
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = np.mean(scores[i]), np.std(scores[i])
        print('Param = %d: %.3f%% (+/-%.3f)' % (params[i], m, s))
    # boxplot of scores
    pyplot.boxplot(scores, labels=params)
    if saveit:
        pyplot.savefig('figures/batches_CNN_LSTM_2.png')
    pyplot.show()

# run an experiment
def run_experiment(repeats=1):
    sizes = 256
    checkpoint_path = 'training_1/cp_'+ str(sizes)+'.ckpt'
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    final_scores = list()

    n_features, n_outputs = trainX.shape[2], trainy.shape[1]
    model = create_model(n_features, n_outputs, sizes)
    # Display the model's architecture
    model.summary()
    # Loads the weights
    model.load_weights(checkpoint_path)

    # define model
    # reshape data into time steps of sub-sequences
    n_steps, n_length = 3, 50  # best so far is 5 30
    for i in range(len(testX)):
        print(testX[i].shape, testy[i])
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    loss, acc = model.evaluate(testX, testy, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# run the experiment
run_experiment()

