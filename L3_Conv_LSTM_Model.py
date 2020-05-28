# convlstm model
import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl

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

    # total acceleration
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

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, batches):
    # define model
    batch_size = batches
    verbose, epochs = 0, 50
    n_features, n_outputs = trainX.shape[2], trainy.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 3, 50
    trainX = trainX.reshape((trainX.shape[0], n_steps, 1, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, 1, n_length, n_features))
    # define model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.ConvLSTM2D(64, (1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    tf.keras.utils.plot_model(model, show_shapes=False, show_layer_names=True, to_file='figues/Conv_LSTM_Model.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(testX, testy))
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy, history

# summarize scores
def summarize_results(scores, params):
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = np.mean(scores[i]), np.std(scores[i])
        print('Param = %d: %.3f%% (+/-%.3f)' % (params[i], m, s))
    # boxplot of scores
    plt.boxplot(scores, labels=params)
    plt.savefig('figures/ConvLSTM2D.png')
    plt.show()

# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    final_scores = list()
    batches = [8, 16, 32, 64, 128, 256]
    for i in range(len(batches)):
        scores = list()
        # repeat experiment
        for r in range(repeats):
            score, history = evaluate_model(trainX, trainy, testX, testy, batches[i])
            score = score * 100.0
            print('>#%d: %.3f' % (r+1, score))
            scores.append(score)
        # summarize results
        final_scores.append(scores)
    summarize_results(final_scores, batches)
    return score, history

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 50, 0, 0.5])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

_, history = run_experiment(10)
plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()