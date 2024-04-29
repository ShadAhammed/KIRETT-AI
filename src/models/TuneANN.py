# Import packages
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.models import Sequential
from sklearn.metrics import make_scorer, accuracy_score

LeakyReLU = LeakyReLU(alpha=0.1)
import warnings

score_acc = make_scorer(accuracy_score)
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)


class ANNTuner:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def build_model(self, hp):
        model = Sequential()
        model.add(tf.keras.Input(shape=self.X_train.shape[1]))
        # Layer-1
        hp_units = hp.Int('units_in_layers_1', min_value=10, max_value=30, step=1)
        model.add(Dense(units=hp_units,
                        activation=hp.Choice('Activation_1', ['relu', 'sigmoid', 'tanh'])))
        hp_dropout1 = hp.Float('Dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.Dropout(hp_dropout1))

        # Layer-2
        hp_units2 = hp.Int('units_in_layers_2', min_value=6, max_value=24, step=1)
        model.add(Dense(units=hp_units2,
                        activation=hp.Choice('Activation_2', ['relu', 'sigmoid', 'tanh'])))
        hp_dropout2 = hp.Float('Dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.Dropout(hp_dropout2))

        # Layer-3
        hp_units3 = hp.Int('units_in_layers_3', min_value=4, max_value=16, step=1)
        model.add(Dense(units=hp_units3,
                        activation=hp.Choice('Activation_3', ['relu', 'sigmoid', 'tanh'])))
        hp_dropout3 = hp.Float('Dropout_rate_3', min_value=0.0, max_value=0.5, step=0.1)
        model.add(tf.keras.layers.Dropout(hp_dropout3))

        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
            loss=tf.losses.binary_crossentropy,
            metrics=['accuracy'])
        #print(model.summary())
        return model

    def SelectTuner(self, model, TunerNr):

        if TunerNr == 1:
            Tuner = RandomSearch(model,
                                 objective='val_accuracy',
                                 max_trials=5,
                                 executions_per_trial=3,
                                 overwrite=True)

        if TunerNr == 2:
            Tuner = Hyperband(model,
                              objective='val_accuracy',
                              max_epochs=20,
                              factor=3,
                              overwrite=True)

        if TunerNr == 3:
            Tuner = BayesianOptimization(
                hypermodel=model,
                objective='val_accuracy',
                max_trials=5,
                overwrite=True)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        Tuner.search(self.X_train, self.y_train, epochs=100, validation_split=0.25, callbacks=[callback], verbose=0)
        params = Tuner.get_best_hyperparameters(num_trials=1)[0]
        print(params.values)

        return params

    # def TuneModel(self):
    #     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    #
    #     tuner.search(self.X_train, self.y_train,
    #              epochs=100,
    #              validation_split=0.25, callbacks= [callback], verbose= 0)
    #     global hps
    #     hps= tuner.get_best_hyperparameters(num_trials= 1)[0]
    #     return hps

# model= tuner.hypermodel.build(hps)
#
# history = model.fit(X_train, y_train, epochs=100, vaHyperlidation_split=0.2, verbose= 0)
# val_acc_per_epoch = history.history['val_accuracy']
#
# best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# print('Best epoch: %d' % (best_epoch))
#
# hypermodel = tuner.hypermodel.build(hps)
# hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2, verbose= 0)
#
# # Evaluate the test performance of the tuned model
# eval_result = hypermodel.evaluate(X_test, y_test)
# print("[Test loss, Test accuracy]:", eval_result)
