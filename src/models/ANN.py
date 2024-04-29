# Compilation of ANN model with tensorflow keras
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization


class ANN:
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

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=tf.losses.binary_crossentropy,
            metrics=['accuracy'])
        # print(model.summary())
        return model

    def TuneModel(self, model, TunerNr):

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
        Tuner.search(self.X_train, self.y_train, epochs=80, validation_split=0.25, callbacks=[callback], verbose=0)
        params = Tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f'\nThe best parameters found for ANN model are: {params.values}\n')
        model = Tuner.hypermodel.build(params)
        history = model.fit(self.X_train, self.y_train, epochs=80, validation_split=0.2, verbose=0)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d\n' % (best_epoch,))
        BestModel = Tuner.hypermodel.build(params)
        return best_epoch, BestModel

    def ANN_model(self, Tuner):
        m = ANN(self.X_train, self.y_train)
        model = m.build_model
        epoch, model = m.TuneModel(model, Tuner)
        model.fit(self.X_train, self.y_train, epochs=epoch, validation_split=0.2, verbose=0)
        return model
