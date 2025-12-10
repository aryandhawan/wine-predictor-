import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import keras_tuner as kt

df = pd.read_csv("wine.csv")
x = df.drop(columns=['quality'])
y = df['quality']

x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full, random_state=42)

# convert to numeric numpy arrays with correct dtype
x_train_np = x_train.values.astype('float32')
x_valid_np = x_valid.values.astype('float32')
x_test_np = x_test.values.astype('float32')

y_train_np = y_train.values.astype('float32')
y_valid_np = y_valid.values.astype('float32')
y_test_np = y_test.values.astype('float32')

input_shape = (x_train_np.shape[1],)

norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(x_train_np)  # adapt on numpy float32

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=input_shape, dtype=tf.float32),
    norm_layer,
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

print(model.summary())

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)
model.compile(loss='mse',
              optimizer='adam',
              metrics=[tf.keras.metrics.RootMeanSquaredError()])  # instantiate metric

model.fit(x_train_np, y_train_np,
          validation_data=(x_valid_np, y_valid_np),
          epochs=100, callbacks=[early_stopping])

print(model.evaluate(x_test_np, y_test_np))
print('Done')

# we have done this via sequential API now let's do the same thing on via functional API

normalisation_layer = tf.keras.layers.Normalization()
normalisation_layer.adapt(x_train_np)
hidden_1 = tf.keras.layers.Dense(units=50, activation="relu")
hidden_2 = tf.keras.layers.Dense(units=50, activation="relu")
hidden_3 = tf.keras.layers.Dense(units=50, activation="relu")
hidden_4 = tf.keras.layers.Dense(units=1)

# flow of NN
input = tf.keras.layers.Input(shape=(11,))
norm_output = normalisation_layer(input)
hidden_1_output = hidden_1(norm_output)
hidden_2_output = hidden_2(hidden_1_output)
hidden_3_output = hidden_3(hidden_2_output)
output = tf.keras.layers.Dense(1)(hidden_3_output)

model = tf.keras.models.Model(inputs=input, outputs=output)

print(model.summary())

model.compile(optimizer="adam", loss="mse", metrics=["RootMeanSquaredError"])

model.fit(x_train_np, y_train_np, epochs=100, batch_size=32,validation_data=(x_valid_np, y_valid_np),callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

print(model.evaluate(x_test_np, y_test_np))

prediction=model.predict(x_test_np)

print(prediction)

# now let's do it via keras Tuner (hyper_tune the model)

def build_model(hp):
    # define hp

    n_hidden=hp.Int('n_hidden',min_value=1,max_value=3,default=2)
    n_neurons=hp.Int('n_neurons',min_value=32,max_value=128,step=32)
    learning_rate=hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log')
    model = tf.keras.models.Sequential()

    norm_layer=tf.keras.layers.Normalization()
    norm_layer.adapt(x_train_np)

    model.add(norm_layer)

    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(units=n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss='mse',metrics=['RootMeanSquaredError'])

    return model

#create a randomSearch tuner

tuner=kt.RandomSearch(build_model,objective='val_RootMeanSquaredError',max_trials=10,executions_per_trial=1,directory=r'C:/Users/Aryan/OneDrive/Desktop/python/DL and DL projects',project_name='hypertune',overwrite=True)

# run the search

early_stopping=tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)

print("starting hyperparameter search")

tuner.search(x_train_np, y_train_np, epochs=100, batch_size=32,validation_data=(x_valid_np, y_valid_np))

print("search completed")

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The optimal number of hidden layers is {best_hps.get('n_hidden')}.
The optimal number of neurons is {best_hps.get('n_neurons')}.
The optimal learning rate is {best_hps.get('learning_rate')}.
""")

best_model = tuner.get_best_models(num_models=1)[0] # this is used to get the best model to work with the dataset.


tuner.results_summary()

print("\nEvaluating the best model on the test set:")
loss, rmse = best_model.evaluate(x_test_np, y_test_np)
print(f"Test Loss (MSE): {loss:.4f}, Test RMSE: {rmse:.4f}")

print(best_model.predict(x_test_np))

print("Done")