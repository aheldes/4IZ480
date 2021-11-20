

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_neural_network(x_train, y_train):
    model = Sequential()

    # Add layers
    # Add a layer with 32 neurons with number of dimensions same as the number of columns
    model.add(Dense(units=32, activation="relu", input_dim=len(x_train.columns)))

    # Add a layer with 64 neurons
    model.add(Dense(units=64, activation="relu"))

    # Add a layer with 1 output so one neuron (either 0 or 1)
    model.add(Dense(units=1, activation="sigmoid"))

    # Compile the model 
    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics="accuracy")

    # Fit model with train data
    model.fit(x_train, y_train, epochs=20, batch_size=32)

    model.save("testmodel")

    return model