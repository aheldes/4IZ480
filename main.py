# Import local
from data_manipulation import modifyData
from neural_network import build_neural_network
from random_forest import build_random_forest, parse_first_tree
import os

# Import external libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


def main():
    # Get data to data frame
    df = pd.read_csv("Banking Prediction Sample 1 - Dataset.csv", index_col = 'Customer_ID')

    # Get first 5 row
    print(f"First 5 rows:\n {df.head()}")

    # Describe data
    described_data = df.describe()
    print(f"Described data:\n {described_data}")

    # Check if there are any null values
    null_values = df.isnull().values.any()
    print(f"Are there any null values: {null_values}")

    # Get correlation matrix
    correlation_matrix = df.corr()
    print(f"Correlation matrix:\n {correlation_matrix}")

    # Get new data frame with modified data
    # For more info see data_manipulation.py
    df_new = modifyData(df)

    # Create new data frame without New_Credit_Card_Flag column
    df_x = df_new.drop(["New_Credit_Card_Flag"], axis=1)

    # Create new data frame with only New_Credit_Card_Flag column
    df_y = df_new["New_Credit_Card_Flag"]

    # Split data into random train and test subsets (Cross validation)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.2)

    try:
        # Load model if already exists
        neural_network_model = load_model("testmodel")
    except OSError:
        # Train neural netowrk on train data
        # For more info see neural_network.py
        neural_network_model = build_neural_network(x_train, y_train)

    # Predict using trained model and test data
    y_hat = neural_network_model.predict(x_test)

    # Modify predicted values to either 0 or 1 depending if < 0.5 or not
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]

    # Calculate accuracy
    neural_network_accuracy = accuracy_score(y_test, y_hat)
    print(f"Accuracy score of neural network is {neural_network_accuracy}")

    random_forest_model = build_random_forest(x_train, y_train)

    y_hat = random_forest_model.predict(x_test)

    random_forest_acurracy = accuracy_score(y_test, y_hat)
    print(f"Accuracy score random forest is {random_forest_acurracy}")

    parse_first_tree(random_forest_model, x_train.columns)

if __name__ == '__main__':
    main()