# Import local
from data_preparation import modifyData, addAgeCategory, columns_to_boolean
from random_forest import build_random_forest, parse_first_tree

# Import external libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    # Get data to data frame
    df = pd.read_csv("Banking Prediction Sample 1 - Dataset.csv", index_col = 'Customer_ID')

    # # Get first 5 row
    # print(f"First 5 rows:\n {df.head()}")

    # # Describe data
    # described_data = df.describe()
    # print(f"Described data:\n {described_data}")

    # # Unique values
    # print("Unique values")
    # for column in df:
    #     unique_values = np.unique(df[column])
    #     nr_value = len(unique_values)
    #     if nr_value <10:
    #         print(f"The number of values for feature {column}: {nr_value} -- {unique_values}")
    #     else:
    #         print(f"The number of values for feature {column}: {nr_value}")

    # # Check if there are any null values
    # null_values = df.isnull().sum()
    # print(f"Number of null values per column:\n {null_values}")

    # # Get correlation matrix
    # correlation_matrix = df.corr()
    # print(f"Correlation matrix:\n {correlation_matrix}")

    # Get new data frame with modified data
    # For more info see data_manipulation.py
    df_new = modifyData(df)

    # Add age category to data frame
    df_new = addAgeCategory(df_new)
    features = [f"{column}_Flag" for column in columns_to_boolean]
    features.extend(["Gender", "Age_Category", "Year_Of_Account_Creation_Category"])

    for feature in features:
        splot = sns.countplot(x=feature, data=df_new, palette="Set3")
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

        plt.xticks(rotation=45)
        plt.show()

    # sns.pairplot(df_new[['Number_Of_Used_Services','Age']])

    # Create new data frame without New_Credit_Card_Flag column
    df_x = df_new.drop(["New_Credit_Card_Flag", "Age_Category", "Year_Of_Account_Creation_Category"], axis=1)

    # Create new data frame with only New_Credit_Card_Flag column
    df_y = df_new["New_Credit_Card_Flag"]

    # Split data into random train and test subsets (Cross validation)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.2)

    # try:
    #     # Load model if already exists
    #      from tensorflow.keras.models import load_model
    #     neural_network_model = load_model("testmodel")
    # except OSError:
    #     # Train neural netowrk on train data
    #     # For more info see neural_network.py
    #     from neural_network import build_neural_network
    #     neural_network_model = build_neural_network(x_train, y_train)

    # # Predict using trained model and test data
    # y_hat = neural_network_model.predict(x_test)

    # # Modify predicted values to either 0 or 1 depending if < 0.5 or not
    # y_hat = [0 if val < 0.5 else 1 for val in y_hat]

    # # Calculate accuracy
    # neural_network_accuracy = accuracy_score(y_test, y_hat)
    # print(f"Accuracy score of neural network is {neural_network_accuracy}")

    random_forest_model = build_random_forest(x_train, y_train)

    y_hat = random_forest_model.predict(x_test)

    random_forest_acurracy = accuracy_score(y_test, y_hat)
    print(f"Accuracy score random forest is {random_forest_acurracy}")

    parse_first_tree(random_forest_model, x_train.columns)

    # df_x = df_x[:30]
    # df_y = df_y[:30]

    # cluster_model = KMeans(n_clusters=2)
    # cluster_model.fit(df_x)
    
    # color_scheme = np.array(["lightsalmon", "powderblue"])

    # plt.subplot(1,2,1)

    # plt.scatter(x=df_x.Tenure, y=df_x.Age, c=color_scheme[df_y],s=50)

    # plt.title('Ground Truth Classification')

    # plt.subplot(1,2,2)

    # plt.scatter(x=df_x.Tenure, y=df_x.Age, c=color_scheme[cluster_model.labels_],s=50)

    # plt.title('K-Mean Classification')

    # plt.show()

    # plt.savefig("KMeans.png")

    # print(classification_report(df_y, cluster_model.labels_))

if __name__ == '__main__':
    main()