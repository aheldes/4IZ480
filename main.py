# Import local
from numpy.core.fromnumeric import argmax
from data_preparation import modifyData, changeToDummies, columns_to_boolean
from random_forest import build_random_forest, parse_first_tree

# Import external libraries
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

    # Get correlation matrix
    correlation_matrix = df.corr()
    print(f"Correlation matrix:\n {correlation_matrix}")

    # Get new data frame with modified data
    # For more info see data_manipulation.py
    df_new = modifyData(df)


    # Add age category to data frame
    features = [f"{column}_Flag" for column in columns_to_boolean]
    features.extend(["Gender", "Age_Category", "Year_Of_Account_Creation_Category"])

    # for feature in features:
    #     splot = sns.countplot(x=feature, data=df_new, palette="Set3")
    #     for p in splot.patches:
    #         splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

    #     plt.xticks(rotation=45)
    #     plt.show()

    # sns.pairplot(df_new[['Number_Of_Used_Services','Age']])

    df_new = changeToDummies(df_new)

    # Create new data frame without New_Credit_Card_Flag column
    df_x = df_new.drop(["New_Credit_Card_Flag"], axis=1)

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

    # random_forest_model = build_random_forest(x_train, y_train)

    # y_hat = random_forest_model.predict(x_test)

    # random_forest_acurracy = accuracy_score(y_test, y_hat)
    # print(f"Accuracy score random forest is {random_forest_acurracy}")

    # parse_first_tree(random_forest_model, x_train.columns)

    number_of_clusters = range(2,20)
    intertia = list()
    kmeans_data = df_x.values

    # for number in number_of_clusters:
    #     cluster_test_model = KMeans(n_clusters=number, random_state=4)
    #     cluster_test_model.fit(kmeans_data)
    #     models_inertia = cluster_test_model.inertia_
    #     intertia.append(models_inertia)
    #     print(f"The inertia for: {number} clusters is {models_inertia}")

    # fig, (ax1) = plt.subplots(1, figsize=(16,6))
    # xx = np.arange(len(number_of_clusters))
    # ax1.plot(xx, intertia)
    # ax1.set_xticks(xx)
    # ax1.set_xticklabels(number_of_clusters, rotation="vertical")
    # plt.xlabel("Number of clusters")
    # plt.ylabel("Inertia score")
    # plt.title("Inertia Plot per k")
    # plt.show()

    
    cluster_model = KMeans(n_clusters=3, random_state=2)
    cluster_model.fit(kmeans_data)
    
    print(f"The clusters are: {cluster_model.labels_}")

    print(f"The inertia is: {cluster_model.inertia_}")

    predictions = cluster_model.predict(kmeans_data)

    unique, counts = np.unique(predictions, return_counts=True)
    counts = counts.reshape(1,3)

    cluster_names = ["Cluster 0","Cluster 1","Cluster 2"]

    countscldf = pd.DataFrame(counts, columns = cluster_names)
    print(countscldf)  

    X = kmeans_data
    y_num = predictions

    pca = PCA(n_components=2, random_state = 453)
    X_r = pca.fit(X).transform(X)

    print(f'Explained variance ratio (first two components): {str(pca.explained_variance_ratio_)}')

    plt.figure()
    plt.figure(figsize=(12,8))
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2


    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4], cluster_names):
        plt.scatter(X_r[y_num == i, 0], X_r[y_num == i, 1], color=color, alpha=.8, lw=lw,label=target_name)
        
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.6)   
    plt.title('PCA of 2 Items')
    plt.show()

    n_components = X.shape[1]
    print(n_components)

    # Running PCA with all components
    pca = PCA(n_components=n_components, random_state = 453)
    X_r = pca.fit(X).transform(X)


    # Calculating the 95% Variance
    total_variance = sum(pca.explained_variance_)
    print("Total Variance in our dataset is: ", total_variance)
    var_95 = total_variance * 0.95
    print("The 95% variance we want to have is: ", var_95)
    print("")

    # Creating a df with the components and explained variance
    a = zip(range(0,n_components), pca.explained_variance_)
    a = pd.DataFrame(a, columns=["PCA Comp", "Explained Variance"])

    # Trying to hit 95%
    print("Variance explain with 1 n_compononets: ", sum(a["Explained Variance"][0:1]))
    print("Variance explain with 2 n_compononets: ", sum(a["Explained Variance"][0:2]))
    print("Variance explain with 3 n_compononets: ", sum(a["Explained Variance"][0:3]))
    print("Variance explain with 4 n_compononets: ", sum(a["Explained Variance"][0:4]))
    print("Variance explain with 5 n_compononets: ", sum(a["Explained Variance"][0:5]))
    print("Variance explain with 10 n_compononets: ", sum(a["Explained Variance"][0:10]))
    print("Variance explain with 15 n_compononets: ", sum(a["Explained Variance"][0:15]))
    print("Variance explain with 20 n_compononets: ", sum(a["Explained Variance"][0:20]))
    print("Variance explain with 25 n_compononets: ", sum(a["Explained Variance"][0:25]))
    print("Variance explain with 30 n_compononets: ", sum(a["Explained Variance"][0:30]))
    print("Variance explain with 32 n_compononets: ", sum(a["Explained Variance"][0:32]))

    # Plotting the Data
    plt.figure(1, figsize=(14, 8))
    plt.plot(pca.explained_variance_ratio_, linewidth=2, c="r")
    plt.xlabel('n_components')
    plt.ylabel('explained_ratio_')

    # Plotting line with 95% e.v.
    plt.axvline(53,linestyle=':', label='n_components - 95% explained', c ="blue")
    plt.legend(prop=dict(size=12))

    # adding arrow
    plt.annotate('2 eigenvectors used to explain 95% variance', xy=(2, pca.explained_variance_ratio_[2]), 
                xytext=(58, pca.explained_variance_ratio_[10]),
                arrowprops=dict(facecolor='blue', shrink=0.05))

    plt.show()

    pca = PCA(n_components=2, random_state = 453)
    X_r = pca.fit(X).transform(X)

    inertia = list()
    inertia_difference = dict()

    #running Kmeans

    for f in number_of_clusters:
        kmeans = KMeans(n_clusters=f, random_state=2)
        kmeans = kmeans.fit(X_r)
        u = kmeans.inertia_
        inertia.append(u)
        print("The innertia for :", f, "Clusters is:", u)
        if len(inertia) >= 2:
            difference = inertia[-2]-u
            print(f"Difference in inertia between clusters: {difference}")
            inertia_difference[f] = difference

    print(inertia_difference)
    # Creating the scree plot for Intertia - elbow method
    fig, (ax1) = plt.subplots(1, figsize=(16,6))
    xx = np.arange(len(number_of_clusters))
    ax1.plot(xx, inertia)
    ax1.set_xticks(xx)
    ax1.set_xticklabels(number_of_clusters, rotation='vertical')
    plt.xlabel('n_components Value')
    plt.ylabel('Inertia Score')
    plt.title("Inertia Plot per k")
    plt.show()

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