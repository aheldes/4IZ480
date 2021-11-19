import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


def main():
    df = pd.read_csv("Banking Prediction Sample 1 - Dataset.csv", index_col = 'Customer_ID')
    # print(df.describe())
    # print(df.isnull().values.any())
    # print(df.corr())
    # print(df.head())
    df_new = df

    columns_to_boolean = ['Saving_Amount', 'Current_Amount', 'Time_Deposits_Amount', 'Funds_Amount',
                            'Stocks_Amount', 'Bank_Assurance_Amount', 'Life_Assurance_Amount', 'Business_Loan_Amount', 
                            'Home_Loan_Amount', 'Consumer_Loan_Amount', 'Branch_Transactions', 'ATM_Transactions',
                            'Phone_Transactions', 'Internet_Transactions', 'Standing_Orders']

    for column in columns_to_boolean:
        df_new.loc[df_new[column] > 0, column] = 1
        # df_new[column] = df_new[column].astype('bool')
        df_new = df_new.rename(columns={column: f"{column}_Flag"})

    df_new['Gender'] = df_new['Gender'].map({'F': 1, 'M': 0})
    # df_new['Gender'] = df_new['Gender'].astype('bool')

    df_new['Number_Of_Used_Services'] = df_new[[f"{column}_Flag" for column in columns_to_boolean if f"{column}_Flag".endswith("Amount_Flag") == True]].sum(axis=1)

    df_x = df_new.drop(["New_Credit_Card_Flag"], axis=1)
    df_y = df_new["New_Credit_Card_Flag"]

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(df_x, df_y, test_size=.2)

    model = Sequential()
    model.add(Dense(units=32, activation="relu", input_dim=len(X_Train.columns)))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="sgd", metrics="accuracy")

    model.fit(X_Train, Y_Train, epochs=20, batch_size=32)

    Y_Hat = model.predict(X_Test)
    Y_Hat = [0 if val < 0.5 else 1 for val in Y_Hat]

    print(f"Accuract score is {accuracy_score(Y_Test, Y_Hat)}")

    model.save("testmodel")

if __name__ == '__main__':
    main()