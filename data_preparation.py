import numpy as np
import pandas as pd

columns_to_boolean = ['Saving_Amount', 'Current_Amount', 'Time_Deposits_Amount', 'Funds_Amount',
                            'Stocks_Amount', 'Bank_Assurance_Amount', 'Life_Assurance_Amount', 'Business_Loan_Amount', 
                            'Home_Loan_Amount', 'Consumer_Loan_Amount', 'Branch_Transactions', 'ATM_Transactions',
                            'Phone_Transactions', 'Internet_Transactions', 'Standing_Orders']

columns = ["Age", "Year_Of_Account_Creation"]

def modifyData(df):

    for column in columns_to_boolean:
        df.loc[df[column] > 0, column] = 1
        # df[column] = df[column].astype('bool')
        df = df.rename(columns={column: f"{column}_Flag"})

    df['Gender'] = df['Gender'].map({'F': 1, 'M': 0})
    df['Year_Of_Account_Creation'] = df["Age"] - df["Tenure"] // 12

    df['Number_Of_Used_Services'] = df[[f"{column}_Flag" for column in columns_to_boolean if f"{column}_Flag".endswith("Amount_Flag") == True]].sum(axis=1)

    for name in columns:
        conditions = [df[name] <= 18, df[name] <= 27, df[name] <= 33, df[name] <= 43, df[name] <= 56]
        choices = ["18-22","22-27","27-33","33-43","43-56"]
        df[f"{name}_Category"] = np.select(conditions, choices, default="65+")

    return df

def changeToDummies(df):  
    df = pd.get_dummies(df, columns=[f"{name}_Category" for name in columns])
        
    return df