def modifyData(df):
    columns_to_boolean = ['Saving_Amount', 'Current_Amount', 'Time_Deposits_Amount', 'Funds_Amount',
                            'Stocks_Amount', 'Bank_Assurance_Amount', 'Life_Assurance_Amount', 'Business_Loan_Amount', 
                            'Home_Loan_Amount', 'Consumer_Loan_Amount', 'Branch_Transactions', 'ATM_Transactions',
                            'Phone_Transactions', 'Internet_Transactions', 'Standing_Orders']

    for column in columns_to_boolean:
        df.loc[df[column] > 0, column] = 1
        # df[column] = df[column].astype('bool')
        df = df.rename(columns={column: f"{column}_Flag"})

    df['Gender'] = df['Gender'].map({'F': 1, 'M': 0})
    # df['Gender'] = df['Gender'].astype('bool')

    df['Number_Of_Used_Services'] = df[[f"{column}_Flag" for column in columns_to_boolean if f"{column}_Flag".endswith("Amount_Flag") == True]].sum(axis=1)

    return df