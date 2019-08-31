import pandas as pd
from pathlib import Path


def clean(data_dir):
    '''Import all CSVs from 'data_dir'. Associated notes may be used for
       testing the data to ensure reprooducability. This is a great library for
       codifying assumptions in data.
       https://github.com/great-expectations/great_expectations'''

    # Countries CSV has 226 countries and associated codes with 0 null values.
    countries = pd.read_csv(Path(data_dir) + "/countries.csv
    # Currencies CSV has 184 types of currencies
    currencies = pd.read_csv(Path(data_dir) + "/currency_details.csv")
    # In known_fraudsters, there are 9646 non-fraudsters and 298 fraudsters
    known_fraudsters = pd.read_csv(Path(data_dir) + "/fraudsters.csv")
    # In transactions, there are 273 users that are not in the users table.
    transactions = pd.read_csv(Path(data_dir) + "/transactions.csv")


    all_users = pd.read_csv(Path(data_dir) + "/users.csv")


    # Change feature names to more be more interpretable
    fraudster_info = all_users
    fraudster_info = fraudster_info.rename(columns={"ID": 'USER_ID'})
    result = pd.merge(transactions, fraudster_info, how='left', on='USER_ID')
    result = result.rename(columns={"STATE_x": "TRANSACTION_STATE",
                                    "STATE_y": "USER_STATE",
                                    "CREATED_DATE_x": "TRANSACTION_DATE",
                                    "CREATED_DATE_y": "ACCOUNT_CREATED_DATE"})

    # Put date feature in pandas datetime format
    result['TRANSACTION_DATE'] = pd.to_datetime(result['TRANSACTION_DATE'])
    result['ACCOUNT_CREATED_DATE'] = pd.to_datetime(result['ACCOUNT_CREATED_DATE'])

    # Change ENTRY_TYPE to NaN if not using a card
    no_card_mask = ~((result['TYPE']=='CARD_PAYMENT') | (result['TYPE']=='ATM'))
    result['ENTRY_METHOD'][no_card_mask] = np.nan
    



if __name__ == "__main__":
