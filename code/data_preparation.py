import numpy as np
import os
import pandas as pd


class Data:
    ''' Pulls data from all CSVs in 'data_dir'. Merges, cleans, and extracts
    features from the ata. Locally, you can use '../data/'.

    Args:
        data_dir (str): Whichever directory holds your CSVs.
                        Locally, you can use'../data/'.
    '''
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = pd.DataFrame()

    def get_data(self):
        '''Import all CSVs from 'data_dir' and creates instance variables.

        Associated notes may be used for testing the data to ensure
        reprooducability. This is a great library for codifying assumptions in
        data:
        https://github.com/great-expectations/great_expectations
        '''
        # Countries CSV has 226 countries with 0 null values.
        self.countries = pd.read_csv(os.path.join(self.data_dir,
                                                  "countries.csv"))
        # Currencies CSV has 184 types of currencies.
        self.currencies = pd.read_csv(os.path.join(self.data_dir,
                                                   "currency_details.csv"))
        # In known_fraudsters, there are 9646 non-fraudsters and 298 fraudsters.
        self.known_fraudsters = pd.read_csv(os.path.join(self.data_dir,
                                                         "fraudsters.csv"))
        # In the transactions CSV, that are 688651 transactions.
        self.transactions = pd.read_csv(os.path.join(self.data_dir,
                                                     "transactions.csv"))
        # In all_users, there are 273 users that are not in the users table.
        self.all_users = pd.read_csv(os.path.join(self.data_dir, "users.csv"))

    def clean_data(self, returns=True):
        '''Clean and merge the data to be used for training.

        Args:
            returns (bool): Whether or not to return clean DataFrame
        '''
        # Change feature names to more be more interpretable
        self.all_users = self.all_users.rename(columns={"ID": 'USER_ID'})
        self.data = pd.merge(self.transactions,
                             self.all_users,
                             how='left',
                             on='USER_ID')
        self.data = self.data.rename(columns={"STATE_x": "TRANSACTION_STATE",
                                    "STATE_y": "USER_STATE",
                                    "CREATED_DATE_x": "TRANSACTION_DATE",
                                    "CREATED_DATE_y": "ACCOUNT_CREATED_DATE"})

        # Put date features in pandas datetime format
        cols = ['TRANSACTION_DATE', 'ACCOUNT_CREATED_DATE']
        for col in cols:
            self.data[col] = pd.to_datetime(self.data[col])

        # Change ENTRY_TYPE to NaN if not using a card
        no_card_mask = ~((self.data['TYPE']=='CARD_PAYMENT') \
                          | (self.data['TYPE']=='ATM'))

        self.data['ENTRY_METHOD'].copy()[no_card_mask] = np.nan

        # Drop AMOUNT for multicollinearity, ID because it is unique for each
        # transaction, and USER_STATE for leakage.
        self.data = self.data.drop(['ID', 'AMOUNT', 'USER_STATE',
                                    'MERCHANT_CATEGORY'], axis=1)

        if returns:
            return self.data

    def get_df(self):
        '''DataFrame getter method.'''
        return self.data
