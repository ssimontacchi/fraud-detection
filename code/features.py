from config import connection, data_path, engine, schema_path
from data_preparation import Data
from etl import ETL
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import utils


class Feature_Extractor:
    ''' Used to extract features from clean DataFrame. Has two optional
    parameters if you would like to run extract_features() with initialization.

    Args:
        df (pd.DataFrame): DataFrame with cleaned data.
        run (bool): If you would like to run extract_features in one call.
    '''
    def __init__(self, df=pd.DataFrame(), run=False):
        if run == True:
            return self.extract_features(df)

    def extract_features(self, df, verbose=False):
        '''Extracts features from DataFrame. These include:
        - Converting 'TRANSACTION_DATE' and 'ACCOUNT_CREATED_DATE' to features
        the model can use.
        - Running totals of 'CURRENCIES_USED' and 'MERCH_COUNTRIES_USED'.
        - Time since user's last transaction, 'TIME_SINCE_LAST_TRANSACTION'.
        - Running count of a user's total number of transactions, 'COUNT'.
        - 'ROLLING_AVG' and 'ROLLING_SUM' of 'AMOUNT_USD'.

        Args:
            data (pd.DataFrame): DataFrame with cleaned data.
            verbose (bool): If you would like status print statements (optional).
        '''
        self.verbose = verbose

        # Sort df to calculate running_totals, etc.
        df = df.sort_values(['USER_ID', 'TRANSACTION_DATE'])

        # Calculates number of currencies and countries from which a user has
        # made a transaction
        df = self.running_total_unique(df, 'CURRENCY','CURRENCIES_USED')
        df = self.running_total_unique(df, 'MERCHANT_COUNTRY',
                                       'MERCH_COUNTRIES_USED')

        # Calculate time since last transaction by user
        df = self.time_since_last_transaction(df)

        # Count of how many transactions the user has made
        df = self.running_transaction_count(df, 'COUNT')

        # Calculate rolling average and rolling sum by user
        df = self.rolling_avg_or_sum(df, 'ROLLING_AVG', avg_or_sum='avg')
        df = self.rolling_avg_or_sum(df, 'ROLLING_SUM', avg_or_sum='sum')

        # Transform date features into num. of days since min. value of feature
        df = self.transform_date(df, 'ACCOUNT_CREATED_DATE')
        df = self.transform_date(df, 'TRANSACTION_DATE')

        # Use median filling for NaN values
        df = self.fill_nans(df)

        return df


    def running_total_unique(self, df, category, new_feature_name):
        ''' Takes a Pandas DataFrame, sorted by 'USER_ID' and 'TRANSACTION_DATE',
        and calculates the total number of a unique category, grouped by 'USER_ID',
        at any given point.

        Args:
            df (pd.DataFrame): Pandas DataFrame in which to merge new feature.
            category (str): Feature in df from which to make running total.
            new_feature_name (str): Name of new feature.

        Returns:
            pd.DataFrame: parameter 'df' merged with new column.
        '''
        if self.verbose: print(f'Creating {new_feature_name}')
        user_group = df.groupby(['USER_ID', category], sort=False)
        count = user_group[category].nunique().groupby(level=0).cumsum()
        count = count.reset_index(name=new_feature_name)
        df = df.merge(count, on=['USER_ID', category], how='left')
        df[category] = df[category].fillna(0)

        return df

    def rolling_avg_or_sum(self, df, new_feature_name, avg_or_sum='avg'):
        ''' Takes a Pandas DataFrame, sorted by 'USER_ID' and 'TRANSACTION_DATE',
        and calculates the rolling avg or sum of 'AMOUNT_USD' feature for a user
        at any given point.

        Args:
            df (pd.DataFrame): DataFrame in which to merge new feature.
            category (str): Feature in df from which to make running total.
            new_feature_name (str): Name of new feature.
            avg_or_sum (str): Determines whether to

        Returns:
            pd.DataFrame: parameter 'df' merged with new column.
        '''
        if self.verbose: print(f'Creating {new_feature_name}')
        if avg_or_sum == 'avg':
            grouped_amount = df.groupby('USER_ID')['AMOUNT_USD'].expanding().mean()
        elif avg_or_sum == 'sum':
            grouped_amount = df.groupby('USER_ID')['AMOUNT_USD'].expanding().sum()
        else:
            raise ValueError("avg_or_sum parameter must equal 'avg' or 'sum'.")

        rounded = round(grouped_amount, 2).reset_index()['AMOUNT_USD']
        feature = rounded.rename(columns={'AMOUNT_USD': new_feature_name})
        df = pd.concat([df, feature], axis=1).rename(columns={0: new_feature_name})

        return df

    def transform_date(self, df, feature):
        ''' Takes a Pandas DataFrame and pd.datetime feature and, for each value,
        calculates the the number of days since the min. value.

        Args:
            df (pd.DataFrame): DataFrame that holds the feature.
            feature (str): Feature in df.

        Returns:
            pd.Series: parameter 'df' with updated feature.
        '''
        if self.verbose: print(f'Creating {feature}')
        df[feature] = (df[feature] - df[feature].min()) / np.timedelta64(1,'D')
        return df

    def time_since_last_transaction(self, df):
        ''' Takes a Pandas DataFrame, sorted by 'USER_ID' and 'TRANSACTION_DATE',
        and pd.datetime feature and, for each row, calculates the time since the
        last transaction.

        Args:
            df (pd.DataFrame): DataFrame that holds the 'TRANSACTION_DATE' feature.

        Returns:
            pd.Series: parameter 'df' with new feature.
        '''
        if self.verbose: print(f'Creating TIME_SINCE_LAST_TRANSACTION')
        user_and_date = df[['USER_ID', 'TRANSACTION_DATE']]
        user_groups = user_and_date.groupby('USER_ID')
        time_since = user_groups['TRANSACTION_DATE'].diff()
        filled_nas = time_since.fillna(pd.Timedelta(seconds=0))
        df['TIME_SINCE_LAST_TRANSACTION'] = filled_nas.astype('timedelta64[m]')

        return df

    def running_transaction_count(self, df, new_feature_name):
        ''' Takes a Pandas DataFrame, sorted by 'USER_ID' and 'TRANSACTION_DATE',
        and pd.datetime feature and, for each row, calculates the time since the
        last transaction.

        Args:
            df (pd.DataFrame): DataFrame that holds 'TRANSACTION_DATE' feature.

        Returns:
            pd.Series: parameter 'df' with new feature.
        '''
        if self.verbose: print(f'Creating {new_feature_name}')
        new_user_marker = (df['USER_ID'] != df['USER_ID'].shift(1))
        group_new_user = df.groupby(new_user_marker.cumsum())
        df[new_feature_name] = group_new_user.cumcount() + 1

        return df

    def fill_nans(self, df):
        ''' Gets categorical variables for a DataFrame and fills in median
        for NaNs for non-categorical columns.

        Args:
            df (object): Pandas DataFrame

        Returns:
            df (object): Pandas DataFrame with median filled NaNs
        '''
        categoricals = []
        for col in list(df.columns):
            if df[col].dtype.name in ['object', 'category']:
                categoricals.append(col)

        for col in list(df.columns):
            if col not in categoricals:
                df[col] = df[col].fillna(df[col].median())
        return df


if __name__ == "__main__":
    path = '../data/'
    d = Data(path)
    d.get_data()
    df = d.clean_data()
    fe = Feature_Extractor()
    training_data = fe.extract_features(df, verbose=True)
    utils.save_csv(training_data, path, 'training_data')
    etl = ETL(connection, data_path, schema_path, engine,
              df_to_write=training_data, table_name="training_data",
              remove=False, create=False, load=False, verbose=True)
    etl.pipeline()
