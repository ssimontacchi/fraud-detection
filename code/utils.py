import numpy as np
import os
import pandas as pd


def save_csv(df, path, filename):
    """Save Pandas DataFrame to CSV."""
    dest = os.path.join(path, filename) + '.csv'
    df.to_csv(dest, index=False)
    print(f"DataFrame saved to {dest}.")


def get_data(table_name, engine):
    """
    Converts PostgreSQL to Pandas DataFrame
    Args:
       table (str): PostgreSQL table name
       engine (object): Sqlalchemy create_engine object
    Returns:
        Pandas DataFrame
    """
    return pd.read_sql_query(f'SELECT * FROM {table_name}', con=engine)
