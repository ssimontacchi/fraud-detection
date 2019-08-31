from config import connection, data_path, engine, schema_path
import pandas as pd
from pathlib import Path
import psycopg2 as pg
import utils
import yaml


class ETL:
    '''ETL pipeline that, by default, takes in a connection, data_path,
    and schema_path set in self.config.py, which can of course be changed when
    initializing the pipeline.

    This pipeline can also write DataFrames to PostgreSQL.

    Args:
        connection (object): psycopg2 connection set in config.py
        data_path (str): a directory where to find and write CSVs
        schema_path (str): yaml file store in config.py
        engine (object): sqlalchemy connection in config.py
        df_to_write (object): Pandas DataFrame to write to PostgreSQL
        table_name (str): name of table to write to PostgreSQL
        remove (bool): option to remove existing tables in connect_to_db
        create (bool): option to create tables in connect_to_db
        load (bool): option to load tables in connect_to_db
        display (bool): option to display tables in connect_to_db
        verbose (bool): option for status print statements

    Returns:
        None
    '''
    def __init__(self, connection, data_path, schema_path, engine='',
                 df_to_write=pd.DataFrame(), table_name='', remove=True,
                 create=True, load=True, display=True, verbose=True):
        self.conn = connection
        self.cur = self.conn.cursor()
        self.create = create
        self.data_path = data_path
        self.df_to_write = df_to_write
        self.display = display
        self.engine = engine
        self.load = load
        self.remove = remove
        self.schema_path = schema_path
        self.table_name = table_name
        self.verbose = verbose

        # Opens schema
        try:
            with open(self.schema_path) as schema_file:
                self.config = yaml.load(schema_file, Loader=yaml.FullLoader)
        except (Exception, FileNotFoundError) as error:
            print(error)

    def remove_tables(self):
        """Removes tables from the PostgreSQL database if they already exist."""
        self.cur = self.conn.cursor()
        for table in self.config:
            name = table.get('name')
            ddl = f"""DROP TABLE IF EXISTS {name}"""
            self.cur.execute(ddl)
        self.cur.close()
        self.conn.commit()
        if self.verbose:
            print("Tables removed.")

    def create_tables(self):
        """Creates tables using yaml file in the directory '../misc' and its
        corresponding CSVs.
        """
        self.cur = self.conn.cursor()
        for table in self.config:
            name = table.get('name')
            schema = table.get('schema')
            ddl = f"""CREATE TABLE IF NOT EXISTS {name} ({schema})"""
            self.cur.execute(ddl)
        self.cur.close()
        self.conn.commit()
        if self.verbose:
            print("Tables created.")

    def load_tables(self):
        """Loads a CSV into the PostgreSQL database."""
        def load_helper(path, tbl_name):
            table_source = Path(path).joinpath(f"{tbl_name}.csv")
            with open(table_source, 'r') as f:
                next(f)
                self.cur.copy_expert(f"COPY {tbl_name} FROM STDIN CSV NULL AS ''", f)
            self.conn.commit()

        self.cur = self.conn.cursor()
        for table in self.config:
            table_name = table.get('name')
            load_helper(self.data_path, table_name)
        if self.verbose:
            print("Tables have been loaded into database.")


    def display_table(self, table_name = '', table_num = ''):
        """Method used for validation and unit testing for SQL tables.

        Args:
            table_name (str): optional way to access a table for diplaying
            data_path (str): optional way to access a table by number

        Returns:
            result (str): the result of the SQL query
        """
        # If only table_num, pick the corresponding table from self.config
        if not table_name and table_num:
            table_name = self.config[table_num].get('name')
        # If neither parameter, pick the first table
        elif not table_name and not table_num:
            table_name = self.config[0].get('name')
        self.cur = self.conn.cursor()
        # Otherwise, use the supplied table_name
        self.cur.execute(f"SELECT * FROM {table_name};")
        result = self.cur.fetchone()
        return result

    def display_all_tables(self):
        """Display first row for each table written from CSVs."""
        num_tables = len(self.config)
        print(f"There are {num_tables} tables written from CSVs in the DB with " \
              "the following first entries:")
        for i in range(num_tables):
            print()
            print(self.display_table(table_num=i))

    def df_to_postgres(self, df, name=''):
        """Creates a new table and write it to our PostgreSQL database."""
        print(f'Writing {name} DataFrame to PostgreSQL...')
        df.to_sql(name, con=self.engine, if_exists='replace', index=False)
        self.display_table(table_name=name)
        print("Successful write to DB.")

    def pipeline(self):
        """ETL pipeline that removes existing tables, creates new tables, loads
        data into those tables, creates and loads tables from dataframes, and
        displays them for validation and unit-testing.

        This pipeline also catches errors and always closes the connection.
        """
        try:
            if self.remove: self.remove_tables()
            if self.create: self.create_tables()
            if self.load: self.load_tables()
            if len(self.df_to_write) > 0 and self.engine:
                self.df_to_postgres(self.df_to_write, self.table_name)
            if self.display: self.display_all_tables()

        except (Exception, pg.DatabaseError) as error:
            print(error)
        finally:
            if self.conn is not None:
                self.conn.close()
                print('Database connection closed.')


if __name__ == "__main__":
    etl = ETL(connection, data_path, schema_path)
    etl.pipeline()
