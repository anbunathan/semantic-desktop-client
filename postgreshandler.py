import psycopg2
from config import config
from pathlib import Path

class postgressql:
    def connect(self):
        print('Entered into connect...')
        """ Connect to the PostgreSQL database server """
        conn = None
        try:
            # read connection parameters
            params = config()

            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(**params)

            # create a cursor
            cur = conn.cursor()

            # execute a statement
            print('PostgreSQL database version:')
            cur.execute('SELECT version()')

            # display the PostgreSQL database server version
            db_version = cur.fetchone()
            print(db_version)

            # close the communication with the PostgreSQL
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')

    def create_tables(self):
        """ create tables in the PostgreSQL database"""
        commands = (
            """
            CREATE TABLE master (
                file_id SERIAL PRIMARY KEY,
                directory_path VARCHAR(255) NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                file_extension VARCHAR(5) NOT NULL
            )
            """,
            """
            CREATE TABLE paragraphs (
                    paragraph_id SERIAL PRIMARY KEY,
                    file_id INTEGER NOT NULL,
                    paragraph TEXT NOT NULL,
                    automatic_tag TEXT NULL,
                    manual_tag TEXT NULL           
                    
            )
            """)
        conn = None
        try:
            # read the connection parameters
            params = config()
            # connect to the PostgreSQL server
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            # create table one by one
            for command in commands:
                cur.execute(command)
            # close communication with the PostgreSQL database server
            cur.close()
            # commit the changes
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()

    def insert_master(self, directory_path, file_name, file_extension):
        """ insert a new vendor into the vendors table """
        sql = """INSERT INTO master(directory_path, file_name, file_extension)
                 VALUES(%s, %s, %s) RETURNING file_id;"""
        conn = None
        file_id = None
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(sql, (directory_path, file_name, file_extension))
            # get the generated id back
            file_id = cur.fetchone()[0]
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return file_id

    def insert_paragraph_list(self, file_id, paragraph_list, automatic_tag=None, manual_tag=None):
        """ insert multiple vendors into the vendors table  """
        sql = """INSERT INTO paragraphs(file_id, paragraph, automatic_tag, manual_tag)
                         VALUES(%s,%s,%s,%s);"""
        conn = None
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            #cur.executemany(sql, [file_id, paragraph_list, automatic_tag, manual_tag])
            for paragraph in paragraph_list:
                cur.execute(sql, (file_id, paragraph, automatic_tag, manual_tag))
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()

    def update_paragraph(self, paragraph_id, automatic_tag=None, manual_tag=None):
        """ update vendor name based on the vendor id """
        sql = """ UPDATE paragraphs
                    SET automatic_tag = %s,
                    manual_tag = %s
                    WHERE paragraph_id = %s"""
        conn = None
        updated_rows = 0
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the UPDATE  statement
            cur.execute(sql, (automatic_tag, manual_tag, paragraph_id))
            # get the number of updated rows
            updated_rows = cur.rowcount
            # Commit the changes to the database
            conn.commit()
            # Close communication with the PostgreSQL database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return updated_rows

    def update_autotag(self, paragraph_id, automatic_tag=None):
        """ update vendor name based on the vendor id """
        sql = """ UPDATE paragraphs
                    SET automatic_tag = %s                    
                    WHERE paragraph_id = %s"""
        conn = None
        updated_rows = 0
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the UPDATE  statement
            cur.execute(sql, (automatic_tag, paragraph_id))
            # get the number of updated rows
            updated_rows = cur.rowcount
            # Commit the changes to the database
            conn.commit()
            # Close communication with the PostgreSQL database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return updated_rows

    def update_manualtag(self, paragraph_id, manual_tag=None):
        """ update vendor name based on the vendor id """
        sql = """ UPDATE paragraphs
                    SET manual_tag = %s                    
                    WHERE paragraph_id = %s"""
        conn = None
        updated_rows = 0
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the UPDATE  statement
            cur.execute(sql, (manual_tag, paragraph_id))
            # get the number of updated rows
            updated_rows = cur.rowcount
            # Commit the changes to the database
            conn.commit()
            # Close communication with the PostgreSQL database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return updated_rows

    def get_fileinfo(self, directory_path, file_name ):
        """ query data from the vendors table """
        conn = None
        matching_rows = 0
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT file_id FROM master WHERE  (directory_path = directory_path AND file_name = file_name);")
            matching_rows = cur.rowcount
            print("The number of matching rows: ", matching_rows)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return matching_rows

    def get_paragraphs(self):
        """ query parts from the parts table """
        conn = None
        paras = None
        paraids = None
        autotags = None
        manualtags = None
        dirs = []
        files = []
        fileids = []
        filepaths = []
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT paragraph FROM paragraphs ORDER BY paragraph_id")
            paras = cur.fetchall()
            print("The number of paragraphs: ", cur.rowcount)
            cur.execute("SELECT paragraph_id FROM paragraphs ORDER BY paragraph_id")
            paraids = cur.fetchall()
            print("The number of paragraph ids: ", cur.rowcount)
            cur.execute("SELECT automatic_tag FROM paragraphs ORDER BY paragraph_id")
            autotags = cur.fetchall()
            print("The number of autotags: ", cur.rowcount)
            cur.execute("SELECT manual_tag FROM paragraphs ORDER BY paragraph_id")
            manualtags = cur.fetchall()
            print("The number of manualtags: ", cur.rowcount)
            cur.execute("SELECT file_id FROM paragraphs ORDER BY paragraph_id")
            rows = cur.fetchall()
            print("The number of filepaths: ", cur.rowcount)
            for row in rows:
                cur.execute("SELECT directory_path, file_name FROM master WHERE (file_id=%s);",(row))
                info = cur.fetchone()
                filepath = Path(info[0] + "\\" + info[1])
                filepaths.append(filepath)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return paras, filepaths, paraids, autotags, manualtags

    def get_manualtag(self, paraid):
        """ query parts from the parts table """
        conn = None
        manualtag = None
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT manual_tag FROM paragraphs WHERE (paragraph_id=%s);", (str(paraid)))
            manualtag = cur.fetchone()
            print("manualtag = ", manualtag)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return manualtag

    def delete_file(self, file_id):
        """ delete part by part id """
        conn = None
        rows_deleted = 0
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the UPDATE  statement
            cur.execute("DELETE FROM master WHERE file_id = %s", (file_id,))
            # get the number of updated rows
            rows_deleted = cur.rowcount
            # Commit the changes to the database
            conn.commit()
            # Close communication with the PostgreSQL database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return rows_deleted

    def delete_paragraphs(self, file_id):
        """ delete part by part id """
        conn = None
        rows_deleted = 0
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the UPDATE  statement
            cur.execute("DELETE FROM paragraphs WHERE file_id = %s", (file_id,))
            # get the number of updated rows
            rows_deleted = cur.rowcount
            # Commit the changes to the database
            conn.commit()
            # Close communication with the PostgreSQL database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return rows_deleted