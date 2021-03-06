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
                directory_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_extension VARCHAR(5) NOT NULL,
                dir_id INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE directory (
                directory_id SERIAL PRIMARY KEY,
                directory_path TEXT NOT NULL,
                input_type TEXT NOT NULL                
            )
            """,
            """
            CREATE TABLE results (
                result_id SERIAL PRIMARY KEY,
                para TEXT NOT NULL,
                location TEXT NOT NULL,
                autotag TEXT NOT NULL,
                manualtag TEXT NULL,
                distance TEXT NOT NULL,
                rank TEXT NOT NULL,
                searchstring TEXT NOT NULL
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

    def get_directory_info(self, directory_path):
        """ query data from the vendors table """
        conn = None
        matching_rows = 0
        status = 'success'
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT directory_id FROM directory WHERE (directory_path=%s);",(directory_path,))
            matching_rows = cur.rowcount
            print("The number of matching rows: ", matching_rows)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error in get_directory_info = ",error)
            status = error
        finally:
            if conn is not None:
                conn.close()
        return matching_rows, status

    def delete_directory(self, directoryid):
        """ delete part by part id """
        conn = None
        rows_deleted = 0
        status = 'success'
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the UPDATE  statement
            cur.execute("DELETE FROM directory WHERE directory_id = %s", (directoryid,))
            # get the number of updated rows
            rows_deleted = cur.rowcount
            # Commit the changes to the database
            conn.commit()
            # Close communication with the PostgreSQL database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error in delete_directory = ", error)
            status = error
        finally:
            if conn is not None:
                conn.close()
        return rows_deleted, status

    def update_directory_path(self, directoryid, directory_path, input_type):
        """ update vendor name based on the vendor id """
        sql = """ UPDATE directory
                    SET directory_path = %s,
                    input_type = %s
                    WHERE directory_id = %s"""
        conn = None
        updated_rows = 0
        status = 'success'
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the UPDATE  statement
            cur.execute(sql, (directory_path, input_type, directoryid))
            # get the number of updated rows
            updated_rows = cur.rowcount
            # Commit the changes to the database
            conn.commit()
            # Close communication with the PostgreSQL database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error in update_directory_path = ", error)
            status = error
        finally:
            if conn is not None:
                conn.close()
        return updated_rows, status

    def get_directory_path_byid(self, directoryid):
        """ query parts from the parts table """
        conn = None
        directory = None
        inputtype = None
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT directory_path, input_type FROM directory WHERE (directory_id=%s);", (str(directoryid)),)
            directory, inputtype = cur.fetchone()
            print("directory = ", directory)
            print("inputtype = ", inputtype)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error in get_directory_path_byid = ", error)
        finally:
            if conn is not None:
                conn.close()
        return directory, inputtype

    def insert_directory(self, directory_path, input_type):
        """ insert a new vendor into the vendors table """
        sql = """INSERT INTO directory(directory_path, input_type)
                 VALUES(%s, %s) RETURNING directory_id;"""
        conn = None
        directory_id = None
        status='success'
        try:
            # read database configuration
            params = config()
            # connect to the PostgreSQL database
            conn = psycopg2.connect(**params)
            # create a new cursor
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute(sql, (directory_path, input_type))
            # get the generated id back
            directory_id = cur.fetchone()[0]
            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error in insert_directory = ", error)
            status=error
        finally:
            if conn is not None:
                conn.close()
        return directory_id, status

    def get_directory_path(self):
        """ query parts from the parts table """
        conn = None
        directories = None
        inputtypes = None
        directoryids = None
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT directory_path FROM directory ORDER BY directory_id")
            directories = cur.fetchall()
            print("The number of paragraphs: ", cur.rowcount)
            cur.execute("SELECT input_type FROM directory ORDER BY directory_id")
            inputtypes = cur.fetchall()
            print("The number of input types: ", cur.rowcount)
            cur.execute("SELECT directory_id FROM directory ORDER BY directory_id")
            directoryids = cur.fetchall()
            print("The number of autotags: ", cur.rowcount)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error in get_directory_path = ", error)
        finally:
            if conn is not None:
                conn.close()
        return directories, inputtypes, directoryids

    def insert_result_list(self, para, location, distance, rank, searchstring, autotag, manualtag=None ):
        """ insert multiple vendors into the vendors table  """
        sql = """INSERT INTO results(para, location, autotag, manualtag, distance, rank, searchstring)
                         VALUES(%s,%s,%s,%s,%s,%s,%s);"""
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
            cur.execute(sql, (para, location, autotag, manualtag, distance, rank, searchstring))

            # commit the changes to the database
            conn.commit()
            # close communication with the database
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()

    def insert_master(self, directory_path, file_name, file_extension, directory_id):
        """ insert a new vendor into the vendors table """
        sql = """INSERT INTO master(directory_path, file_name, file_extension, dir_id)
                 VALUES(%s, %s, %s, %s) RETURNING file_id;"""
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
            cur.execute(sql, (directory_path, file_name, file_extension, directory_id))
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
            cur.execute("SELECT file_id FROM master WHERE  (directory_path=%s AND file_name=%s);",(directory_path,file_name))
            matching_rows = cur.rowcount
            print("The number of matching rows: ", matching_rows)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return matching_rows

    def check_fileid_exists(self, file_id):
        """ query data from the vendors table """
        conn = None
        fileid_exists = False
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            fileid_exists = cur.execute("SELECT EXISTS(SELECT 1 FROM master WHERE file_id=%s);",(file_id, ))
            matching_rows = cur.rowcount
            print("The number of matching rows: ", matching_rows)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return matching_rows

    def get_results(self):
        """ query parts from the parts table """
        conn = None
        paras = None
        locations = None
        autotags = None
        manualtags = None
        distances = None
        ranks = None
        searchstrings = None
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT para FROM results ORDER BY result_id DESC")
            paras = cur.fetchall()
            print("The number of paragraphs: ", cur.rowcount)
            cur.execute("SELECT location FROM results ORDER BY result_id DESC")
            locations = cur.fetchall()
            print("The number of locations: ", cur.rowcount)
            cur.execute("SELECT autotag FROM results ORDER BY result_id DESC")
            autotags = cur.fetchall()
            print("The number of autotags: ", cur.rowcount)
            cur.execute("SELECT manualtag FROM results ORDER BY result_id DESC")
            manualtags = cur.fetchall()
            print("The number of manualtags: ", cur.rowcount)
            cur.execute("SELECT distance FROM results ORDER BY result_id DESC")
            distances = cur.fetchall()
            print("The number of distances: ", cur.rowcount)
            cur.execute("SELECT rank FROM results ORDER BY result_id DESC")
            ranks = cur.fetchall()
            print("The number of ranks: ", cur.rowcount)
            cur.execute("SELECT searchstring FROM results ORDER BY result_id DESC")
            searchstrings = cur.fetchall()
            print("The number of searchstrings: ", cur.rowcount)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return paras, locations, autotags, manualtags, distances, ranks, searchstrings

    def get_paragraphs_fileid(self, file_id):
        """ query parts from the parts table """
        conn = None
        paras = None
        paraids = None
        autotags = None
        manualtags = None
        try:
            params = config()
            conn = psycopg2.connect(**params)
            cur = conn.cursor()
            cur.execute("SELECT paragraph FROM paragraphs WHERE (file_id=%s) ORDER BY paragraph_id;",(file_id,))
            paras = cur.fetchall()
            print("The number of paragraphs: ", cur.rowcount)
            cur.execute("SELECT paragraph_id FROM paragraphs WHERE (file_id=%s) ORDER BY paragraph_id;",(file_id,))
            paraids = cur.fetchall()
            print("The number of paragraph ids: ", cur.rowcount)
            cur.execute("SELECT automatic_tag FROM paragraphs WHERE (file_id=%s) ORDER BY paragraph_id;",(file_id,))
            autotags = cur.fetchall()
            print("The number of autotags: ", cur.rowcount)
            cur.execute("SELECT manual_tag FROM paragraphs WHERE (file_id=%s) ORDER BY paragraph_id;",(file_id,))
            manualtags = cur.fetchall()
            print("The number of manualtags: ", cur.rowcount)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
        return paras, paraids, autotags, manualtags

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