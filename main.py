from flask import Flask, request
from flask_cors import CORS, cross_origin
from tikaparser import *
from postgreshandler import *
from semanticsearch import *
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'
#Allow specific origin
cors = CORS(app, resources={r"/search": {"origins": ["http://54.225.26.77", "http://localhost:3000"]}})

@app.route('/search', methods=['GET', 'POST'])
#@cross_origin()
def search():
    directory = "D:/Business/Infineon contest/SemanticSearch/textparser/documents"
    filename = "LearningParseStructures.docx"
    if request.method == 'POST':
        #time.sleep(5)
        parser = tikaparser(directory,filename)
        filepath = os.path.join(directory, filename)
        file, file_extension = os.path.splitext(filepath)
        print(filepath)
        paragraphs = parser.parse()
        # for paragraph in paragraphs:
        #     print(paragraph)
        #     print('\n')
        matching_rows = postgres.get_fileinfo(directory, filename)
        print("matching rows", matching_rows)
        if matching_rows==0:
            file_id = postgres.insert_master(directory, filename, file_extension)
            postgres.insert_paragraph_list(file_id, paragraphs)
            print("file_id = ", file_id)
        # updated_rows = postgres.update_paragraph('1', "r u ok", "fine")
        # print("updated_rows = ", updated_rows)
        paras, filepaths = postgres.get_paragraphs()
        # print("number of paras = ", len(paras))
        # print("number of filepaths = ", len(filepaths))
        search = semantic()
        search.create_vector(paras, filepaths)
        auto_tag = search.create_autotag()
        print("size of auto_tag = ", len(auto_tag))
        search.create_refdf()
        search.create_searchindex()
        search.search_engine("How are you?", 10)
        # deleted_rows = postgres.delete_file('1')
        # print("deleted_rows in master = ", deleted_rows)
        # deleted_rows = postgres.delete_paragraphs('1')
        # print("deleted_rows in paragraphs = ", deleted_rows)
        return "You are using POST"
    else:
        return "You are probably using GET"

if __name__ == "__main__":
    postgres = postgressql()
    # postgres.connect()
    postgres.create_tables()
    app.run(debug=True)