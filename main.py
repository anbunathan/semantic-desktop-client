from flask import Flask, request
from flask import jsonify as JFY
from flask_cors import CORS, cross_origin
from tikaparser import *
from postgreshandler import *
from semanticsearch import *
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'
#Allow specific origin
cors = CORS(app, resources={r"/search/*": {"origins": ["http://54.225.26.77", "http://localhost:3000"]}})
q2emb = None

@app.route('/search/records/<uuid>', methods=['GET', 'POST'])
def records(uuid):
    if request.method == 'POST':
        print("You are getting records request")
        json_result = search.get_document_records(postgres)
        json_object = JFY({"items":json_result})
        print("json_results = ",json_object)
        print("response = ", json_object.get_json())
        return json_object
    else:
        print("may be a GET request")
        return "may be a GET request"

@app.route('/search/getmanualtag/<uuid>', methods=['GET', 'POST'])
def getmanualtag(uuid):
    if request.method == 'POST':
        print("You are getting getmanualtag request")
        content = request.json
        print("paragraph id = ", content)
        json_object = JFY({})
        json_result = search.get_manualtag(postgres,content)
        json_object = JFY({"item":json_result})
        print("json_results = ",json_object)
        print("response = ", json_object.get_json())
        return json_object
    else:
        print("may be a GET request")
        return "may be a GET request"

@app.route('/search/updatemanualtag/<uuid>', methods=['GET', 'POST'])
def updatemanualtag(uuid):
    if request.method == 'POST':
        print("You are getting updatemanualtag request")
        content = request.json
        manualtag = content['manual_tag']
        paraid = content['paraid']
        print("update input = ", content)
        print("manualtag input = ", manualtag)
        print("paraid input = ", paraid)
        json_object = JFY({})
        json_result = search.update_manualtag(postgres,paraid, manualtag)
        json_object = JFY({"updatedrows":json_result})
        print("json_results = ",json_object)
        print("response = ", json_object.get_json())
        return json_object
    else:
        print("may be a GET request")
        return "may be a GET request"

@app.route('/search/query/<uuid>', methods=['GET', 'POST'])
def query(uuid):
    global q2emb
    if request.method == 'POST':
        content = request.json
        print ("search string = ", content['todo_description'])
        print("You are getting search request")
        # json_dump = jsonify({"uuid": uuid})
        # json_object = json_response({})
        json_object = JFY({})
        resultset=None
        if q2emb!=None:
            json_result = search.search_query(q2emb, "How are you?", 10)
            # json_result = json.dumps(json_result)
            json_object = JFY({"items":json_result})
            print("json_results = ",json_object)

        else:
            print("Search Engine is not Valid")
        # jsonStr = json_response(json_object)
        # return jsonify(Results=jsonStr)
        print("response = ", json_object.get_json())
        return json_object
    else:
        print("may be a GET request")
        return "may be a GET request"

# def json_response(payload, status=200):
#  return (json.dumps(payload), status, {'content-type': 'application/json'})

@app.route('/search/setenv/<uuid>', methods=['GET', 'POST'])
#@cross_origin()
def setenv(uuid):
    global q2emb
    content = request.json
    print("content = ", content)
    print("content = ", len(content))
    directory_list = []
    for item in content:
        print("item = ", item)
        print("key = ", item['key'])
        print("ID = ", item['props']['todo']['_id'])
        print("directory = ", item['props']['todo']['todo_description'])
        print("type = ", item['props']['todo']['todo_priority'])
        directory_list.append(item['props']['todo']['todo_description'])
    # list = str(request.values.get("todoList", "None"))
    # print("directory list = ", list)
    print("You are getting environment setup request")
    base_dir = ''
    data_path = Path(base_dir + './data/processed_data/')
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
        paras, filepaths, paraids, autotags, manualtags = postgres.get_paragraphs()
        with open(data_path/'without_docstrings.function', 'w', encoding='utf-8') as f:
            for item in paragraphs:
                f.write("%s" % item)
        with open(data_path/'without_docstrings.paraids', 'w', encoding='utf-8') as f:
            for item in paraids:
                f.write("%s\n" % item)
        with open(data_path/'without_docstrings.lineage', 'w', encoding='utf-8') as f:
            for item in filepaths:
                f.write("%s\n" % item)
        with open(data_path/'without_docstrings.manualtags', 'w', encoding='utf-8') as f:
            for item in manualtags:
                f.write("%s\n" % item)
        # print("number of paras = ", len(paras))
        # print("number of filepaths = ", len(filepaths))
        search.create_vector()
        search.create_autotag(postgres)
        search.create_refdf()
        search.create_searchindex()
        q2emb = search.search_engine()
        print("Search Environment is Set")
        # deleted_rows = postgres.delete_file('1')
        # print("deleted_rows in master = ", deleted_rows)
        # deleted_rows = postgres.delete_paragraphs('1')
        # print("deleted_rows in paragraphs = ", deleted_rows)
        return "You are using POST"
    else:
        return "You are probably using GET"



if __name__ == "__main__":
    search = semantic()
    postgres = postgressql()
    # postgres.connect()
    #execute DROP TABLE paragraphs; from sql shell if schema is changed
    postgres.create_tables()
    app.run(debug=True)