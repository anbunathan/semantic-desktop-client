from flask import Flask, request
from flask import jsonify as JFY
from flask_cors import CORS, cross_origin
from tikaparser import *
from postgreshandler import *
from semanticsearch import *
import time
from os import listdir
from os.path import isfile, join
import sys
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'
#Allow specific origin
cors = CORS(app, resources={r"/search/*": {"origins": ["http://54.225.26.77", "http://localhost:3000"]}})
q2emb = None

@app.route('/search/deletedirectory/<uuid>', methods=['GET', 'POST'])
def deletedirectory(uuid):
    print("You are getting delete directory request")
    content = request.json
    directory = content['directory']
    inputtype = content['inputtype']
    directoryid = content['directoryid']
    print("directory path = ", directory)
    print("input type = ", inputtype)
    print("directoryid = ", directoryid)
    json_result = search.delete_directory_path(postgres, directoryid)
    print("response for delete_directory_path = ", json_result)
    return json_result

@app.route('/search/updatedirectory/<uuid>', methods=['GET', 'POST'])
def updatedirectory(uuid):
    print("You are getting update directory request")
    content = request.json
    directory = content['directory']
    inputtype = content['inputtype']
    directoryid = content['directoryid']
    print("directory path = ", directory)
    print("input type = ", inputtype)
    print("directoryid = ", directoryid)
    json_result = search.update_directory_path(postgres, directoryid, directory, inputtype)
    print("response for update_directory_path = ", json_result)
    return json_result

@app.route('/search/getdirectorybyid/<uuid>', methods=['GET', 'POST'])
def getdirectorybyid(uuid):
    print("You are getting get directory by id request")
    content = request.json
    directoryid = content
    print("directoryid = ",directoryid)
    json_result = search.get_directory_path_byid(postgres, directoryid)
    json_object = JFY({"item": json_result})
    print("json_results = ", json_object)
    print("response for get_directory_path_byid = ", json_object.get_json())
    return json_object

@app.route('/search/getdirectory/<uuid>', methods=['GET', 'POST'])
def getdirectory(uuid):
    print("You are getting insert directory request")
    content = request.json
    json_result = search.get_directory_path(postgres)
    json_object = JFY({"items": json_result})
    print("response for get_directory_path = ", json_object.get_json())
    return json_object

@app.route('/search/adddirectory/<uuid>', methods=['GET', 'POST'])
def adddirectory(uuid):
    print("You are getting insert directory request")
    content = request.json
    directory_path = content['todo_description']
    directory_path = directory_path.replace("\\","/")
    input_type = content['todo_priority']
    print("directory path = ", directory_path)
    print("input type = ", input_type)
    json_result = search.insert_directory_path(postgres, directory_path, input_type)
    print("response for insert_directory_path = ", json_result)
    return json_result


@app.route('/search/results/<uuid>', methods=['GET', 'POST'])
def results(uuid):
    if request.method == 'POST':
        print("You are getting results request")
        json_result = search.get_results(postgres)
        json_object = JFY({"items":json_result})
        print("json_results = ",json_object)
        print("response = ", json_object.get_json())
        return json_object
    else:
        print("may be a GET request")
        return "may be a GET request"

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
    # global q2emb
    if request.method == 'POST':
        content = request.json
        print ("search string = ", content['todo_description'])
        print("You are getting search request")
        search_string = content['todo_description']
        # json_dump = jsonify({"uuid": uuid})
        # json_object = json_response({})
        json_object = JFY({})
        resultset=None
        if search.q2emb!=None:
            json_result = search.search_query(postgres, search_string, 10)
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

# @app.route('/search/setenv/<uuid>', methods=['GET', 'POST'])
# #@cross_origin()
# def setenv(uuid):
#     # global q2emb
#     print("You are getting environment setup request")
#     content = request.json
#     print("content = ", content)
#     print("content = ", len(content))
#     base_dir = ''
#     data_path = Path(base_dir + './data/processed_data/')
#     directory_list = []
#     inputtype_list = []
#     directoryid_list = []
#     for item in content:
#         print(item['props']['directory']['directory'])
#         directory = item['props']['directory']['directory']
#         inputtype = item['props']['directory']['inputtype']
#         directoryid = item['props']['directory']['directoryid']
#         print("directory  = ", directory)
#         print("inputtype = ", inputtype)
#         print("directoryid = ", directoryid)
#         directory_list.append(directory[0])
#         inputtype_list.append(inputtype[0])
#         directoryid_list.append(directoryid[0])
#     print("directory list = ", directory_list)
#     print("inputtype_list = ", inputtype_list)
#     print("directoryid_list = ", directoryid_list)
#     try:
#         idx = 0
#         for directory in directory_list:
#             print("directory = ", directory)
#             if (inputtype_list[idx] == 'Directory'):
#                 files = [f for f in listdir(directory) if isfile(join(directory, f))]
#                 print("files", files)
#                 for filename in files:
#                     print("file = ", filename)
#                     matching_rows = postgres.get_fileinfo(directory, filename)
#                     print("matching rows", matching_rows)
#                     if matching_rows == 0:
#                         filepath = os.path.join(directory, filename)
#                         file, file_extension = os.path.splitext(filepath)
#                         print(file_extension)
#                         directory_id = directoryid_list[idx]
#                         if file_extension in [".doc", ".docx", ".csv", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf"]:
#                             print("File type is supported")
#                             paragraphs = parser.parse(directory, filename)
#                             file_id = postgres.insert_master(directory, filename, file_extension, directory_id)
#                             postgres.insert_paragraph_list(file_id, paragraphs)
#                             print("file_id = ", file_id)
#             else:
#                 print("Directory = %s is of type 'SQLfile'" % directory)
#             idx = idx+1
#         paras, filepaths, paraids, autotags, manualtags = postgres.get_paragraphs()
#         with open(data_path / 'without_docstrings.function', 'w', encoding='utf-8') as f:
#             for item in paras:
#                 item = re.sub("\n", "", str(item))
#                 f.write("%s\n" % item)
#         with open(data_path / 'without_docstrings.paraids', 'w', encoding='utf-8') as f:
#             for item in paraids:
#                 f.write("%s\n" % item)
#         with open(data_path / 'without_docstrings.lineage', 'w', encoding='utf-8') as f:
#             for item in filepaths:
#                 f.write("%s\n" % item)
#         with open(data_path / 'without_docstrings.manualtags', 'w', encoding='utf-8') as f:
#             for item in manualtags:
#                 f.write("%s\n" % item)
#         print("without_docstrings files are created")
#         search.create_vector()
#         print("vectors are created")
#         search.create_autotag(postgres)
#         print("Autotags are created")
#         search.create_refdf()
#         print("Ref_dfs are created")
#         search.create_searchindex()
#         print("Searchindexs are created")
#         # q2emb = search.search_engine()
#         print("Search Environment is Set")
#         return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
#     except Exception as e:
#         print("Error in set environement = ", e)
#         # print(sys.exc_value)
#         response, status = search.unexpected_error('Error in set environement')
#         return response, status
#     finally:
#         print("Exit from set environment")
#     # return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}

@app.route('/search/setenv/<uuid>', methods=['GET', 'POST'])
#@cross_origin()
def setenv(uuid):
    # global q2emb
    print("You are getting environment setup request")
    content = request.json
    print("content = ", content)
    print("content = ", len(content))
    base_dir = ''
    data_path = Path(base_dir + './data/processed_data/')
    directory_list = []
    inputtype_list = []
    directoryid_list = []
    for item in content:
        print(item['props']['directory']['directory'])
        directory = item['props']['directory']['directory']
        inputtype = item['props']['directory']['inputtype']
        directoryid = item['props']['directory']['directoryid']
        print("directory  = ", directory)
        print("inputtype = ", inputtype)
        print("directoryid = ", directoryid)
        directory_list.append(directory[0])
        inputtype_list.append(inputtype[0])
        directoryid_list.append(directoryid[0])
    print("directory list = ", directory_list)
    print("inputtype_list = ", inputtype_list)
    print("directoryid_list = ", directoryid_list)
    try:
        idx = 0
        for directory in directory_list:
            print("directory = ", directory)
            if (inputtype_list[idx] == 'Directory'):
                files = [f for f in listdir(directory) if isfile(join(directory, f))]
                print("files", files)
                for filename in files:
                    print("file = ", filename)
                    matching_rows = postgres.get_fileinfo(directory, filename)
                    print("matching rows", matching_rows)
                    if matching_rows == 0:
                        filepath = os.path.join(directory, filename)
                        file, file_extension = os.path.splitext(filepath)
                        print(file_extension)
                        directory_id = directoryid_list[idx]
                        if file_extension in [".doc", ".docx", ".csv", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf"]:
                            print("File type is supported")
                            paragraphs = parser.parse(directory, filename)
                            file_id = postgres.insert_master(directory, filename, file_extension, directory_id)
                            print("file_id = ", file_id)
                            size_para = len(paragraphs)
                            print("size of paragraphs = ", size_para)
                            if size_para==0:
                                continue
                            postgres.insert_paragraph_list(file_id, paragraphs)
                            search.create_vector(postgres, file_id)
                            print("vectors are created")
                            search.create_autotag(postgres, file_id)
                            print("Autotags are created")

            else:
                print("Directory = %s is of type 'SQLfile'" % directory)
            idx = idx+1
        paras, filepaths, paraids, autotags, manualtags = postgres.get_paragraphs()
        print("paraids = ", paraids)
        with open(data_path / 'without_docstrings.function', 'w', encoding='utf-8') as f:
            for item in paras:
                item = re.sub("\n", "", str(item))
                f.write("%s\n" % item)
        with open(data_path / 'without_docstrings.paraids', 'w', encoding='utf-8') as f:
            for item in paraids:
                f.write("%s\n" % item)
        with open(data_path / 'without_docstrings.lineage', 'w', encoding='utf-8') as f:
            for item in filepaths:
                f.write("%s\n" % item)
        with open(data_path / 'without_docstrings.autotag', 'w', encoding='utf-8') as f:
            for item in autotags:
                f.write("%s\n" % item)
        with open(data_path / 'without_docstrings.manualtags', 'w', encoding='utf-8') as f:
            for item in manualtags:
                f.write("%s\n" % item)
        print("without_docstrings files are created")
        # search.create_vector()
        # print("vectors are created")
        # search.create_autotag(postgres)
        # print("Autotags are created")
        search.create_refdf()
        print("Ref_dfs are created")
        search.create_searchindex(postgres)
        print("Searchindexs are created")
        # q2emb = search.search_engine()
        print("Search Environment is Set")
        return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}
    except Exception as e:
        print("Error in set environement = ", e)
        # print(sys.exc_value)
        response, status = search.unexpected_error('Error in set environement')
        return response, status
    finally:
        print("Exit from set environment")
    # return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


if __name__ == "__main__":
    # global q2emb
    search = semantic()
    postgres = postgressql()
    postgres.create_tables()
    parser = tikaparser()
    search.search_engine()
    app.run(debug=True)