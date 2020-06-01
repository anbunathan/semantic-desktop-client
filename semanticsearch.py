
import os
import glob
from os import path
from flask import jsonify
from keras.models import load_model
from pathlib import Path
import numpy as np
from sqlalchemy import JSON
from postgreshandler import *
from seq2seq_utils import load_text_processor
from ktext.preprocess import processor
import torch
import nmslib
from lang_model_utils import load_lm_vocab
from lang_model_utils import Query2Emb
from general_utils import create_nmslib_search_index
import csv
from fastai import *
from seq2seq_utils import Seq2Seq_Inference
import pandas as pd
import json
from keras import backend as K


class semantic:
    base_dir = ''
    code2emb_path = Path(base_dir+'./data/code2emb/')
    seq2seq_path = Path(base_dir+'./data/seq2seq/')
    data_path = Path(base_dir+'./data/processed_data/')
    output_path = Path(base_dir + './data/search')
    input_path = Path(base_dir +'./data/processed_data/')
    npy_path = Path(base_dir + './data/npy/')
    ref_df = None
    code2emb_model = None
    num_encoder_tokens_vector = None
    enc_pp_vector = None
    seq2seq_inf = None
    q2emb = None

    def load_models(self):
        K.clear_session()
        seq2seq_Model = load_model(str(self.seq2seq_path / 'code_summary_seq2seq_model.h5'))
        num_encoder_tokens, enc_pp = load_text_processor(self.seq2seq_path / 'py_code_proc_v2.dpkl')
        num_decoder_tokens, dec_pp = load_text_processor(self.seq2seq_path / 'py_comment_proc_v2.dpkl')
        self.seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                             decoder_preprocessor=dec_pp,
                                             seq2seq_model=seq2seq_Model)
        self.code2emb_model = load_model(str(self.code2emb_path / 'code2emb_model.hdf5'), custom_objects=None,
                                         compile=False)
        self.num_encoder_tokens_vector, self.enc_pp_vector = load_text_processor(
            self.seq2seq_path / 'py_code_proc_v2.dpkl')

    def clear_session(self):
        K.clear_session()

    def load_seq2seq_model(self):
        K.clear_session()
        seq2seq_Model = load_model(str(self.seq2seq_path / 'code_summary_seq2seq_model.h5'))
        num_encoder_tokens, enc_pp = load_text_processor(self.seq2seq_path / 'py_code_proc_v2.dpkl')
        num_decoder_tokens, dec_pp = load_text_processor(self.seq2seq_path / 'py_comment_proc_v2.dpkl')
        self.seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                        decoder_preprocessor=dec_pp,
                                        seq2seq_model=seq2seq_Model)

    def load_code2emb_model(self):
        K.clear_session()
        self.code2emb_model = load_model(str(self.code2emb_path / 'code2emb_model.hdf5'), custom_objects=None, compile=False)
        self.num_encoder_tokens_vector, self.enc_pp_vector = load_text_processor(self.seq2seq_path / 'py_code_proc_v2.dpkl')

    def create_vector_trial(self, postgres, file_id):
        # with open(self.data_path/'without_docstrings.function', 'r', encoding='utf-8') as f:
        #     no_docstring_funcs = f.readlines()
        paras, paraids, autotags, manualtags = postgres.get_paragraphs_fileid(file_id)
        paras = [str(item) for item in paras]
        no_docstring_funcs = paras
        print("no_docstring_funcs = ", no_docstring_funcs)
        print("Going to transform_parallel")
        # encinp = self.enc_pp_vector.transform_parallel(no_docstring_funcs)
        encinp = self.enc_pp_vector.transform(no_docstring_funcs)
        # np.save(self.code2emb_path/'nodoc_encinp.npy', encinp)
        # encinp = np.load(self.code2emb_path/'nodoc_encinp.npy')
        print("Going to create the vector")
        nodoc_vecs = self.code2emb_model.predict(encinp, batch_size=2000)
        # make sure the number of output rows equal the number of input rows
        assert nodoc_vecs.shape[0] == encinp.shape[0]
        # np.save(self.code2emb_path/'nodoc_vecs.npy', nodoc_vecs)
        npy_filename = str(file_id) + "####" + "nodoc_vecs.npy"
        np.save(self.npy_path / npy_filename, nodoc_vecs)
        print("Vector is created successfully")

    def create_autotag_trial1(self, postgres, file_id):
        paras, paraids, autotags, manualtags = postgres.get_paragraphs_fileid(file_id)
        paras = [str(item) for item in paras]
        no_docstring_funcs = paras
        no_docstring_paraids = paraids
        print("no_docstring_paraids = ", no_docstring_paraids)
        print("size of paragraphs = ", len(no_docstring_funcs))
        print("size of paraids = ", len(no_docstring_paraids))
        demo_testdf = pd.DataFrame({'code': no_docstring_funcs, 'comment': '', 'ref': ''})
        auto_tag = self.seq2seq_inf.demo_model_predictions(n=15, df=demo_testdf)
        print("size of auto_tag = ", len(auto_tag))
        with open(self.data_path/'without_docstrings.autotag', 'w', encoding='utf-8') as f:
            index = 0
            for item in auto_tag:
                f.write("%s\n" % item)
                paraid = no_docstring_paraids[index]
                # paraid = paraid.strip()
                updated_rows = postgres.update_autotag(paraid, item)
                index=index+1
        print("Autotag is created successfully")

    def create_autotag_trial(self, postgres, file_id_list):
        paras_list = []
        paras_id_list = []
        for file_id in file_id_list:
            paras, paraids, autotags, manualtags = postgres.get_paragraphs_fileid(file_id)
            paras = [str(item) for item in paras]
            paras_list = paras_list + paras
            paras_id_list = paras_id_list+paraids
        no_docstring_funcs = paras_list
        no_docstring_paraids = paras_id_list
        print("no_docstring_funcs = ", no_docstring_funcs)
        print("no_docstring_paraids = ", no_docstring_paraids)
        print("size of paragraphs = ", len(no_docstring_funcs))
        print("size of paraids = ", len(no_docstring_paraids))
        demo_testdf = pd.DataFrame({'code': no_docstring_funcs, 'comment': '', 'ref': ''})
        auto_tag = self.seq2seq_inf.demo_model_predictions(n=15, df=demo_testdf)
        print("size of auto_tag = ", len(auto_tag))
        with open(self.data_path/'without_docstrings.autotag', 'w', encoding='utf-8') as f:
            index = 0
            for item in auto_tag:
                f.write("%s\n" % item)
                paraid = no_docstring_paraids[index]
                # paraid = paraid.strip()
                updated_rows = postgres.update_autotag(paraid, item)
                index=index+1
        print("Autotag is created successfully")

    def create_vector(self, postgres, file_id):
        K.clear_session()
        print("Going to load code2emb_model")
        self.code2emb_model = load_model(str(self.code2emb_path / 'code2emb_model.hdf5'), custom_objects=None,
                                         compile=False)
        print("Going to load_text_processor")
        self.num_encoder_tokens_vector, self.enc_pp_vector = load_text_processor(
            self.seq2seq_path / 'py_code_proc_v2.dpkl')
        # with open(self.data_path/'without_docstrings.function', 'r', encoding='utf-8') as f:
        #     no_docstring_funcs = f.readlines()
        paras, paraids, autotags, manualtags = postgres.get_paragraphs_fileid(file_id)
        paras = [str(item) for item in paras]
        no_docstring_funcs = paras
        print("no_docstring_funcs = ", no_docstring_funcs)
        print("Going to transform_parallel")
        # encinp = self.enc_pp_vector.transform_parallel(no_docstring_funcs)
        encinp = self.enc_pp_vector.transform(no_docstring_funcs)
        # np.save(self.code2emb_path/'nodoc_encinp.npy', encinp)
        # encinp = np.load(self.code2emb_path/'nodoc_encinp.npy')
        print("Going to create the vector")
        nodoc_vecs = self.code2emb_model.predict(encinp, batch_size=2000)
        # make sure the number of output rows equal the number of input rows
        assert nodoc_vecs.shape[0] == encinp.shape[0]
        # np.save(self.code2emb_path/'nodoc_vecs.npy', nodoc_vecs)
        npy_filename = str(file_id) + "####" + "nodoc_vecs.npy"
        np.save(self.npy_path / npy_filename, nodoc_vecs)

        K.clear_session()
        print("Vector is created")

    def create_autotag(self, postgres, file_id):
        K.clear_session()
        seq2seq_Model = load_model(str(self.seq2seq_path / 'code_summary_seq2seq_model.h5'))
        num_encoder_tokens, enc_pp = load_text_processor(self.seq2seq_path / 'py_code_proc_v2.dpkl')
        num_decoder_tokens, dec_pp = load_text_processor(self.seq2seq_path / 'py_comment_proc_v2.dpkl')
        self.seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                             decoder_preprocessor=dec_pp,
                                             seq2seq_model=seq2seq_Model)
        paras, paraids, autotags, manualtags = postgres.get_paragraphs_fileid(file_id)
        paras = [str(item) for item in paras]
        no_docstring_funcs = paras
        no_docstring_paraids = paraids
        print("no_docstring_paraids = ", no_docstring_paraids)
        print("size of paragraphs = ", len(no_docstring_funcs))
        print("size of paraids = ", len(no_docstring_paraids))
        demo_testdf = pd.DataFrame({'code': no_docstring_funcs, 'comment': '', 'ref': ''})
        auto_tag = self.seq2seq_inf.demo_model_predictions(n=15, df=demo_testdf)
        print("size of auto_tag = ", len(auto_tag))
        with open(self.data_path/'without_docstrings.autotag', 'w', encoding='utf-8') as f:
            index = 0
            for item in auto_tag:
                f.write("%s\n" % item)
                paraid = no_docstring_paraids[index]
                # paraid = paraid.strip()
                updated_rows = postgres.update_autotag(paraid, item)
                index=index+1
        K.clear_session()

    def create_refdf(self):
        # read file of urls
        url_df = pd.read_csv(self.input_path/'without_docstrings.lineage', header=None, names=['url'])
        code_df = pd.read_csv(open(self.input_path/'without_docstrings.function','rU',encoding='utf-8'), sep='delimiter',quoting=csv.QUOTE_NONE, engine='python', header=None, names=['code'])
        aututag_df = pd.read_csv(open(self.data_path/'without_docstrings.autotag', 'rU', encoding='utf-8'),
                              sep='delimiter', quoting=csv.QUOTE_NONE, engine='python', header=None, names=['autotag'])
        manualtag_df = pd.read_csv(open(self.data_path / 'without_docstrings.manualtags', 'rU', encoding='utf-8'),
                                 sep='delimiter', quoting=csv.QUOTE_NONE, engine='python', header=None,
                                 names=['manualtag'])
        # make sure these files have same number of rows
        print("code_df.shape[0] = ",code_df.shape[0])
        print("url_df.shape[0] = ",url_df.shape[0])
        print("aututag_df.shape[0] = ", aututag_df.shape[0])
        print("manualtag_df.shape[0] = ", manualtag_df.shape[0])
        assert code_df.shape[0] == url_df.shape[0] == aututag_df.shape[0] == manualtag_df.shape[0]
        # collect these two together into a dataframe
        ref_df = pd.concat([url_df, code_df, aututag_df, manualtag_df], axis=1).reset_index(drop=True)
        self.ref_df = ref_df
        print(ref_df.head())

    def create_searchindex(self, postgres):
        npyfilespath = self.npy_path
        with open(self.code2emb_path / 'nodoc_vecs.npy', 'wb') as f_handle:
            # os.chdir(npyfilespath)
            first = False
            dataArray = None
            # os.chdir(npyfilespath)
            for npfile in glob.glob("data/npy/*"):
                # Find the path of the file
                filepath = os.path.join("", npfile)
                print("filepath = ", filepath)
                temp = npfile.split('####')
                print("temp[0] = ", temp[0])
                if ("\\" in temp[0]):
                    temp = temp[0].split("\\")
                elif ("/" in temp[0]):
                    temp = temp[0].split("/")
                print("temp=", temp)
                fileid = temp[-1]
                print("fileid = ", fileid)
                matching_rows = postgres.check_fileid_exists(fileid)
                if (matching_rows > 0):
                    if first == False:
                        # Load file
                        dataArray = np.load(filepath)
                        first = True
                        print("this is first file")
                    else:
                        dataArray = np.concatenate((dataArray, np.load(filepath)), axis=0)
                        print("this is not first file")
            np.save(f_handle, dataArray)
        nodoc_vecs = np.load(self.code2emb_path / 'nodoc_vecs.npy')
        print("self.ref_df.shape[0] = ", self.ref_df.shape[0])
        print("nodoc_vecs.shape[0] = ", nodoc_vecs.shape[0])
        assert nodoc_vecs.shape[0] == self.ref_df.shape[0]
        search_index = create_nmslib_search_index(nodoc_vecs)
        search_index.saveIndex('search_index.nmslib')
        print("SearchIndex is created")

    def search_engine(self):
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        lang_model = torch.load(Path(self.base_dir + './data/lang_model/lang_model_cpu_v2.torch'),map_location='cpu')
        # lang_model = torch.load('lang_model_cpu_v2.torch', map_location='cpu')
        vocab = load_lm_vocab(Path(self.base_dir + './data/lang_model/vocab_v2.cls'))
        # embpath = Path(self.base_dir + './data/search/embeddings.var')
        self.q2emb = Query2Emb(lang_model=lang_model,vocab=vocab)

    def search_query(self, postgres, str_search, k=2):
        query2emb_func = self.q2emb.emb_mean
        search_index = nmslib.init(method='hnsw', space='cosinesimil')
        search_index.loadIndex('search_index.nmslib')
        query = query2emb_func(str_search)
        idxs, dists = search_index.knnQuery(query, k=k)
        length = len(dists)
        print("length of dists = ", length)
        rankcounter=length+1
        resultset = []
        json_results = []
        # iterate over todos
        for idx, dist in zip(sorted(idxs, reverse=True), sorted(dists, reverse=True)):
            rankcounter = rankcounter-1
            code = self.ref_df.iloc[idx].code
            url = self.ref_df.iloc[idx].url
            autotag = self.ref_df.iloc[idx].autotag
            manualtag = self.ref_df.iloc[idx].manualtag
            print(f'cosine dist:{dist:.4f}  Location: {url}\n---------------\n')
            print("auto tag: ",str(autotag))
            print("Paragraph = ", str(code))
            rankstr = str(rankcounter)
            diststr = str(dist)
            postgres.insert_result_list(
                code, url, diststr, rankstr, str_search, autotag, manualtag
            )
            t = {'para': code,
                 'location': url,
                 'autotag': autotag,
                 'manualtag': manualtag,
                 'distance': diststr,
                 'rank': rankstr

                 }
            json_results.append(t)
        return json_results

    def get_document_records(self, postgres):
        paras, filepaths, paraids, autotags, manualtags = postgres.get_paragraphs()
        length = len(paras)
        json_results = []
        for idx in range(length):
            filepath = str(filepaths[idx])
            paraid = (paraids[idx])
            para = (paras[idx])
            autotag = (autotags[idx])
            manualtag = (manualtags[idx])
            t = {'filepath': filepath,
                 'paraid': paraid,
                 'para': para,
                 'autotag': autotag,
                 'manualtag': manualtag
                 }
            json_results.append(t)
        return json_results

    def get_results(self, postgres):
        paras, locations, autotags, manualtags, distances, ranks, searchstrings = postgres.get_results()
        length = len(paras)
        json_results = []
        for idx in range(length):
            para = (paras[idx])
            location = (locations[idx])
            autotag = (autotags[idx])
            manualtag = (manualtags[idx])
            distance = (distances[idx])
            rank = (ranks[idx])
            searchstring = (searchstrings[idx])
            t = {
                 'para': para,
                 'location': location,
                 'autotag': autotag,
                 'manualtag': manualtag,
                 'distance': distance,
                 'rank': rank,
                 'searchstring': searchstring
                 }
            json_results.append(t)
        return json_results

    def get_manualtag(self, postgres, paraid):
        manualtag = postgres.get_manualtag(paraid)
        return manualtag

    def update_manualtag(self, postgres, paraid, manualtag):
        updated_rows = postgres.update_manualtag(paraid, manualtag)
        return str(updated_rows)

    def unexpected_error(self, errror_message):
        status_code = 400
        success = False
        response = {
            'success': success,
            'error': {
                'type': 'UnexpectedException',
                'message': errror_message
            }
        }
        return jsonify(response), status_code

    def success_response(self, success_message):
        status_code = 200
        success = True
        response = {
            'success': success,
            'message': {
                'type': 'SuccessfulOperation',
                'message': success_message
            }
        }
        return jsonify(response), status_code

    def insert_directory_path(self, postgres, directory_path, input_type):
        matching_rows, status = postgres.get_directory_info(directory_path)
        if(matching_rows == 0 and status == 'success'):
            directory_id, status = postgres.insert_directory(directory_path, input_type)
            print("directory id of inserted direcory = ", directory_id)
        if(matching_rows != 0):
            print("Directory already exists in database")
        if(status=='success'):
            response = self.success_response('directory added successfully')
        else:
            response = self.success_response('adding directory failed')
        return response

    def get_directory_path(self, postgres):
        directories, inputtypes, directoryids = postgres.get_directory_path()
        print("directories = ", directories)
        print("inputtypes = ", inputtypes)
        length = len(directories)
        json_results = []
        for idx in range(length):
            directory = (directories[idx])
            inputtype = (inputtypes[idx])
            directoryid = (directoryids[idx])
            t = {'directory': directory,
                 'inputtype': inputtype,
                 'directoryid': directoryid
                 }
            json_results.append(t)
        return json_results

    def get_directory_path_byid(self, postgres, directoryid):
        directory, inputtype = postgres.get_directory_path_byid(directoryid)
        json_results = {
            'directory': directory,
            'inputtype': inputtype,
            'directoryid': directoryid
             }
        return json_results

    def update_directory_path(self, postgres, directoryid, directory_path, input_type):
        updated_rows, status = postgres.update_directory_path(directoryid, directory_path, input_type)
        print("updated_rows = ", updated_rows)
        if(status=='success'):
            response = self.success_response('directory updated successfully')
        else:
            response = self.success_response('updating directory failed')
        return response

    def delete_directory_path(self, postgres, directoryid):
        deleted_rows, status = postgres.delete_directory(directoryid)
        print("deleted_rows = ", deleted_rows)
        if(status=='success'):
            response = self.success_response('directory deleted successfully')
        else:
            response = self.success_response('deleting directory failed')
        return response