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
from lang_model_utils import load_lm_vocab, Query2Emb
from general_utils import create_nmslib_search_index
import csv
from fastai import *
from seq2seq_utils import Seq2Seq_Inference
import pandas as pd
import json

class semantic:
    base_dir = ''
    code2emb_path = Path(base_dir+'./data/code2emb/')
    seq2seq_path = Path(base_dir+'./data/seq2seq/')
    data_path = Path(base_dir+'./data/processed_data/')
    output_path = Path(base_dir + './data/search')
    input_path = Path(base_dir +'./data/processed_data/')
    ref_df = None

    def create_vector(self):
        code2emb_model = load_model(str(self.code2emb_path/'code2emb_model.hdf5'), custom_objects=None, compile=False)
        num_encoder_tokens, enc_pp = load_text_processor(self.seq2seq_path/'py_code_proc_v2.dpkl')
        # no_docstring_funcs = self.data_path/'train.function'
        with open(self.data_path/'without_docstrings.function', 'r', encoding='utf-8') as f:
            no_docstring_funcs = f.readlines()
        encinp = enc_pp.transform_parallel(no_docstring_funcs)
        np.save(self.code2emb_path/'nodoc_encinp.npy', encinp)
        encinp = np.load(self.code2emb_path/'nodoc_encinp.npy')
        nodoc_vecs = code2emb_model.predict(encinp, batch_size=2000)
        # make sure the number of output rows equal the number of input rows
        assert nodoc_vecs.shape[0] == encinp.shape[0]
        np.save(self.code2emb_path/'nodoc_vecs.npy', nodoc_vecs)
        print("Vector is created")

    def create_autotag(self, postgres):
        seq2seq_Model = load_model(str(self.seq2seq_path / 'code_summary_seq2seq_model.h5'))
        num_encoder_tokens, enc_pp = load_text_processor(self.seq2seq_path / 'py_code_proc_v2.dpkl')
        num_decoder_tokens, dec_pp = load_text_processor(self.seq2seq_path / 'py_comment_proc_v2.dpkl')
        seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                        decoder_preprocessor=dec_pp,
                                        seq2seq_model=seq2seq_Model)
        with open(self.data_path/'without_docstrings.function', 'r', encoding='utf-8') as f:
            no_docstring_funcs = f.readlines()
        with open(self.data_path / 'without_docstrings.paraids', 'r', encoding='utf-8') as f:
            no_docstring_paraids = f.readlines()
        print("size of paraids = ", len(no_docstring_paraids))
        demo_testdf = pd.DataFrame({'code': no_docstring_funcs, 'comment': '', 'ref': ''})
        auto_tag = seq2seq_inf.demo_model_predictions(n=15, df=demo_testdf)
        print("size of auto_tag = ", len(auto_tag))

        with open(self.data_path/'without_docstrings.autotag', 'w', encoding='utf-8') as f:
            index = 0
            for item in auto_tag:
                f.write("%s\n" % item)
                paraid = no_docstring_paraids[index]
                paraid = paraid.strip()
                updated_rows = postgres.update_autotag(paraid, item)
                index=index+1


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

    def create_searchindex(self):
        nodoc_vecs = np.load(self.code2emb_path / 'nodoc_vecs.npy')
        assert nodoc_vecs.shape[0] == self.ref_df.shape[0]
        search_index = create_nmslib_search_index(nodoc_vecs)
        search_index.saveIndex('search_index.nmslib')
        print("SearchIndex is created")

    def search_engine(self):
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        lang_model = torch.load(Path(self.base_dir + './data/lang_model/lang_model_cpu_v2.torch'),map_location='cpu')
        # lang_model = torch.load('lang_model_cpu_v2.torch', map_location='cpu')
        vocab = load_lm_vocab(Path(self.base_dir + './data/lang_model/vocab_v2.cls'))
        embpath = Path(self.base_dir + './data/search/embeddings.var')
        q2emb = Query2Emb(lang_model=lang_model,vocab=vocab)
        return q2emb

    def search_query(self, postgres, q2emb, str_search, k=2):
        query2emb_func = q2emb.emb_mean
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

    def unexpected_error(self):
        status_code = 500
        success = False
        response = {
            'success': success,
            'error': {
                'type': 'UnexpectedException',
                'message': 'An unexpected error has occurred.'
            }
        }

        return jsonify(response), status_code