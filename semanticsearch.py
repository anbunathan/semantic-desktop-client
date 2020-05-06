from keras.models import load_model
from pathlib import Path
import numpy as np
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

class semantic:
    base_dir = ''
    code2emb_path = Path(base_dir+'./data/code2emb/')
    seq2seq_path = Path(base_dir+'./data/seq2seq/')
    data_path = Path(base_dir+'./data/processed_data/')
    output_path = Path(base_dir + './data/search')
    input_path = Path(base_dir +'./data/processed_data/')
    ref_df = None

    def create_vector(self, paragraphs, filepaths):
        code2emb_model = load_model(str(self.code2emb_path/'code2emb_model.hdf5'), custom_objects=None, compile=False)
        num_encoder_tokens, enc_pp = load_text_processor(self.seq2seq_path/'py_code_proc_v2.dpkl')
        with open(self.data_path/'without_docstrings.function', 'w', encoding='utf-8') as f:
            for item in paragraphs:
                f.write("%s" % item)
        with open(self.data_path/'without_docstrings.lineage', 'w', encoding='utf-8') as f:
            for item in filepaths:
                f.write("%s\n" % item)
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

    def create_autotag(self):
        seq2seq_Model = load_model(str(self.seq2seq_path / 'code_summary_seq2seq_model.h5'))
        num_encoder_tokens, enc_pp = load_text_processor(self.seq2seq_path / 'py_code_proc_v2.dpkl')
        num_decoder_tokens, dec_pp = load_text_processor(self.seq2seq_path / 'py_comment_proc_v2.dpkl')
        seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                        decoder_preprocessor=dec_pp,
                                        seq2seq_model=seq2seq_Model)
        with open(self.data_path/'without_docstrings.function', 'r', encoding='utf-8') as f:
            no_docstring_funcs = f.readlines()
        demo_testdf = pd.DataFrame({'code': no_docstring_funcs, 'comment': '', 'ref': ''})
        auto_tag = seq2seq_inf.demo_model_predictions(n=15, df=demo_testdf)
        return auto_tag

    def create_refdf(self):
        # read file of urls
        url_df = pd.read_csv(self.input_path/'without_docstrings.lineage', header=None, names=['url'])
        code_df = pd.read_csv(open(self.input_path/'without_docstrings.function','rU',encoding='utf-8'), sep='delimiter',quoting=csv.QUOTE_NONE, engine='python', header=None, names=['code'])
        # make sure these files have same number of rows
        print("code_df.shape[0] = ",code_df.shape[0])
        print("url_df.shape[0] = ",url_df.shape[0])
        assert code_df.shape[0] == url_df.shape[0]
        # collect these two together into a dataframe
        ref_df = pd.concat([url_df, code_df], axis=1).reset_index(drop=True)
        self.ref_df = ref_df
        print(ref_df.head())

    def create_searchindex(self):
        nodoc_vecs = np.load(self.code2emb_path / 'nodoc_vecs.npy')
        assert nodoc_vecs.shape[0] == self.ref_df.shape[0]
        search_index = create_nmslib_search_index(nodoc_vecs)
        search_index.saveIndex('search_index.nmslib')
        print("SearchIndex is created")

    def search_engine(self, str_search, k=2):
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        lang_model = torch.load(Path(self.base_dir + './data/lang_model/lang_model_cpu_v2.torch'),map_location='cpu')
        # lang_model = torch.load('lang_model_cpu_v2.torch', map_location='cpu')
        vocab = load_lm_vocab(Path(self.base_dir + './data/lang_model/vocab_v2.cls'))
        q2emb = Query2Emb(lang_model=lang_model,vocab=vocab)
        search_index = nmslib.init(method='hnsw', space='cosinesimil')
        search_index.loadIndex('search_index.nmslib')

        nmslib_index = search_index
        ref_df = self.ref_df
        query2emb_func = q2emb.emb_mean

        query = query2emb_func(str_search)
        idxs, dists = search_index.knnQuery(query, k=k)

        for idx, dist in zip(idxs, dists):
            code = self.ref_df.iloc[idx].code
            url = self.ref_df.iloc[idx].url
            print(f'cosine dist:{dist:.4f}  Location: {url}\n---------------\n')
            print(code)

