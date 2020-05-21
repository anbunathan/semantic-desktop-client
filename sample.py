import os
import glob
from os import path
from flask import jsonify
from keras.models import load_model
from pathlib import Path
import numpy as np
from postgreshandler import *
from general_utils import create_nmslib_search_index

def create_searchindex(postgres):
    base_dir = ''
    code2emb_path = Path(base_dir + './data/code2emb/')
    seq2seq_path = Path(base_dir + './data/seq2seq/')
    data_path = Path(base_dir + './data/processed_data/')
    output_path = Path(base_dir + './data/search')
    input_path = Path(base_dir + './data/processed_data/')
    npy_path = Path(base_dir + './data/npy/')

    with open(code2emb_path / 'nodoc_vecs.npy', 'wb') as f_handle:
        # os.chdir(npyfilespath)
        first = False
        dataArray = None
        # os.chdir(npyfilespath)
        for npfile in glob.glob("data/npy/*"):

            # Find the path of the file
            filepath = os.path.join("", npfile)
            print("filepath = ", filepath)
            temp = npfile.split('####')
            if ("\\" in temp):
                temp = temp[0].split("\\")
            elif ("/" in temp):
                temp = temp[0].split("/")
            print("temp=", temp)
            fileid = temp[-1]
            print("fileid = ", fileid)
            matching_rows = postgres.check_fileid_exists(fileid)
            if (matching_rows>0):
                if first == False:
                    # Load file
                    dataArray = np.load(filepath)
                    first = True
                    print("this is first file")
                else:
                    dataArray = np.concatenate((dataArray, np.load(filepath)), axis=0)
                    print("this is not first file")
        np.save(f_handle, dataArray)
    nodoc_vecs = np.load(code2emb_path / 'nodoc_vecs.npy')
    print("nodoc_vecs.shape[0] = ", nodoc_vecs.shape[0])
    # assert nodoc_vecs.shape[0] == self.ref_df.shape[0]
    search_index = create_nmslib_search_index(nodoc_vecs)
    search_index.saveIndex('search_index.nmslib')
    print("SearchIndex is created")

postgres = postgressql()
create_searchindex(postgres)

# fpath = self.code2emb_path / 'nodoc_vecs.npy'
# if os.path.exists(fpath):
#     os.remove(fpath)
# else:
#     print("The file does not exist")