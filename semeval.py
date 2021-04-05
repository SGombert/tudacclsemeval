from operator import mul
from catboost import CatBoostRegressor, Pool
from flair.embeddings.token import WordEmbeddings
import numpy as np
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings, SentenceTransformerDocumentEmbeddings
from flair.data import Sentence
from tqdm import tqdm
import joblib as jb

import os

import sklearn.metrics as me
import scipy.stats as sps


from big_phoney import BigPhoney

import pandas as pd
import json

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.linear_model import LinearRegression

import stanza

import tensorflow as tf


from nltk.corpus import wordnet as wn
import nltk

import torch

tf.config.set_visible_devices([], 'GPU')

def read(path):
    lines = []
    seenfirst = False
    with open(path, 'r', encoding = 'utf8') as f:
        for l in f:
            if not seenfirst:
                seenfirst = True
                continue

            lines.append(l.split('\t'))
    return lines

def copy_to_vec(index, input_vec, copy_vec):
    for i in range(len(copy_vec)):
        input_vec[index + i] = copy_vec[i]
    return index + len(copy_vec)

def get_sylllable_cats(corpus, label_binarizer=None):
    phoney = BigPhoney()

    ret = [[] for _ in range(len(corpus))]
    syl_count = [0 for _ in range(len(corpus))]

    for i in tqdm(range(len(corpus))):
        ret[i] = phoney.phonize(corpus[i][3]).split(' ')
        syl_count[i] = len(ret[i])

    transformed = None
    if label_binarizer != None:
        transformed = label_binarizer.transform(ret)
    else:
        label_binarizer = MultiLabelBinarizer()

        transformed = label_binarizer.fit_transform(ret)

    return pd.concat([pd.DataFrame(transformed), pd.DataFrame(syl_count)], axis=1), label_binarizer
   
def get_tagger_cats(corpus, label_binarizers=None):
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse,ner')
    ret_xpos = ['N/A'] * len(corpus)
    ret_xpost_left_1 = ['N/A'] * len(corpus)
    ret_xpost_left_2 = ['N/A'] * len(corpus)
    ret_xpost_right_1 = ['N/A'] * len(corpus)
    ret_xpost_right_2 = ['N/A'] * len(corpus)
    ret_upos = ['N/A'] * len(corpus)
    ret_upost_left_1 = ['N/A'] * len(corpus)
    ret_upost_left_2 = ['N/A'] * len(corpus)
    ret_upost_right_1 = ['N/A'] * len(corpus)
    ret_upost_right_2 = ['N/A'] * len(corpus)
    ret_dep = ['N/A'] * len(corpus)
    ret_dep_siblings = [[] for _ in range(len(corpus))]
    ret_dep_children = [[] for _ in range(len(corpus))]
    ret_dep_parent = ['N/A'] * len(corpus)
    ret_feats = [[] for _ in range(len(corpus))]
    ret_ner = ['O'] * len(corpus)
    
    for i in tqdm(range(len(corpus))):
        doc = nlp(corpus[i][2])
        sents = doc.to_dict()
        for dic in sents:
            found = False
            for token in dic:
                if token['text'].lower() == corpus[i][3].strip().lower():
                    ret_xpos[i] = token['xpos']
                    ret_xpost_left_1[i] = dic[token['id'] - 1]['xpos'] if token['id'] > 0 else 'N/A'
                    ret_xpost_left_2[i] = dic[token['id'] - 2]['xpos'] if token['id'] > 1 else 'N/A'
                    ret_xpost_right_1[i] = dic[token['id'] - 1]['xpos'] if token['id'] < len(dic) else 'N/A'
                    ret_xpost_right_2[i] = dic[token['id'] - 2]['xpos'] if token['id'] < len(dic) - 1 else 'N/A'
                    ret_upos[i] = token['upos']
                    ret_upost_left_1[i] = dic[token['id'] - 1]['upos'] if token['id'] > 0 else 'N/A'
                    ret_upost_left_2[i] = dic[token['id'] - 2]['upos'] if token['id'] > 1 else 'N/A'
                    ret_upost_right_1[i] = dic[token['id'] - 1]['upos'] if token['id'] < len(dic) else 'N/A'
                    ret_upost_right_2[i] = dic[token['id'] - 2]['upos'] if token['id'] < len(dic) - 1 else 'N/A'
                    ret_dep[i] = token['deprel']
                    ret_dep_siblings[i] = [t['deprel'] for t in dic if ('head' in token and 'head' in t and t['head'] == token['head'])]
                    ret_dep_children[i] = [t['deprel'] for t in dic if ('head' in t and t['head'] == token['id'])]
                    ret_dep_parent[i] = dic[token['head']]['deprel'] if ('head' in token and token['head'] > 0 and token['head'] < len(dic)) else 'N/A'
                    ret_feats[i] = token['feats'].split('|') if 'feats' in 'feats' in token else 'N/A'
                    ret_ner[i] = token['ner']
                    found = True
                    break
            
            if found:
                break

    del nlp

    xpos_binarizer = LabelBinarizer()
    xpost_left_1_binarizer = LabelBinarizer()
    xpost_left_2_binarizer = LabelBinarizer()
    xpost_right_1_binarizer = LabelBinarizer()
    xpost_right_2_binarizer = LabelBinarizer()
    upos_binarizer = LabelBinarizer()
    upost_left_1_binarizer = LabelBinarizer()
    upost_left_2_binarizer = LabelBinarizer()
    upost_right_1_binarizer = LabelBinarizer()
    upost_right_2_binarizer = LabelBinarizer()
    dep_binarizer = LabelBinarizer()
    dep_siblings_binarizer = MultiLabelBinarizer()
    dep_parent_binarizer = MultiLabelBinarizer()
    dep_children_binarizer = MultiLabelBinarizer()
    feats_binarizer = MultiLabelBinarizer()
    ner_binarizer = LabelBinarizer()

    xpos_transformed = None
    xpost_left_1_transformed = None
    xpost_left_2_transformed = None
    xpost_right_1_transformed = None
    xpost_right_2_transformed = None
    upos_transformed = None
    upost_left_1_transformed = None
    upost_left_2_transformed = None
    upost_right_1_transformed = None
    upost_right_2_transformed = None
    dep_transformed = None
    dep_siblings_transformed = None
    dep_children_transformed = None
    dep_parent_transformed = None
    feats_transformed = None
    ner_transformed = None

    if label_binarizers != None:
        xpos_binarizer = label_binarizers[0]
        xpost_left_1_binarizer = label_binarizers[1]
        xpost_left_2_binarizer = label_binarizers[2]
        xpost_right_1_binarizer = label_binarizers[3]
        xpost_right_2_binarizer = label_binarizers[4]
        upos_binarizer = label_binarizers[5]
        upost_left_1_binarizer = label_binarizers[6]
        upost_left_2_binarizer = label_binarizers[7]
        upost_right_1_binarizer = label_binarizers[8]
        upost_right_2_binarizer = label_binarizers[9]
        dep_siblings_binarizer = label_binarizers[10]
        dep_children_binarizer = label_binarizers[11]
        dep_binarizer = label_binarizers[12]
        dep_parent_binarizer = label_binarizers[13]
        feats_binarizer = label_binarizers[14]
        ner_binarizer = label_binarizers[15]

        xpos_transformed = xpos_binarizer.transform(ret_xpos)
        xpost_left_1_transformed = xpost_left_1_binarizer.transform(ret_xpost_left_1)
        xpost_left_2_transformed = xpost_left_2_binarizer.transform(ret_xpost_left_2)
        xpost_right_1_transformed = xpost_right_1_binarizer.transform(ret_xpost_right_1)
        xpost_right_2_transformed = xpost_right_2_binarizer.transform(ret_xpost_right_2)
        upos_transformed = upos_binarizer.transform(ret_upos)
        upost_left_1_transformed = upost_left_1_binarizer.transform(ret_upost_left_1)
        upost_left_2_transformed = upost_left_2_binarizer.transform(ret_upost_left_2)
        upost_right_1_transformed = upost_right_1_binarizer.transform(ret_upost_right_1)
        upost_right_2_transformed = upost_right_2_binarizer.transform(ret_upost_right_2)
        dep_transformed = dep_binarizer.transform(ret_dep)
        dep_parent_transformed = dep_parent_binarizer.transform(ret_dep_parent)
        dep_siblings_transformed = dep_siblings_binarizer.transform(ret_dep_siblings)
        dep_children_transformed = dep_children_binarizer.transform(ret_dep_children)
        feats_transformed = feats_binarizer.transform(ret_feats)
        ner_transformed = ner_binarizer.transform(ret_ner)
    else:
        xpos_transformed = xpos_binarizer.fit_transform(ret_xpos)
        xpost_left_1_transformed = xpost_left_1_binarizer.fit_transform(ret_xpost_left_1)
        xpost_left_2_transformed = xpost_left_2_binarizer.fit_transform(ret_xpost_left_2)
        xpost_right_1_transformed = xpost_right_1_binarizer.fit_transform(ret_xpost_right_1)
        xpost_right_2_transformed = xpost_right_2_binarizer.fit_transform(ret_xpost_right_2)
        upos_transformed = upos_binarizer.fit_transform(ret_upos)
        upost_left_1_transformed = upost_left_1_binarizer.fit_transform(ret_upost_left_1)
        upost_left_2_transformed = upost_left_2_binarizer.fit_transform(ret_upost_left_2)
        upost_right_1_transformed = upost_right_1_binarizer.fit_transform(ret_upost_right_1)
        upost_right_2_transformed = upost_right_2_binarizer.fit_transform(ret_upost_right_2)
        dep_transformed = dep_binarizer.fit_transform(ret_dep)
        dep_parent_transformed = dep_parent_binarizer.fit_transform(ret_dep_parent)
        dep_siblings_transformed = dep_siblings_binarizer.fit_transform(ret_dep_siblings)
        dep_children_transformed = dep_children_binarizer.fit_transform(ret_dep_children)
        feats_transformed = feats_binarizer.fit_transform(ret_feats)
        ner_transformed = ner_binarizer.fit_transform(ret_ner)
    
    xpos_df = pd.DataFrame(xpos_transformed)
    xpos_left_1_df = pd.DataFrame(xpost_left_1_transformed)
    xpos_left_2_df = pd.DataFrame(xpost_left_2_transformed)
    xpos_right_1_df = pd.DataFrame(xpost_right_1_transformed)
    xpos_right_2_df = pd.DataFrame(xpost_right_2_transformed)
    upos_df = pd.DataFrame(upos_transformed)
    upos_left_1_df = pd.DataFrame(upost_left_1_transformed)
    upos_left_2_df = pd.DataFrame(upost_left_2_transformed)
    upos_right_1_df = pd.DataFrame(upost_right_1_transformed)
    upos_right_2_df = pd.DataFrame(upost_right_2_transformed)
    dep_df = pd.DataFrame(dep_transformed)
    dep_parent_df = pd.DataFrame(dep_parent_transformed)
    dep_children_df = pd.DataFrame(dep_children_transformed)
    dep_siblings_df = pd.DataFrame(dep_siblings_transformed)
    feats_df = pd.DataFrame(feats_transformed)
    ner_df = pd.DataFrame(ner_transformed)

    ret = pd.concat([xpos_df, xpos_left_1_df, xpos_left_2_df, xpos_right_1_df, xpos_right_2_df, upos_left_1_df, upos_left_2_df, upos_right_1_df, upos_right_2_df, dep_children_df, dep_siblings_df, upos_df, dep_df, dep_parent_df, feats_df, ner_df], axis=1)

    return ret, (xpos_binarizer, xpost_left_1_binarizer, xpost_left_2_binarizer, xpost_right_1_binarizer, xpost_right_2_binarizer, upos_binarizer, upost_left_1_binarizer, upost_left_2_binarizer, upost_right_1_binarizer, upost_right_2_binarizer, dep_siblings_binarizer, dep_children_binarizer, dep_binarizer, dep_parent_binarizer, feats_binarizer, ner_binarizer)

def get_wordnet_cats(corpus):
    nums_hypernyms = [0.0] * len(corpus)
    nums_hyponyms = [0.0] * len(corpus)
    nums_member_holonyms = [0.0] * len(corpus)
    nums_part_meronyms = [0.0] * len(corpus)
    nums_member_meronyms = [0.0] * len(corpus)
    hypernym_paths = [0.0] * len(corpus)
    nums_examples = [0.0] * len(corpus)
    nums_root_hypernyms = [0.0] * len(corpus)

    for i in tqdm(range(len(corpus))):
        syns = wn.synsets(corpus[i][3],lang='eng')
        jj = 0
        for syn in syns:
            nums_hypernyms[i] += len(syn.hypernyms())
            nums_hyponyms[i] += len(syn.hyponyms())
            nums_member_holonyms[i] += len(syn.member_holonyms())
            nums_part_meronyms[i] += len(syn.part_meronyms())
            nums_member_meronyms[i] += len(syn.member_meronyms())
            hypernym_paths[i] += min([len(path) for path in syn.hypernym_paths()])
            nums_examples[i] += len(syn.examples())
            nums_root_hypernyms[i] += len(syn.root_hypernyms())

            jj += 1.0

        if jj == 0:
            jj = 1.0

        nums_hypernyms[i] /= jj
        nums_hyponyms[i] /= jj
        nums_member_holonyms[i] /= jj
        nums_part_meronyms[i] /= jj
        nums_member_meronyms[i] /= jj
        hypernym_paths[i] /= jj
        nums_examples[i] /= jj
        nums_root_hypernyms[i] /= jj

    return [
        pd.Series(nums_hypernyms, dtype="float"),
        pd.Series(nums_hypernyms, dtype="float"),
        pd.Series(nums_member_holonyms, dtype="float"),
        pd.Series(nums_part_meronyms, dtype="float"),
        pd.Series(nums_member_meronyms, dtype="float"),
        pd.Series(hypernym_paths, dtype="float"),
        pd.Series(nums_examples, dtype="float"),
        pd.Series(nums_root_hypernyms, dtype="float")
    ]

def get_embedding_cats(corpus, embeddings):
    dims = [np.zeros(len(corpus)) for _ in range(embeddings.embedding_length)]

    for i in tqdm(range(len(corpus))):
        sentence = Sentence(corpus[i][2])
        embeddings.embed(sentence)
        for token in sentence:
            if token.text.strip().lower() == corpus[i][3].strip().lower():
                embeddingVec = token.embedding.cpu().detach().numpy()
                for dim in range(len(embeddingVec)):
                    dims[dim][i] = embeddingVec[dim]

                break
    
    return [pd.Series(k) for k in dims]

def get_subtlex_cats(corpus, subtlex):
    dims = [np.zeros(len(corpus)) * len(corpus), [0.0] * len(corpus)]

    for i in tqdm(range(len(corpus))):
        w = corpus[i][3].lower()
        if w not in subtlex:
            w = w[:-1]
        if w not in subtlex:
            w = w[:-1]
        if w not in subtlex:
            w = w[:-1]
        if w in subtlex:
            dims[0][i] = float(subtlex[w][5])
            dims[1][i] = float(subtlex[w][7])

    return [
        pd.Series(dims[0]),
        pd.Series(dims[1])
    ]

def get_emoti_cats(corpus, emoti):
    dims = [
        [-1.0] * len(corpus), 
        [1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [-1.0] * len(corpus), 
        [-1.0] * len(corpus)
    ]

    for i in tqdm(range(len(corpus))):
        w = corpus[i][3].lower()
        if w not in emoti:
            w = w[:-1]
        if w not in emoti:
            w = w[:-1]
        if w not in emoti:
            w = w[:-1]
        if w not in emoti:
            continue
        for e in range(13):
            dims[e][i] = float(emoti[w][e])
    
    return [
        pd.Series(dims[e]) for e in range(len(dims))
    ]

def get_affec_cats(corpus, affec):
    dims = [np.zeros(len(corpus)) for i in range(63)]

    for i in tqdm(range(len(corpus))):
        w = corpus[i][3].lower()
        if w not in affec:
            w = w[:-1]
        if w not in affec:
            w = w[:-1]
        if w not in affec:
            w = w[:-1]
        if w not in affec:
            continue

        for e in range(63):
            dims[e][i] = float(affec[w][e])

    return [
        pd.Series(dims[e]) for e in range(len(dims))
    ]

def get_bigrams_cats(corpus, bigrams):
    dims = np.zeros(len(corpus))

    for i in tqdm(range(len(corpus))):
        av_bigrams_freq = 0.0
        word_bigrams = list(nltk.bigrams(corpus[i][3].lower()))
        cnnt = 0
        for bi in word_bigrams:
            av_bigrams_freq += bigrams[(bi[0] + bi[1])] if (bi[0] + bi[1]) in bigrams else 0
            cnnt += 1

        if cnnt == 0:
            cnnt = 1

        dims[i] = av_bigrams_freq / cnnt
    
    return [
        pd.Series(dims)
    ]

def get_odgens_cats(corpus, odgens):
    dims = np.zeros(len(corpus))

    for i in tqdm(range(len(corpus))):
        w = corpus[i][3].lower()
        if w not in odgens:
            w = w[:-1]
        if w not in odgens:
            w = w[:-1]
        if w not in odgens:
            w = w[:-1]
        if w not in odgens:
            continue

        dims[i] = 1.0

    return [
        pd.Series(dims)
    ]

def get_awl_cats(corpus, awl):
    dims = np.zeros(len(corpus))

    for i in tqdm(range(len(corpus))):
        w = corpus[i][3].lower()
        if w not in awl:
            continue

        dims[i] = 1.0

    return [
        pd.Series(dims)
    ]

def get_efllex_cats(corpus, efllex):
    dims = [np.zeros(len(corpus)) for _ in range(111)]

    for i in tqdm(range(len(corpus))):
        w = corpus[i][3].lower()
        if w not in efllex:
            w = w[:-1]
        if w not in efllex:
            w = w[:-1]
        if w not in efllex:
            w = w[:-1]
        if w not in efllex:
            continue
        for e in range(111):
            dims[e][i] = float(efllex[w][e])

    return [
        pd.Series(dims[e]) for e in range(len(dims))
    ]

def create_data_frame(corpus, embeddings, subtlex, emoti, affec, bigrams, odgens, awl, efllex, binarizers, syllable_binarizers):
    tags_df, binarizers_ret = get_tagger_cats(corpus, binarizers)
    #syllables_df, syllable_binarizers = get_sylllable_cats(corpus, syllable_binarizers)
    
    lst = get_wordnet_cats(corpus) + get_embedding_cats(corpus, embeddings) + get_subtlex_cats(corpus, subtlex) + get_emoti_cats(corpus, emoti) + get_affec_cats(corpus, affec) + get_bigrams_cats(corpus, bigrams) + get_efllex_cats(corpus, efllex) + get_awl_cats(corpus, awl) + get_odgens_cats(corpus, odgens)

    dic = {
    }
    for i, sr in enumerate(lst):
        dic[str(i)] = sr

    ret = pd.concat([pd.DataFrame.from_dict(dic), tags_df], axis=1)

    ret.columns = ["F" + str(i + 1) for i in range(ret.shape[1])]
    
    return ret, binarizers_ret, syllable_binarizers

def create_target_series(corpus):
    return pd.Series([float(j[4]) for j in corpus])

def encode_single(corpus, embeddings, subtlex, emoti, affec, bigrams, odgens, awl, efllex, add_embeddings, multitagger):
    x = [[0.0] * 7004] * len(corpus)
    y = [0.0] * len(corpus)
    print(len(x))

    embs = StackedEmbeddings([embeddings, add_embeddings])

    for i in tqdm(range(len(corpus))):
        #Wordnet
        syns = wn.synsets(corpus[i][3],lang='eng')
        num_hypernyms = 0.0
        num_hyponyms = 0.0
        num_member_holonyms = 0.0
        num_part_meronyms = 0.0
        num_member_meronyms = 0.0
        hypernym_paths = 0.0
        num_examples = 0.0
        num_root_hypernyms = 0.0

        j = 0
        for syn in syns:
            num_hypernyms += len(syn.hypernyms())
            num_hyponyms += len(syn.hyponyms())
            num_member_holonyms += len(syn.member_holonyms())
            num_part_meronyms += len(syn.part_meronyms())
            num_member_meronyms += len(syn.member_meronyms())
            hypernym_paths += min([len(path) for path in syn.hypernym_paths()])
            num_examples += len(syn.examples())
            num_root_hypernyms += len(syn.root_hypernyms())

            j += 1

        if j == 0:
            j = 1

        wordnetVec = [num_hypernyms / j, num_hyponyms / j, num_member_holonyms / j, num_part_meronyms / j, num_member_meronyms / j, len(syns), hypernym_paths / j, num_examples / j, num_root_hypernyms / j] 
        index = copy_to_vec(0, x[i], wordnetVec)

        #WordEmbeddings
        sentence = Sentence(corpus[i][2])
        embs.embed(sentence)
        tag_sentence = Sentence(corpus[i][2])
        multitagger.predict(tag_sentence)

        embeddingVec = [0.0] * embs.embedding_length
        ner_tag = ['N/A']
        pos_tag = ['N/A']
        for token in sentence:
            if token.text.strip() == corpus[i][3]:
                embeddingVec = token.embedding.cpu().detach().tolist()

        for token in tag_sentence:
            if token.text.strip() == corpus[i][3]:
                pos_tag = [token.get_tag('pos-fast').value]
                ner_tag = [token.get_tag('ner-fast').value]

        subtlexVec = None
        if corpus[i][3].lower() in subtlex:
            subtlexVec = [float(subtlex[corpus[i][3].lower()][5]), float(subtlex[corpus[i][3].lower()][7])]
        else:
            subtlexVec = [0.0, 0.0]

        emotiVec = None
        if corpus[i][3].lower() in emoti:
            emotiVec = [float(emoti[corpus[i][3].lower()][e]) for e in range(13)]
        else:
            emotiVec = [-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

        affecVec = None
        if corpus[i][3].lower() in affec:
            affecVec = [float(affec[corpus[i][3].lower()][e]) for e in range(63)]
        else:
            affecVec = [0.0] * 63

        eflVec = None
        if corpus[i][3].lower() in efllex:
            eflVec = [float(efllex[corpus[i][3].lower()][e]) for e in range(111)]
        else:
            eflVec = [0.0] * 111

        av_bigrams_freq = 0.0
        word_bigrams = list(nltk.bigrams(corpus[i][3].lower()))
        cnnt = 0
        for bi in word_bigrams:
            av_bigrams_freq += bigrams[(bi[0] + bi[1])]
            cnnt += 1

        if cnnt == 0:
            cnnt = 1
        
        bigram_vec = [av_bigrams_freq / cnnt]

        wordLength = [len(corpus[i][3])]

        odg = 0.0
        if corpus[i][3].lower() in odgens:
            odg = 1.0

        odgensVec = [odg]

        aw = 0.0
        if corpus[i][3].lower() in awl:
            aw = 1.0

        awlVec = [aw]

        index = copy_to_vec(0, x[i], pos_tag)
        index = copy_to_vec(index, x[i], ner_tag)
        index = copy_to_vec(index, x[i], embeddingVec)
        index = copy_to_vec(index, x[i], wordnetVec)
        index = copy_to_vec(index, x[i], subtlexVec)
        index = copy_to_vec(index, x[i], emotiVec)
        index = copy_to_vec(index, x[i], wordLength)
        index = copy_to_vec(index, x[i], affecVec)
        index = copy_to_vec(index, x[i], bigram_vec)
        index = copy_to_vec(index, x[i], odgensVec)
        index = copy_to_vec(index, x[i], awlVec)
        index = copy_to_vec(index, x[i], eflVec)

        #x[i] = np.concatenate((pos_tag, ner_tag, embeddingVec, wordnetVec, subtlexVec, emotiVec, wordLength, affecVec, bigram_vec, odgensVec, awlVec, eflVec))

        y[i] = float(corpus[i][4])

    return x, y

def feature_indices():
    idx = {
        
    }

def encode_multi(corpus, embeddings):
    x = np.zeros(shape=(len(corpus),embeddings.embedding_length), dtype=np.float)
    y = np.zeros(shape=len(corpus))

    for i in tqdm(range(len(corpus))):
        sentence = Sentence(corpus[i][2])
        embeddings.embed(sentence)
        words = corpus[i][3]
        for token in sentence:
            if token.text == words[0]:
                x[i] = token.embedding.cpu().numpy()
            if token.text == words[1]:
                x[i] = np.concatenate((x[i], token.embedding.numpy()))
        y[i] = float(corpus[i][4])

    return Pool(data=x, label=y)

import csv

def get_subtlex():
    r = csv.reader(open('subtlex.txt'), delimiter='\t')
    next(r, None)
    ret = {}
    for row in r:
        ret[row[0].lower()] = row[1:]
    
    return ret

def get_emotiword():
    r = csv.reader(open('emotiword.tsv'), delimiter='\t')
    next(r, None)
    ret = {}
    for row in r:
        ret[row[0].lower()] = row[1:]

    return ret

def get_affective():
    r = csv.reader(open('ratings_warriner.csv'), delimiter=',')
    next(r, None)
    ret = {}
    for row in r:
        ret[row[1].lower()] = row[2:]
    
    return ret

def get_efllex():
    r = csv.reader(open('EFLLex'), delimiter='\t')
    next(r, None)
    ret = {}
    for row in r:
        ret[row[0].lower()] = row[2:]
    
    return ret

def get_bigrams():
    r = json.loads(open('bigrams.json', 'r').read())
    ret = {}
    for obs in r:
        ret[obs[0]] = obs[1]

    return ret

def get_odgens():
    ret = set()
    for line in open('Odgens.txt', 'r'):
        for elem in line.strip().split(' , '):
            ret.add(elem.lower())
    
    return ret

def get_awl():
    ret = set()
    for line in open('AWL.txt', 'r'):
        for elem in line.strip().split(', '):
            ret.add(elem.lower())

    return ret

def get_embeddings():
    stacked_embeddings = StackedEmbeddings([
                            TransformerWordEmbeddings('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',           use_scalar_mix=True, layers="all"),
                            TransformerWordEmbeddings('distilgpt2',                                                     use_scalar_mix=True, layers="all"),
                            TransformerWordEmbeddings('distilbert-base-uncased',                                        use_scalar_mix=True, layers="all"),
                            FlairEmbeddings('mix-forward'),
                            WordEmbeddings('glove')
                        ])
    return stacked_embeddings

def add_embeddings():
    stacked_embeddings = StackedEmbeddings([
        FlairEmbeddings('mix-backward'),
        WordEmbeddings('en')
    ])

    return stacked_embeddings

def get_sentence_transformer():
    return SentenceTransformerDocumentEmbeddings('stsb-distilbert-base')

def read_out(path):
    out_file = {}
    for line in open(path, 'r'):
        spl = line.split(',')
        out_file[spl[0]] = float(spl[1])
    
    return out_file

def write_out(path, out_file):
    sr = ''
    for key in out_file:
        sr += key + ',' + str(out_file[key]) + '\n'
    
    with open(path, 'w') as f:
        f.write(sr)

def to_out(corpus, out_file, y_pred):
    for i in range(len(corpus)):
        out_file[corpus[i][0]] = y_pred[i]

print('Loading embeddings')
embeddings = get_embeddings()
add_embedding = add_embeddings()
embeddings.load_state_dict(torch.load('embs.pt'))
print('Loading training data')
path_single_train = 'C:/semeval/train/lcp_single_train.tsv'
path_single_trial = 'C:/semeval/trial/lcp_single_trial.tsv'
path_single_test = 'C:/semeval/test/lcp_single_test.tsv'

ems = StackedEmbeddings([
    embeddings,
    add_embedding
])

#import json
#corpus = read(path_single_train)
#x, binarizers_o, syllable_binarizers_o = create_data_frame(corpus, ems, get_subtlex(), get_emotiword(), get_affective(), get_bigrams(), get_odgens(), get_awl(), get_efllex(), binarizers=None, syllable_binarizers=None)
#y = pd.DataFrame.from_dict({ 'y': create_target_series(corpus)})
#x.to_feather('train_x.feather')
#y.to_feather('train_y.feather')

#corpus = read(path_single_trial)
#x_e, binarizers_o, syllable_binarizers_o = create_data_frame(corpus, ems, get_subtlex(), get_emotiword(), get_affective(), get_bigrams(), get_odgens(), get_awl(), get_efllex(), binarizers=binarizers_o, syllable_binarizers=syllable_binarizers_o)
#y_e = pd.DataFrame.from_dict({ 'y': create_target_series(corpus)})
#x_e.to_feather('eval_x.feather')
#y_e.to_feather('eval_y.feather')

#corpus = read(path_single_test)
#x_t, binarizers_o, syllable_binarizers_o = create_data_frame(corpus, ems, get_subtlex(), get_emotiword(), get_affective(), get_bigrams(), get_odgens(), get_awl(), get_efllex(), binarizers=binarizers_o, syllable_binarizers=syllable_binarizers_o)
#x_t.to_feather('test_x.feather')

#jb.dump(binarizers_o, 'binarizers_o.bin')
#jb.dump(syllable_binarizers_o, 'syllable_binarizers_o.bin')

x = pd.read_feather('train_x.feather')
y = pd.read_feather('train_y.feather')
x_e = pd.read_feather('eval_x.feather')
y_e = pd.read_feather('eval_y.feather')
x_t = pd.read_feather('test_x.feather')

#print(x)

pool = Pool(data=x, label=y)
pool_e = Pool(data=x_e, label=y_e)
pool_t = Pool(data=x_t)
pool.quantize()
pool.save_quantization_borders("borders.dat")
pool_e.quantize(input_borders="borders.dat")
pool_t.quantize(input_borders="borders.dat")

#grid = {'depth': [4, 5, 6, 7, 8, 9, 10],
#        'l2_leaf_reg': [1, 3, 5, 7, 9]}

catboost_model = CatBoostRegressor(verbose=True, eval_metric='R2', grow_policy='Lossguide', l2_leaf_reg=15, learning_rate=0.01, depth=6, iterations=1500, max_leaves=15, loss_function='RMSE')
catboost_model.fit(X=pool, eval_set=pool_e, use_best_model=True)
catboost_model.save_model('single.cbm')

y_pred = catboost_model.predict(pool_e)
print(str(me.r2_score(y_e, y_pred)))
print(str(sps.pearsonr(y_e['y'].to_numpy(), y_pred)))

y_pred = catboost_model.predict(pool_t)

importances = catboost_model.get_feature_importance(data=pool, prettified=True)
print(importances)
with open('feature_importances_single.txt', 'w') as f:
    f.write(importances.to_string())

#out_file = read_out('random_predictions_test_single.csv')
#to_out(corpus, out_file, y_pred)
#write_out('random_predictions_test_single.csv', out_file)

#os.system("shutdown /s /t 1")