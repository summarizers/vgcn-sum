# -*- coding: utf-8 -*-

# # # #
# prepare_data.py
# @author Zhibin.LU
# @created 2019-05-22T09:28:29.705Z-04:00
# @last-modified 2020-04-15T14:51:35.359Z-04:00
# @website: https://louis-udm.github.io
# @description: pre-process data and prepare the vocabulary graph
# # # #

#%%
import random
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import re
import time
import argparse
import os
import sys
from sklearn.utils import shuffle
import pandas as pd
#from pytorch_pretrained_bert import BertTokenizer

from nltk.corpus import stopwords
import nltk

random.seed(44)
np.random.seed(44)


'''
Config:
'''

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='naver')
parser.add_argument('--sw', type=int, default=0) #stopwords
args = parser.parse_args()
cfg_ds = args.ds
cfg_del_stop_words=True if args.sw==1 else False

dataset_list={'sst', 'cola','naver'}

if cfg_ds not in dataset_list:
    sys.exit("Dataset choice error!")

will_dump_objects=True
dump_dir='data/dump_data'
if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)

if cfg_del_stop_words:
    freq_min_for_word_choice=5 
    # freq_min_for_word_choice=10 #best
else:
    freq_min_for_word_choice=1 # for bert+ <<이거실행됨

valid_data_taux = 0.05 
test_data_taux = 0.10

# word co-occurence with context windows
window_size = 20
if cfg_ds in ('mr','sst','cola','naver'):
    window_size = 1000 

tfidf_mode='only_tf'  

cfg_use_bert_tokenizer_at_clean=True

#bert_model_scale='bert-base-uncased'
#bert_lower_case=True

print('---data prepare configure---')
print('Data set: ',cfg_ds,'freq_min_for_word_choice',freq_min_for_word_choice,'window_size',window_size)
print('\n')


##########################################
# CLEARNING STRING. 
##########################################
def clean_str(text):
  # Common
  # E-mail 제거
  text = re.sub("([\w\d.]+@[\w\d.]+)", "", text)
  text = re.sub("([\w\d.]+@)", "", text)
  # 괄호 안 제거
  text = re.sub("<[\w\s\d‘’=/·~:&,`]+>", "", text)
  text = re.sub("\([\w\s\d‘’=/·~:&,`]+\)", "", text)
  text = re.sub("\[[\w\s\d‘’=/·~:&,`]+\]", "", text)
  text = re.sub("【[\w\s\d‘’=/·~:&,`]+】", "", text)
  # 전화번호 제거
  text = re.sub("(\d{2,3})-(\d{3,4}-\d{4})", "", text)  # 전화번호
  text = re.sub("(\d{3,4}-\d{4})", "", text)  # 전화번호
  # 홈페이지 주소 제거
  text = re.sub("(www.\w.+)", "", text)
  text = re.sub("(.\w+.com)", "", text)
  text = re.sub("(.\w+.co.kr)", "", text)
  text = re.sub("(.\w+.go.kr)", "", text)
  # 기자 이름 제거
  text = re.sub("/\w+[=·\w@]+\w+\s[=·\w@]+", "", text)
  text = re.sub("\w{2,4}\s기자", "", text)
  # 한자 제거
  text = re.sub("[\u2E80-\u2EFF\u3400-\u4DBF\u4E00-\u9FBF\uF900]+", "", text)
  # 특수기호 제거
  text = re.sub("[◇#/▶▲◆■●△①②③★○◎▽=▷☞◀ⓒ□?㈜♠☎]", "", text)
  # 따옴표 제거
  text = re.sub("[\"'”“‘’]", "", text)
  text = text.strip()
  return text



##########################################
# (숫자 정해주기)
# train.tsv >> train, valid
# dev.tsv >> test

# (앞으로 쓰일 df 만들기)
# df = pd.concat((train_valid_df, test_df)) 
##########################################
if cfg_ds=='naver':
    label2idx = {'0':0, '1':1} #나중에 dump할때 필요
    idx2label = {0:'0', 1:'1'}
    
    train_valid_df = pd.read_csv('data/naver/ratings_train.txt', encoding='utf-8', sep='\t')
    train_valid_df.dropna(inplace=True)    
    train_valid_df['document'] = train_valid_df['document'].apply(clean_str)
    train_valid_df = train_valid_df.loc[train_valid_df['document']!='',:]
    #train_valid_df = shuffle(train_valid_df)
    
    # use dev set as test set, because we can not get the ground true label of the real test set.
    test_df = pd.read_csv('data/naver/ratings_test.txt', encoding='utf-8', sep='\t')
    test_df.dropna(inplace=True)
    test_df['document'] = test_df['document'].apply(clean_str)
    test_df = test_df.loc[test_df['document']!='',:]
    #test_df = shuffle(test_df)

    #train을 train/valid 나누기
    train_valid_size=train_valid_df['id'].count() # train dataset의 문장갯수 #1은 걍 label
    valid_size=int(train_valid_size*valid_data_taux) #train size에 0.05곱한거
    train_size=train_valid_size-valid_size #문장갯수에서 valid size빼기..? 그럼 99.5% 문장수

    #dev를 testset으로 쓰기
    test_size=test_df['id'].count() #dev dataset 갯수
    print('NAVER train_valid Total:',train_valid_size,'test Total:',test_size)
    
    #모든거 다 합친거 만들기
    df=pd.concat((train_valid_df,test_df))
    corpus = df['document'] 

    y = df['label'].values  
    y_prob=np.eye(len(y), len(label2idx))[y] #각각 문장갯수마다 2개 space만듬
    corpus_size=len(y)
    
##########################################
# CORPUS STAT(MIN/MAX/AVERAGE LENGTH) 구하기
##########################################
doc_content_list=[]
for t in corpus:
    doc_content_list.append(t) 
max_len_seq=0
max_len_seq_idx=-1
min_len_seq=1000
min_len_seq_idx=-1
sen_len_list=[]

for i,seq in enumerate(doc_content_list):
    seq=seq.split() 
    sen_len_list.append(len(seq)) #length 만 넣어주는 vector
    if len(seq)<min_len_seq:
        min_len_seq=len(seq)
        min_len_seq_idx=i #제일 짧은 문장을 track (length랑 idx만)
    if len(seq)>max_len_seq:
        max_len_seq=len(seq)
        max_len_seq_idx=i #제일 긴 문장을 track (length랑 idx만)
print('Statistics for original text: max_len%d,id%d, min_len%d,id%d, avg_len%.2f' \
    %(max_len_seq, max_len_seq_idx,min_len_seq, min_len_seq_idx,np.array(sen_len_list).mean()))



##########################################
#우리는 안쓰는 REMOVE STOPWORDS
#Remove stop words from tweets
##########################################

print('Remove stop words from tweets...')

stop_words={}

##########################################
# CLEAN THE STRING AND TOKENIZE USING BERT
##########################################
tmp_word_freq = {}  # to remove rare words
new_doc_content_list = []

from tokenization_kobert import KoBertTokenizer

#setup bert tokenizer
if cfg_use_bert_tokenizer_at_clean: #default true
    print('Use bert_tokenizer for seperate words to bert vocab')
    bert_tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

for doc_content in doc_content_list: # doc_content_list == corpus 라인마다 한거
    new_doc = doc_content #clean_str(doc_content)
    if cfg_use_bert_tokenizer_at_clean:
        sub_words = bert_tokenizer.tokenize(new_doc) #['bill', 'left', 'when', 'that', 'no', 'one', 'else', 'was', 'awake', 'is', 'certain']
        sub_doc=' '.join(sub_words).strip() #bill left when that no one else was awake is certain
        new_doc=sub_doc
    new_doc_content_list.append(new_doc) 
    
    # 걍 word counter
    for word in new_doc.split():
        if word in tmp_word_freq:
            tmp_word_freq[word] += 1
        else:
            tmp_word_freq[word] = 1 

doc_content_list=new_doc_content_list #tokenize된걸로 교체



##########################################
# GET INDEX, NUMBER OF EMPTY STRING
# empty string 바꿔주지도 않음..
##########################################
clean_docs = []
count_void_doc=0
for i,doc_content in enumerate(doc_content_list):
    words = doc_content.split() #['bill', 'left', 'when', 'that', 'no', 'one', 'else', 'was', 'awake', 'is', 'certain']
    doc_words = []
    for word in words:
        if cfg_ds in ('mr','sst','cola','naver'):
            doc_words.append(word)
        elif word not in stop_words and tmp_word_freq[word] >= freq_min_for_word_choice:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip() #bill left when that no one else was awake is certain

    #if there is a blank 문장
    if doc_str == '': 
        count_void_doc+=1
        # doc_str = '[unk]'
        # doc_str = 'normal'
        # doc_str = doc_content
        print('No.',i, 'is a empty doc after treat, replaced by \'%s\'. original:%s'%(doc_str,doc_content))
    clean_docs.append(doc_str)

print('Total',count_void_doc,' docs are empty.') #46개 나옴

##########################################
# GIVE STATS (MIN, MAX, AVERAGE) OF SENTENCE LENGTH
# AFTER TOKENIZATION.....?????
##########################################
min_len = 10000
min_len_id = -1
max_len = 0
max_len_id = -1
aver_len = 0

for i,line in enumerate(clean_docs): #clean_docs == empty 없는거...;;
    temp = line.strip().split() #['bill', 'left', 'when', 'that', 'no', 'one', 'else', 'was', 'awake', 'is', 'certain']
    aver_len = aver_len + len(temp) # keep adding ==> gives #of ALL words
    if len(temp) < min_len:
        min_len = len(temp)
        min_len_id=i #tracking len and idx of shortest sentence
    if len(temp) > max_len:
        max_len = len(temp)
        max_len_id=i  #tracking len and idx of longest sentence

aver_len = 1.0 * aver_len / len(clean_docs)
print('After tokenizer:')
print('Min_len : ' + str(min_len)+' id: '+str(min_len_id))
print('Max_len : ' + str(max_len)+' id: '+str(max_len_id))
print('Average_len : ' + str(aver_len))


##########################################
# 위에서 만든 train_size, valid_size를 갖고
# 찐 train/val 데이터 만듬

# (숫자 정한것들)
# train.tsv >> train, valid
# dev.tsv >> test
# df = pd.concat((train_valid_df, test_df))
##########################################

if cfg_ds in ('mr', 'sst','cola', 'naver'):
    shuffled_clean_docs=clean_docs #cleand_docs[0] : bill left when that no one else was awake is certain

    train_docs=shuffled_clean_docs[:train_size] # train_size == 원래 train.tsv 의 99.5%
    valid_docs=shuffled_clean_docs[train_size:train_size+valid_size] # valid_size == 원래 train.tsv 의 0.05%

    #train & valid 합쳐진거
    train_valid_docs=shuffled_clean_docs[:train_size+valid_size]

    train_y = y[:train_size]
    valid_y = y[train_size:train_size+valid_size]
    test_y = y[train_size+valid_size:]

    train_y_prob = y_prob[:train_size]
    valid_y_prob = y_prob[train_size:train_size+valid_size]
    test_y_prob = y_prob[train_size+valid_size:]


##########################################
# Start Build graph...
# by creating word sets
##########################################
print('Build graph...')

# df의 word set 만들어주기/list로 바꾸고 len 구하기
word_set = set()
for doc_words in shuffled_clean_docs:
    words = doc_words.split() #['bill', 'left', 'when', 'that', 'no', 'one', 'else', 'was', 'awake', 'is', 'certain']
    for word in words:
        word_set.add(word)
vocab = list(word_set)
vocab_size = len(vocab)

# df vocab이랑 idx 랑 mapping 해주는 dict만들기
vocab_map = {}
for i in range(vocab_size):
    vocab_map[vocab[i]] = i

# train_valid 의 wordset 만들기 / list로 바꾸고 len 구하기
word_set_train_valid = set()
for doc_words in train_valid_docs:
    words = doc_words.split() #['bill', 'left', 'when', 'that', 'no', 'one', 'else', 'was', 'awake', 'is', 'certain']
    for word in words:
        word_set_train_valid.add(word)
vocab_train_valid = list(word_set_train_valid)
vocab_train_valid_size = len(vocab_train_valid)
    

##########################################
# create dictionary of 
# the word and the indexes where it occurs
# ex) 'bill': [0, 13, 24, 33, 45, 71, 73]

# frequency dict도 만들어줌
# ex) 'bill': 7
##########################################
# a map for word -> doc_list
if tfidf_mode=='all_tf_train_valid_idf':
    for_idf_docs = train_valid_docs
else: #맨처음에 세팅 = only_tf
    for_idf_docs = shuffled_clean_docs 

word_doc_list = {}
for i,doc_words in enumerate(for_idf_docs):
    words = doc_words.split() #['bill', 'left', 'when', 'that', 'no', 'one', 'else', 'was', 'awake', 'is', 'certain']
    
    appeared = set()
    for word in words:
        if word in appeared:
            continue #skip if the word appeared
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list 
        else:
            word_doc_list[word] = [i] #처음보는 단어면 dict에 넣어주기
        appeared.add(word)

# 단어 frequency 알려주는 dict 만들기
word_doc_freq = {}
for word, doc_list in word_doc_list.items(): 
    # ex) word = 'bill' , doc_list = [0, 13, 24, 33, 45, 71, 73]
    word_doc_freq[word] = len(doc_list)



'''
Doc word heterogeneous graph
and Vocabulary graph
'''
print('Calculate First isomerous adj and First isomorphic vocab adj, get word-word PMI values')

#쓰지도않음;;;;;;;;
'''
adj_y=np.hstack( ( train_y, np.zeros(vocab_size), valid_y, test_y )) # 한줄로 stack [0. 1. 1. ... 0. 1. 1.]
adj_y_prob=np.vstack(( train_y_prob, np.zeros((vocab_size,len(label2idx)),dtype=np.float32), valid_y_prob, test_y_prob ))
'''


##########################################
# CREATE WINDOW OF TRAIN_VALID_DOCS
##########################################
#윈도우는 걍 sentence 하나
# 1000을 넘어갈때는 마지막 1000개 단어만 treat
windows = []
for doc_words in train_valid_docs:
    words = doc_words.split()
    length = len(words)
    if length <= window_size: #initially 1000
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

print('Train_valid size:',len(train_valid_docs),'Window number:',len(windows))
#Train_valid size: 8551 Window number: 8551


##########################################
# 단어 frequency 구해주는 dict 만듬
# 단 같은 문장에 있으면 하나로만 쳐줌
##########################################
word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared: #if 단어가 appeared 안에 있으면
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])


##########################################
# word pair frequency 구해주는 dict 만듬
# [i,j] 도 구하고
# [j, i] 도 구함 (symmetric)

#* vocab id 는 df로 하네...신기방기 (data는 train_valid)
##########################################
word_pair_count = {}
for window in windows:
    appeared = set()
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = vocab_map[word_i] #vocab map = df vocab이랑 idx 랑 mapping 해주는 dict만들기
            word_j = window[j]
            word_j_id = vocab_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)

            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in appeared:
                continue
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            appeared.add(word_pair_str)


##########################################
#PMI 구하기..!

#pmi 로 구한건 row, col,weight에
#npmi로 구한건 vocab_adj_row, vocab_adj_col,vocab_adj_weight 에
##########################################
from math import log

row = []
col = []
weight = []
tfidf_row = []
tfidf_col = []
tfidf_weight = []
vocab_adj_row=[]
vocab_adj_col=[]
vocab_adj_weight=[]

num_window = len(windows)
tmp_max_npmi=0
tmp_min_npmi=0
tmp_max_pmi=0
tmp_min_pmi=0

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]] # word_window_freq = 단어 frequency 구해주는 dict
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    # 使用normalized pmi:
    npmi = log(1.0 * word_freq_i * word_freq_j/(num_window * num_window))/log(1.0 * count / num_window) -1
    if npmi>tmp_max_npmi: tmp_max_npmi=npmi
    if npmi<tmp_min_npmi: tmp_min_npmi=npmi
    if pmi>tmp_max_pmi: tmp_max_pmi=pmi
    if pmi<tmp_min_pmi: tmp_min_pmi=pmi
    if pmi>0:
        row.append(train_size + i)
        col.append(train_size + j)
        weight.append(pmi)
    if npmi>0:
        vocab_adj_row.append(i)
        vocab_adj_col.append(j)
        vocab_adj_weight.append(npmi)
print('max_pmi:',tmp_max_pmi,'min_pmi:',tmp_min_pmi)
print('max_npmi:',tmp_max_npmi,'min_npmi:',tmp_min_npmi)


##########################################
#doc-word tf-idf 구하기..!

# 물어보기
##########################################
print('Calculate doc-word tf-idf weight')

#doc-word frequency 구하는 dict 만들기
n_docs = len(shuffled_clean_docs) 
doc_word_freq = {}
for doc_id in range(n_docs):
    doc_words = shuffled_clean_docs[doc_id]
    words = doc_words.split() #['bill', 'left', 'when', 'that', 'no', 'one', 'else', 'was', 'awake', 'is', 'certain']

    for word in words:
        word_id = vocab_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

# for each word, create 
for i in range(n_docs):
    doc_words = shuffled_clean_docs[i]
    words = doc_words.split() #['bill', 'left', 'when', 'that', 'no', 'one', 'else', 'was', 'awake', 'is', 'certain']

    doc_word_set = set()
    tfidf_vec = []

    for word in words:
        if word in doc_word_set:
            continue
        j = vocab_map[word]
        key = str(i) + ',' + str(j) #j: word, i: sentence
        tf = doc_word_freq[key]
        tfidf_row.append(i)
        if i < train_size: 
            row.append(i) 
        else:
            row.append(i + vocab_size) 
        tfidf_col.append(j)
        col.append(train_size + j)
        # smooth
        idf = log((1.0 + n_docs) / (1.0+word_doc_freq[vocab[j]])) +1.0
        # weight.append(tf * idf)
        if tfidf_mode=='only_tf':
            tfidf_vec.append(tf)
        else:
            tfidf_vec.append(tf * idf)
        doc_word_set.add(word)
    if len(tfidf_vec)>0:
        weight.extend(tfidf_vec)
        tfidf_weight.extend(tfidf_vec)
    


node_size = vocab_size + corpus_size


##########################################
# CREATE PMI ADJACENCY MATRICES
##########################################
node_size = vocab_size + corpus_size #corpus_size: len(y), y는 df로 만듬

#모든거 다 합쳐진 adj matrix 만들기
# https://github.com/tkipf/pygcn/issues/3 
# https://github.com/yao8839836/text_gcn/issues/17 directed -> undirected 
adj_list = []
adj_list.append(sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size), dtype=np.float32))
for i,adj in enumerate(adj_list):
    adj_list[i] = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) 
    adj_list[i].setdiag(1.0)

# vocab adj matrix 만들기 
vocab_adj=sp.csr_matrix((vocab_adj_weight, (vocab_adj_row, vocab_adj_col)), shape=(vocab_size, vocab_size), dtype=np.float32)
vocab_adj.setdiag(1.0)

##########################################
# CREATE TF-IDF ADJACENCY MATRICES
##########################################
print('Calculate isomorphic vocab adjacency matrix using doc\'s tf-idf...')
tfidf_all=sp.csr_matrix((tfidf_weight, (tfidf_row, tfidf_col)), shape=(corpus_size, vocab_size), dtype=np.float32)

tfidf_train=tfidf_all[:train_size]
tfidf_valid=tfidf_all[train_size:train_size+valid_size]
tfidf_test=tfidf_all[train_size+valid_size:]
tfidf_X_list=[tfidf_train,tfidf_valid,tfidf_test]

vocab_tfidf=tfidf_all.T.tolil()

for i in range(vocab_size):
    norm=np.linalg.norm(vocab_tfidf.data[i])
    if norm>0:
        vocab_tfidf.data[i]/=norm
vocab_adj_tf=vocab_tfidf.dot(vocab_tfidf.T)

# check
print('Check adjacent matrix...')
for k in range(len(adj_list)):
    count=0
    for i in range(adj_list[k].shape[0]):
        if adj_list[k][i,i]<=0:
            count+=1
            print('No.%d adj, abnomal diagonal found, No.%d'%(k,i))
    if count>0:
        print('No.%d adj, totoal %d zero diagonal found.'%(k,count))

# dump objects
if will_dump_objects:
    print('Dump objects...')

    with open(dump_dir+"/data_%s.labels"%cfg_ds, 'wb') as f:
        pkl.dump([label2idx,idx2label], f)

    with open(dump_dir+"/data_%s.vocab_map"%cfg_ds, 'wb') as f:
        pkl.dump(vocab_map, f)

    with open(dump_dir+"/data_%s.vocab"%cfg_ds, 'wb') as f:
        pkl.dump(vocab, f)

    with open(dump_dir+"/data_%s.adj_list"%cfg_ds, 'wb') as f:
        pkl.dump(adj_list, f)
    with open(dump_dir+"/data_%s.y"%cfg_ds, 'wb') as f:
        pkl.dump(y, f)
    with open(dump_dir+"/data_%s.y_prob"%cfg_ds, 'wb') as f:
        pkl.dump(y_prob, f)
    with open(dump_dir+"/data_%s.train_y"%cfg_ds, 'wb') as f:
        pkl.dump(train_y, f)
    with open(dump_dir+"/data_%s.train_y_prob"%cfg_ds, 'wb') as f:
        pkl.dump(train_y_prob, f)
    with open(dump_dir+"/data_%s.valid_y"%cfg_ds, 'wb') as f:
        pkl.dump(valid_y, f)
    with open(dump_dir+"/data_%s.valid_y_prob"%cfg_ds, 'wb') as f:
        pkl.dump(valid_y_prob, f)
    with open(dump_dir+"/data_%s.test_y"%cfg_ds, 'wb') as f:
        pkl.dump(test_y, f)
    with open(dump_dir+"/data_%s.test_y_prob"%cfg_ds, 'wb') as f:
        pkl.dump(test_y_prob, f)
    with open(dump_dir+"/data_%s.tfidf_list"%cfg_ds, 'wb') as f:
        pkl.dump(tfidf_X_list, f)
    with open(dump_dir+"/data_%s.vocab_adj_pmi"%(cfg_ds), 'wb') as f:
        pkl.dump(vocab_adj, f)
    with open(dump_dir+"/data_%s.vocab_adj_tf"%(cfg_ds), 'wb') as f:
        pkl.dump(vocab_adj_tf, f)
    with open(dump_dir+"/data_%s.shuffled_clean_docs"%cfg_ds, 'wb') as f:
        pkl.dump(shuffled_clean_docs, f)
        
# print('Data prepared, spend %.2f s'%(time.time()-start))
