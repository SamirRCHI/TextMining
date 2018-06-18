# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:15:43 2018

@author: sfarooq1
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, merge, GRU, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
import html2text
import nltk
import operator
from scipy.cluster.hierarchy import linkage
import hierarchical as hc
from sklearn.linear_model import LinearRegression as LR
import scipy.spatial.distance as ssd
from copy import deepcopy as dc
import string
from time import time
import sys
import pickle
from axishelper import adj_axis

def import_data(filename = 'IRB_paragraph_data_modified.csv'):
    f = open(filename)
    fcsv = csv.reader(f)
    P,IDS,i = [],[],0
    h = html2text.HTML2Text()
    for row in fcsv:
        if i == 0:
            i += 1
            continue
        p = h.handle(row[2])
        newp = []
        for i in p:
            if i == '\n':
                i = ' '
            newp.append(i)
        P.append("".join(newp))
        IDS.append(row[0])
    return P

def flush_text(l, L = ''):
    if type(l) is str:
        L += l
        L += ' '
        return L
    else:
        for i in range(len(l)):
            L = flush_text(l[i], L)
    return L

def sentenizer(P, check_abrupt_start = True, check_space=True,
               balance_paranthesis = False, separate_by_P = False):
    s2 = []
    caps = set(string.ascii_uppercase)
    landd = set(string.ascii_lowercase+string.digits+')')
    alll = set(string.ascii_letters)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    punct = set(string.punctuation)
    s4 = []
    for i in range(len(P)): # Deal with spacing issue in sentences!
        s = nltk.sent_tokenize(P[i])
        mini_s = []
        for m in range(len(s)):
            j = s[m]
            k,k0 = 0,0
            while k < len(j):
                if check_abrupt_start and (j[k] in caps):
                    if (k0 < k):
                        if (j[k-1] in landd) and (j[k+1] in alll):
                            s2.append(j[k0:k])
                            mini_s.append(j[k0:k])
                            k0 = k
                        elif check_space and (j[k-1] == ' '):
                            rng = np.arange(k-1,k0-1,-1)
                            started_word = False
                            wrd = ''
                            for idx in rng:
                                if started_word and (j[idx] == ' '):
                                    break
                                elif started_word:
                                    wrd = j[idx] + wrd
                                elif j[idx] != ' ':
                                    wrd = j[idx] + wrd
                                    started_word = True
                            thiswrd = j[k]
                            for idx in range(k+1,len(j)):
                                if j[idx] in alll:
                                    thiswrd += j[idx]
                                else:
                                    break
                            nxtwrd,started_word = '',False
                            for nidx in range(idx,len(j)):
                                if (j[nidx] not in alll) and not started_word:
                                    continue
                                elif j[nidx] not in alll:
                                    break
                                else:
                                    nxtwrd += j[nidx]
                            nsw = wrd not in stopwords
                            nc = not any(c in punct for c in wrd)
                            truths = [c in caps for c in thiswrd]
                            na = (sum(truths) < 2) and (len(truths) > 1)
                            if len(wrd) > 0:
                                prt1 = wrd[0] not in caps
                            else:
                                prt1 = True
                            if len(nxtwrd) > 0:
                                prt2 = nxtwrd[0] not in caps
                            else:
                                prt2 = True
                            ns = prt1 and prt2
                            if nsw and nc and na and ns:
                                s2.append(j[k0:k])
                                mini_s.append(j[k0:k])
                                k0 = k
                        elif j[k-1] == '.':
                            if (j[k-2] == ')') or (j[k==2] in alll):
                                s2.append(j[k0:k])
                                mini_s.append(j[k0:k])
                                k0 = k
                k += 1
            s2.append(j[k0:k])
            mini_s.append(j[k0:k])
        s4.append(mini_s)
    if balance_paranthesis:
        s3 = []
        stack = []
        open_found,closed_found = False,True
        for s in s2:
            for i in range(len(s)):
                l = s[i]
                if l == '(' and closed_found:
                    open_found = True
                    closed_found = False
                elif (l == ')') and open_found:
                    closed_found = True
                    open_found = False
                elif (l == '.') and (i > (len(s)-3)):
                    closed_found = True
                    open_found = False
            if closed_found:
                if len(stack) > 0:
                    new_s = ''.join(stack)
                    stack = []
                    new_s += s
                else:
                    new_s = s
                s3.append(new_s)
            else:
                stack.append(s)
        s2 = s3
    if separate_by_P:
        return s4
    return s2

def get_sentence_and_vocabulary(P):
    sentences = sentenizer(P,True,True)
    words = []
    for sentence in sentences:
        words += list(nltk.word_tokenize(sentence))
    count, D, iD, iV = {},{},[],[]
    for word in words:
        word = word.lower()
        try:
            count[word] += 1
        except KeyError:
            count[word] = 1
    scount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(len(scount)):
        word = scount[i][0]
        D[word] = i
        iD.append(word)
    for word in words:
        word = word.lower()
        iV.append(D[word])
    ss,se = len(D),len(D)+1
    iD.append("SENTENCE_START")
    iD.append("SENTENCE_END")
    D["SENTENCE_START"] = ss
    D["SENTENCE_END"] = se
    return iV, count, D, np.array(iD), sentences

class word2vec:
    def __init__(self, vocab_size = None, window_size = 3, vector_dim = 300,
                 valid_size = 16, valid_window = 100, epochs = 1000000):
        self.vector_dim,self.epochs = vector_dim,epochs
        self.window_size = window_size
        self.valid_size = valid_size
        self.valid_examples = np.random.choice(valid_window,valid_size,False)
        self.vocab_size = vocab_size
        input_target = Input((1,))
        input_context = Input((1,)) # The name (below) allows us to access it.
        embedding = Embedding(self.vocab_size, self.vector_dim, input_length=1, 
                              name='embedding') #hidden layers of shape vs x vd
        target = embedding(input_target)
        target = Reshape((self.vector_dim, 1))(target)
        context = embedding(input_context)
        context = Reshape((self.vector_dim, 1))(context)
        similarity = merge([target, context], mode='cos', dot_axes=0)
        dot_product = merge([target, context], mode='dot', dot_axes=1)
        dot_product = Reshape((1,))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)
        self.model = Model(input=[input_target, input_context], output=output)
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        self.validation_model=Model(input=[input_target, input_context],
                               output=similarity)
        self.norms = {}
        self.dots = {}
        self.word_to_summary = {}
    def run_sim(self):
        for i in range(self.valid_size):
            valid_word = self.reverse_dictionary[self.valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(self.valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
    def _get_sim(self,valid_word_idx):
        sim = np.zeros((self.vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(self.vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = self.validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
    def fit(self, V, reverse_dictionary):
        self.reverse_dictionary = reverse_dictionary
        if self.vocab_size == None:
            self.vocab_size = len(np.unique(V))
        sampling_table = sequence.make_sampling_table(self.vocab_size)
        couples,labels=skipgrams(V, self.vocab_size, self.window_size, 
                                 sampling_table=sampling_table)
        word_target, word_context = zip(*couples)
        word_target = np.array(word_target, dtype="int32")
        word_context = np.array(word_context, dtype="int32")
        arr_1 = np.zeros((1,))
        arr_2 = np.zeros((1,))
        arr_3 = np.zeros((1,))
        for cnt in range(self.epochs):
            idx = np.random.randint(0, len(labels)-1)
            arr_1[0,] = word_target[idx]
            arr_2[0,] = word_context[idx]
            arr_3[0,] = labels[idx]
            loss = self.model.train_on_batch([arr_1,arr_2], arr_3)
            if cnt % 10000 == 0:
                print("Iteration {}, loss={}".format(cnt,loss))
                self.run_sim()
    def save_model(self, filename = 'IRB_word2vec_model.hdf5'):
        self.model.save_weights(filename)
    def load_model(self, idxtoword, filename = 'IRB_word2vec_model.hdf5'):
        self.model.load_weights(filename)
        self.reverse_dictionary = idxtoword
    def create_summary(self):
        word_to_summary(self)
    def create_cosine_similarities(self,wordtoidx):
        self.w2i = wordtoidx
        l = len(self.w2i)-2
        self.cs = np.zeros((l,l))
        for i in range(l):
            self.cs[i,i] = 1.0
            word1 = self.reverse_dictionary[i]
            for j in range(i+1,l):
                word2 = self.reverse_dictionary[j]
                try:
                    n1 = self.norms[word1]
                except KeyError:
                    n1 = np.linalg.norm(self.word_to_summary[word1])
                    self.norms[word1] = n1
                try:
                    n2 = self.norms[word2]
                except KeyError:
                    n2 = np.linalg.norm(self.word_to_summary[word2])
                    self.norms[word2] = n2
                cos_sim = np.dot(self.word_to_summary[word1],
                                 self.word_to_summary[word2]) / (n1*n2)
                self.cs[i,j] = cos_sim
                self.cs[j,i] = cos_sim
    def cosine_similarity(self, word1, word2):
        return self.cs[self.w2i[word1],self.w2i[word2]]

#P = import_data()
#V,count,wordtoidx,idxtoword,sentences = get_vocabulary(P)
#wv = word2vec(len(count))
#wv.fit(V, idxtoword) # This is to re-fit the data! -> then save the model!
#wv.save_model('IRB_modified_word2vec_model.hdf5')
#wv.load_model(idxtoword) # This is to load a pre-fitted model!
#hm = wv.model.layers[2].get_weights()[0] # Hidden layer extraction from model

def idxtoonehot(idx, length, shape = False):
    if shape == 'bool':
        x = np.zeros((length,),dtype=np.bool)
        if idx >= 0:
            x[idx] = True
    elif shape == 'matrix':
        x = np.zeros((1,length))
        if idx >= 0:
            x[0,idx] = 1
    else:
        x = np.zeros((length,))
        if idx >= 0:
            x[idx] = 1
    return x

def word_to_summary(model):
    hm = model.model.layers[2].get_weights()[0]
    onehotlen,hiddenlayers = np.shape(hm)
    model.word_to_summary = {}
    for idx in range(len(model.reverse_dictionary)):
        word = model.reverse_dictionary[idx]
        try:
            model.word_to_summary[word] = np.dot(idxtoonehot(idx,onehotlen),hm)
        except IndexError:
            continue
        
#word_to_summary(wv)

def save_hiddenlayer(model,filename='IRB_word2vec_hidden_layer.csv'):
    f = open(filename,'w',newline='')
    a = csv.writer(f)
    hidden_layer = model.model.layers[2].get_weights()[0]
    #or: hidden_layer = model.model.get_layer('embedding').get_weights()[0]
    a.writerows(hidden_layer)
    f.close()

def get_nearest(word,model,count,display = False,return_amt=1,count_max=89):
    if type(word) is str:
        wrdvec = model.word_to_summary[word]
    else:
        wrdvec = word
    distances = []
    words = []
    for wrd,nxtvec in model.word_to_summary.items():
        words.append(wrd)
        cosinesim = np.dot(wrdvec,nxtvec)/\
            (np.linalg.norm(wrdvec)*np.linalg.norm(nxtvec))
        distances.append(cosinesim)
    a = np.argsort(distances)[::-1]
    if display:
        print("Nearest to "+word+":")
        for i in range(8):
            print("\t"+words[a[i]])
    words = np.array(words)
    if type(return_amt) is int:
        i,best_words = 0,[]
        while len(best_words) < return_amt:
            try_word = words[a[i]]
            if type(count) is dict:
                if count[try_word] <= count_max:
                    best_words.append(try_word)
            else:
                count_matrix, wordtoidx = count
                if np.mean(count_matrix[:, wordtoidx[try_word]]) <= count_max:
                    best_words.append(try_word)
            i += 1
        return best_words
    elif return_amt == 'all':
        return words[a]
    
def pick_most_freqs(freqvec, idxtoword, words = 8):
    a = np.argsort(freqvec)[::-1]
    ws = []
    for i in range(words):
        ws.append(idxtoword[a[i]])
    return ', '.join(ws)

def iqr(array):
    Q1 = np.percentile(array,25)
    Q3 = np.percentile(array,75)
    return Q3-Q1

def rotatebyLR(x,y, LR, absolute_distance = False):
    m,b = LR.coef_[0],LR.intercept_
    im = -(1.0/m)
    yd = []
    for i in range(len(x)):
        ib = y[i] - im*x[i] # find equation of perpendicular line to the point
        nx = (ib - b)/(m - im) # x point of intersection
        ny = im*nx + ib # y point of intersection
        d = np.sqrt((x[i]-nx)**2+(y[i]-ny)**2) # distance from LR
        if not absolute_distance:
            if ny > y[i]:
                d = -d
        yd.append(d)
    return np.array(yd)

def nto01(array):
    mn,mx = min(array),max(array)
    return (array - mn)/(mx-mn)

def get_lim(array, area = 0.025):
    mn,mx = np.min(array),np.max(array)
    dif = area*(mx-mn)
    return mn-dif,mx+dif

def prepare_word_list(sentences):
    word_list,mxwords = [],0
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        word_list.append(["SENTENCE_START"] + words + ["SENTENCE_END"])
        if len(word_list[-1]) > mxwords:
            mxwords = len(word_list[-1])
    mxwords -= 1
    samples = 0
    for words in word_list: samples += (len(words) - 1)
    return word_list,mxwords,samples

def word_occurance(P, count, wordtoidx, idxtoword, cmx = 0.005, plot = False,
                   rtrn = True, use_iqr = False, annotate = 0):
    sentences = sentenizer(P,True,True,False,True)
    wc = np.zeros((len(P),len(count)))
    for i in range(len(P)):
        words = []
        for sent in sentences[i]:
            words += nltk.word_tokenize(sent)
        for word in words:
            wc[i,wordtoidx[word.lower()]] += 1
        wc[i,:] *= (1.0/len(words))
    amt,stn = [],[]
    for i in range(len(count)):
        amt.append(np.mean(wc[:,i]))
        if use_iqr:
            s = iqr(wc[:,i])
        else:
            s = np.std(wc[:,i])
        stn.append(s)
    L = LR()
    amt,stn = np.array(amt),np.array(stn)
    mxamt = np.max(amt)
    if cmx > mxamt:
        cmx = mxamt
    indxs = np.arange(len(count))
    amt,stn,indxs = amt[amt <= cmx], stn[amt <= cmx], indxs[amt <= cmx]
    L.fit(amt.reshape((-1,1)),stn)
    rstn = rotatebyLR(amt, stn, L)
    if plot:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(211)
        ax.scatter(amt,stn)
        ax.set_xlim(get_lim(amt)),ax.set_ylim(get_lim(stn))
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        m = np.min(amt)
        ax.plot([m,cmx],[L.predict(m)[0],L.predict(cmx)[0]],'--',color='r')
        ax.set_xlim(xlim),ax.set_ylim(ylim)
        ax = fig.add_subplot(212)
        ax.scatter(amt, rstn,zorder=2)
        if annotate:
            srstn = np.argsort(rstn)[::-1][:annotate]
            wrds = [idxtoword[indxs[i]] for i in srstn]
            for i in range(len(wrds)):
                ax.text(amt[srstn[i]],rstn[srstn[i]],wrds[i],va='center')
        ax.set_xlim(get_lim(amt)),ax.set_ylim(get_lim(rstn))
        xlim,ylim = ax.get_xlim(),ax.get_ylim()
        ax.fill_between(xlim,0,[ylim[0],ylim[0]],color='gray',alpha=0.2,
                        zorder=1)
        ax.set_xlim(xlim), ax.set_ylim(ylim)
    if rtrn:
        return wc,rstn,indxs
    
def best_rep_from_vecs(vecs):
    if len(vecs) <= 1:
        return vecs[0]
    mn,arg = float('inf'),None
    for i in range(len(vecs)):
        max_dists = []
        for j in range(1,len(vecs)):
            cs = 1 - np.dot(vecs[i],vecs[j])/(np.linalg.norm(vecs[i])*\
                        np.linalg.norm(vecs[j]))
            max_dists.append(cs)
        cs = np.max(max_dists)
        if cs < mn:
            mn = cs
            arg = i
    return vecs[arg]

def cluster_by_word2vec(P,model,wordtoidx,count,idxtoword,topics=16,
                        freqmx = 0.005, count_max=89,t=0.4,std_adj='std',
                        lower_thresh = False):
    Matrix = []
    use_iqr = False
    if std_adj == 'iqr':
        use_iqr = True
    wc, rstn, indxs=word_occurance(P,count,wordtoidx,idxtoword,
                                   freqmx,use_iqr=use_iqr)
    for i in range(len(P)):
        if model:
            counts = np.zeros((model.vector_dim,))
        else:
            counts = np.zeros((len(count),))
        used_words = 0
        for idx in range(len(indxs)):
            j = indxs[idx]
            if rstn[idx] < 0: #adjust indexing for first true rstn
                continue
            cntarray = nto01(wc[:, j]) # Norm word freq to [0,1]
            if std_adj:
                cntarray *= rstn[idx] # Multiply by adjusted stddev weight
            n_count = cntarray[i] # Pick the word of the correct row
            if model:
                counts += model.word_to_summary[idxtoword[j]]*n_count
            else:
                counts += idxtoonehot(j, len(count))*n_count
            used_words += n_count
        counts /= used_words
        Matrix.append(counts)
    Matrix = np.array(Matrix)
    Z = linkage(Matrix,'average',metric='cosine')
    LC = hc.linkage_clustering(Z,t)
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(1,3,1)
    LC.plot_dendrogram(ax,orientation='left',yflip=False)
    ylim = ax.get_ylim()
    ax = fig.add_subplot(1,3,2)
    pos = ax.get_position()
    ax.set_position([pos.x0,pos.y0,1-pos.x0,pos.height])
    bnds = LC.get_plot_boundaries()
    best_words = []
    for c in bnds:
        #avg = np.mean(Matrix[np.array(list(LC.Clusters[c]))],axis=0)
        if lower_thresh:
            if len(LC.Clusters[c]) <= lower_thresh:
                continue
        avg = best_rep_from_vecs(Matrix[np.array(list(LC.Clusters[c]))])
        if model:
            bw = get_nearest(avg,model,(wc, wordtoidx),False,topics,freqmx)
            best_words.append(bw)
            txt = ', '.join(bw)
        else:
            txt = pick_most_freqs(avg, idxtoword, topics)
        ax.text(0,bnds[c][1],txt,color=LC.colors[c],va='center')
    ax.set_ylim(ylim)
    ax.axis('off')
    return best_words
    
def get_avg_vects(idxs, idxtoword, model):
    avg = np.zeros((model.vector_dim,))
    for i in idxs:
        avg += model.word_to_summary[idxtoword[i]]
    return avg / len(idxs)

def get_best_rep(idxs, idxtoword, model):
    M = model.cs[idxs,:][:,idxs]
    mn,a = float('inf'),None
    for m in range(len(M)):
        trial = np.max(M[m])
        if trial < mn:
            mn = trial
            a = m
    w = idxtoword[idxs[a]]
    return model.word_to_summary[w]

def find_idx(i, x):
    ncol = x.shape[1]
    return int(i/ncol), i%ncol    

def euc_vs_cosine(doc_mine):
    euc = np.zeros(np.shape(doc_mine.w2v.cs))
    l,l = np.shape(euc)
    for i in range(l):
        v1 = doc_mine.w2v.word_to_summary[doc_mine.i2w[i]]
        euc[i,i] = 0.0
        for j in range(i+1, l):
            v2 = doc_mine.w2v.word_to_summary[doc_mine.i2w[j]]
            e = np.linalg.norm(v1 - v2)
            euc[i,j] = e
            euc[j,i] = e
    cs = 1 - doc_mine.w2v.cs
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.scatter(cs.flatten(), euc.flatten())
    plt.show()
    
def plt_test(test):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(test)):
        ax.scatter(test[i,0],test[i,1],marker = r'$'+str(i)+'$')
    plt.show()
    
class NodeV:
    def __init__(self, value = None, index = None, minarg = None):
        self.value = value
        self.index = index
        self.minarg = minarg
        self.above = None
        self.below = None
    def display(self):
        print(self.value, self.index, self.minarg)

class StackV:
    def __init__(self, L = None):
        self.head = None
        self.length = 0
    def __len__(self):
        return self.length
    def stack_above(self, Node):
        Node.below = self.head
        if self.head != None:
            self.head.above = Node
        self.head = Node
        self.length += 1
    def stack(self, Node):
        if self.head == None:
            self.head = Node
        else:
            cur_node = self.head
            while (cur_node.below != None) and (Node.value > cur_node.value):
                cur_node = cur_node.below
            if Node.value > cur_node.value:
                cur_node.below = Node
                Node.above = cur_node
            else:
                Node.below = cur_node
                if cur_node.above != None:
                    cur_node.above.below = Node
                    Node.above = cur_node.above
                cur_node.above = Node
                if Node.above == None:
                    self.head = Node
        self.length += 1
    def pop(self):
        if self.head == None:
            raise AssertionError("Nothing left to pop!")
        this_node = (self.head.value, self.head.index, self.head.minarg)
        if self.head.below != None:
            self.head.below.above = None
        self.head = self.head.below
        self.length -= 1
        return this_node
    def display(self):
        cur_node = self.head
        while cur_node.below != None:
            cur_node.display()
            cur_node = cur_node.below
        cur_node.display()

class centoid_linkage:
    def __init__(self, M):
        self.S = {s:[s] for s in range(len(M))}
        self.C = {s:[s] for s in range(len(M))}
        self.T = {}
        self.cn = len(M)
        self.M = M
        mins, mina = np.array([]), np.array([])
        for m in range(len(M)):
            a = np.argmin(self.M[m][np.arange(len(M))!=m])
            if a >= m: a += 1
            mina = np.append(mina, a)
            mins = np.append(mins, self.M[m][a])
        a = np.argsort(mins)[::-1]
        self.Stack = StackV()
        #self.Stack = [(mins[arg], arg, mina[arg]) for arg in a]
        for arg in a:
            self.Stack.stack_above(NodeV(mins[arg], arg, int(mina[arg])))
        print("Initial stack size: "+str(len(self.Stack)))
        self.Z = []
        while len(self.Z) < (len(M)-1):
            self.update()
        self.Z = np.array(self.Z)
    def update(self):
        mn,a,argm = self.Stack.pop()
        print("\rStack Size: "+str(len(self.Stack))+", S Size: "+\
              str(len(self.S))+", Z Size: "+str(len(self.Z))+'\t', end = '')
        sys.stdout.flush()
        check_a = a not in self.T
        check_argm = argm not in self.T
        if check_a and check_argm:
            #argm = self.get_last(argm)
            self.T[a] = self.cn
            self.T[argm] = self.cn
            z=[min([a,argm]),max([a,argm]),mn,len(self.C[a])+len(self.C[argm])]
            self.Z.append(z)
            I = self.C[a] + self.C[argm]
            self.C[self.cn] = I
            centoid = self.choose_centoid(I)
            self.S.pop(a)
            self.S.pop(argm)
            newmn,newarg = self.find_closest(centoid, self.cn)
            self.S[self.cn] = centoid
            self.Stack.stack(NodeV(newmn, self.cn, newarg))
            self.cn += 1
        elif check_a and not check_argm:
            newmn,newarg = self.find_closest(self.C[a], a)
            self.Stack.stack(NodeV(newmn, a, newarg))
    def get_last(self, i):
        try:
            return self.get_last(self.T[i])
        except KeyError:
            return i
    def choose_centoid(self, I):
        if len(I) == 2:
            return list(I)
        M = self.M[I,:][:,I]
        s = np.array([np.max(M[m]) for m in range(len(M))])
        a = np.argsort(s)
        i = []
        l = s[a[0]]
        j = 0
        while (j < len(a)) and (s[a[j]] == l):
            i.append(I[a[j]])
            j += 1
        return i
    def find_closest(self, I, clusnum):
        mn,a = float('inf'),None
        for i in I:
            for s in self.S:
                if s == clusnum:
                    continue
                for j in self.S[s]:
                    val = self.M[i,j]
                    if val < mn:
                        mn = val
                        a = s
        return mn,a

def word_similarities(doc_mine, topics = 6, pwords = 10, paragraph = True,
                      freqmx = 0.005):
    P,count,model = doc_mine.P,doc_mine.count,doc_mine.w2v
    wordtoidx,idxtoword = doc_mine.w2i, doc_mine.i2w
    wc, rstn, indxs = word_occurance(P,count,wordtoidx,idxtoword,freqmx,False)
    i2w,w2i,i = np.array([]),{},0
    Mat2 = []
    for idx in indxs:
        word = idxtoword[idx]
        Mat2.append(doc_mine.w2v.word_to_summary[word])
        i2w = np.append(i2w, word)
        w2i[word] = i
        i += 1
    Matrix = np.array(Mat2)
    M = 1 - model.cs[:, indxs][indxs, :]
    #CL = centoid_linkage()
    #CL.linkage(M)
    #Z = CL.Z
    Z = linkage(ssd.squareform(M), 'average', 'cosine')
    s = np.sort(Z[:,2])[::-1]
    avg = (s[topics-1]+s[topics-2])/2
    LC = hc.linkage_clustering(Z, avg)
    F = np.zeros((len(P),topics))
    clus_sum = sum([len(cluster) for cluster in LC.Clusters])
    for i in range(len(doc_mine.osent)):
        for sent in doc_mine.osent[i]:
            for w in sent:
                try:
                    F[i, LC.B[w2i[w]]] += 1
                except KeyError:
                    continue
    for j in range(topics):
        F[:, j] /= (len(LC.Clusters[j])/clus_sum)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.violinplot(F)
    ax.set_ylabel('Actual Occurance / Likelihood of Occurance')
    plt.show()
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(1,3,1)
    LC.plot_dendrogram(ax,orientation='left',yflip=False)
    ylim = ax.get_ylim()
    ax = fig.add_subplot(1,3,2)
    pos = ax.get_position()
    ax.set_position([pos.x0,pos.y0,1-pos.x0,pos.height])
    bnds = LC.get_plot_boundaries()
    for c in bnds:
        l = str(len(LC.Clusters[c]))
        L = list(LC.Clusters[c])
        #a = np.argsort(rstn[L])[::-1][:pwords]
        #txt = ', '.join(i2w[L][a])
        avg = get_best_rep(L, i2w, model)
        bw = get_nearest(avg,model,(wc, wordtoidx),False,pwords,freqmx)
        txt = ', '.join(bw)
        middle = ', including the words:\n' if paragraph else '; '
        txt = str(c+1)+'; Cluster Size: '+l+middle+txt
        ax.text(0,bnds[c][1],txt,color=LC.colors[c],va='center')
    ax.set_ylim(ylim)
    ax.axis('off')

def print_word_similarities(Z, model, wc, wordtoidx, Zidxtoword,
                            topics = 10, pwords = 10, freqmx = 0.005):
    s = np.sort(Z[:,2])[::-1]
    avg = (s[topics-1]+s[topics-2])/2
    LC = hc.linkage_clustering(Z, avg)
    for c in LC.Clusters:
        l = str(len(c))
        avgvec = get_avg_vects(c, Zidxtoword, model)
        print("Cluster Size: "+l)
        print(get_nearest(avgvec,model,(wc, wordtoidx),False,pwords,freqmx))
        print("")

def plot_word_similarities(Z, model, wc, wordtoidx, Zidxtoword,
                           topics=10, pwords=10, freqmx=0.005):
    s = np.sort(Z[:,2])[::-1]
    avg = (s[topics-1]+s[topics-2])/2
    LC = hc.linkage_clustering(Z, avg)
    fig = plt.figure(figsize=(15.5,8.5))
    ax = fig.add_subplot(1,3,1)
    LC.plot_dendrogram(ax,orientation='left',yflip=False)
    ylim = ax.get_ylim()
    ax = fig.add_subplot(1,3,2)
    pos = ax.get_position()
    ax.set_position([pos.x0,pos.y0,1-pos.x0,pos.height])
    bnds = LC.get_plot_boundaries()
    for c in LC.Clusters:
        l = str(len(c))
        avgvec = get_avg_vects(c, Zidxtoword, model)
        txt = get_nearest(avgvec,model,(wc, wordtoidx),False,topics,freqmx)
        txt = l+': '+txt
        ax.text(0,bnds[c][1],txt,color=LC.colors[c],va='center')
    ax.set_ylim(ylim)
    ax.axis('off')
    
def vector_editdist(s1,s2,cs,sub_penalty = 2,normalized=True):
    d = np.zeros((len(s1)+1,len(s2)+1))
    d[:,0] = np.arange(len(s1)+1)
    d[0,:] = np.arange(len(s2)+1)
    for j in range(1,len(s2)+1):
        for i in range(1,len(s1)+1):
            sub_cost = cs.evaluate(s1[i-1].lower(),s2[j-1].lower())*sub_penalty
            d[i,j] = np.min([d[i-1, j] + 1, # deletion
                             d[i, j-1] + 1, # insertion
                             d[i-1, j-1] + sub_cost]) # subsitution
    result = d[len(s1),len(s2)]
    if normalized: result /= max([len(s1),len(s2)])
    return result
    
def document_clustering(P, model):
    redund_matrix = np.zeros((len(P),len(P)))
    for i in range(len(P)):
        P[i] = nltk.word_tokenize(P[i])
        for j in range(len(P[i])):
            P[i][j] = P[i][j].lower()
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            dist = model.cosine_similarity(P[i],P[j])
            redund_matrix[i,j] = dist
            redund_matrix[j,i] = dist
    distarray = ssd.squareform(redund_matrix)
    Z = linkage(distarray, 'ward')
    return Z

def detokenize(word_list):
    a="".join([" "+i if not i.startswith("'") and i not in string.punctuation \
             else i for i in word_list]).strip()
    return a

def get_training_matrix_for_GRU(sentences, wordtoidx):
    word_list,mxwords,samples = prepare_word_list(sentences)
    X = np.zeros((samples,mxwords,len(wordtoidx)), dtype = np.bool)
    y = np.zeros((samples,len(wordtoidx)), dtype = np.bool)
    j = 0
    for words in word_list:
        Z = np.zeros((mxwords,len(wordtoidx)), dtype = np.bool)
        thisword = idxtoonehot(wordtoidx[words[0]],len(wordtoidx),'bool')
        l,w = np.shape(Z)
        for i in range(len(words)-1):
            nxtword = words[i+1]
            if nxtword != "SENTENCE_END":
                nxtword = nxtword.lower()
            nextword = idxtoonehot(wordtoidx[nxtword],len(wordtoidx),'bool')
            if i > 0:
                Z[l-i-1:l-1,:] = Z[l-i:l,:] # Empty words are 0 vectors
            Z[l-1,:] = thisword
            X[j] = Z
            y[j] = nextword
            thisword = nextword
            j += 1
    return X,y

def text_generation(model, wordtoidx, idxtoword, generator = 'max'):
    mxsentlen = model.mxsentlen
    Z = np.zeros((1,mxsentlen, len(idxtoword)), dtype = np.bool)
    n,l,w = np.shape(Z)
    sentence = ''
    Z[0,-1,:] = idxtoonehot(wordtoidx["SENTENCE_START"], len(idxtoword),'bool')
    i = 1
    while (i < mxsentlen):
        prbarray = model.model.predict(Z)
        if generator == 'max':
            a = np.argmax(prbarray[0])
        else:
            raise Warning("No other generator available")
        newword = idxtoword[a]
        if newword == "SENTENCE_END":
            break
        sentence += newword
        sentence += ' '
        newvec = idxtoonehot(a, len(idxtoword), 'bool')
        Z[0,l-i-1:l-1,:] = Z[0,l-i:l,:]
        Z[0,-1,:] = newvec
        i += 1
    return sentence

class rNN_GRU:
    def __init__(self, vocab_size, max_sentence_len, epochs = 10,
                 batch_size = 100, hidden_layers = 128):
        self.vocab_size = vocab_size
        self.mxsentlen = max_sentence_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.model.add(GRU(hidden_layers,input_shape=
                           (max_sentence_len,vocab_size)))
        self.model.add(Dense(vocab_size, activation = 'softmax'))
        optimizer = RMSprop(lr=0.01)
        self.model.compile(loss='categorical_crossentropy',optimizer=optimizer)
    def fit(self,X,y, wordtoidx = None, idxtoword = None):
        try:
            for e in range(self.epochs):
                print("====== EPOCH: "+str(e)+"/"+str(self.epochs-1)+" ======")
                self.model.fit(X,y,epochs=1,batch_size=self.batch_size)
                if type(wordtoidx) is dict:
                    self.generate_text(wordtoidx,idxtoword)
        except KeyboardInterrupt:
            print("Attempting to safely exit...")
    def generate_text(self, wordtoidx, idxtoword, generator = 'max'):
        print(text_generation(self, wordtoidx, idxtoword, generator))
    def save_model(self, filename = 'IRB_GRU_model.hdf5'):
        self.model.save_weights(filename)
    def load_model(self, filename = 'IRB_GRU_model.hdf5'):
        self.model.load_weights(filename)
        
class batched_GRU:
    def __init__(self, sentences, wordtoidx, sentence_batch = 250,
                 epochs = 100, batch_size = 32, hidden_layers = 128,
                 use_LSTM = False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.sent_batch = sentence_batch
        self.vocab_size = len(wordtoidx)
        self.w2idx = wordtoidx
        self.current_sentence = 0
        self.word_list,self.mxsentlen,self.samples=prepare_word_list(sentences)
        learner = LSTM if use_LSTM else GRU
        self.model = Sequential()
        self.model.add(learner(hidden_layers,input_shape=
                           (self.mxsentlen,self.vocab_size)))
        self.model.add(Dense(self.vocab_size, activation = 'softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='adam')
    def generate_batch(self):
        if self.current_sentence >= (len(self.word_list)):
            self.current_sentence = 0
            return False
        end=min([self.current_sentence+self.sent_batch,len(self.word_list)])
        samples = 0
        for i in range(self.current_sentence,end): 
            samples += (len(self.word_list[i])-1)
        X = np.zeros((samples,self.mxsentlen,self.vocab_size), dtype = np.bool)
        y = np.zeros((samples,self.vocab_size), dtype = np.bool)
        sample = 0
        for i in range(self.current_sentence, end):
            words = self.word_list[i]
            Z = np.zeros((self.mxsentlen,self.vocab_size), dtype = np.bool)
            thisword = idxtoonehot(self.w2idx[words[0]],self.vocab_size,'bool')
            l,w = np.shape(Z)
            for j in range(len(words)-1):
                nxtwrd = words[j+1]
                if nxtwrd != "SENTENCE_END":
                    nxtwrd = nxtwrd.lower()
                nextwrd=idxtoonehot(self.w2idx[nxtwrd],self.vocab_size,'bool')
                if j > 0:
                    Z[l-j-1:l-1,:] = Z[l-j:l,:] # Empty words are 0 vectors
                Z[l-1,:] = thisword
                X[sample] = Z
                y[sample] = nextwrd
                thisword = nextwrd
                sample += 1
        self.current_sentence = end
        return X,y
    def fit(self, idxtoword):
        try:
            self.generate_text(idxtoword)
            for e in range(self.epochs):
                print("========== EPOCH: "+str(e)+" =============")
                Train = self.generate_batch()
                while type(Train) is not bool:
                    X,y = Train
                    self.model.fit(X,y,epochs=1,batch_size=self.batch_size)
                    Train = self.generate_batch()
                self.generate_text(idxtoword)
        except KeyboardInterrupt:
            print("\nAttempting to safely exit...")
    def generate_text(self, idxtoword, generator = 'max'):
        print(text_generation(self, self.w2idx, idxtoword, generator))
    def save_model(self, filename = 'IRB_GRU_model.hdf5'):
        self.model.save_weights(filename)
    def load_model(self, filename = 'IRB_GRU_model.hdf5'):
        self.model.load_weights(filename)

def similarity_histogram(wordtovec, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(wordtovec.cs[np.triu_indices_from(wordtovec.cs,1)],bins=20,
                         log=True)
    ax.set_xlim([-1,1])
    ax.set_title(title, fontsize=16)
    adj_axis(ax, {'xtick labelsize':14, 'ytick labelsize':14,
                  'xlabel':('Cosine Similarity (Correlation Coefficient)',14),
                  'ylabel':('Count',14)})
    plt.show()

def fastMemLev(s, t, wordtovec):
    # degenerate cases
    if len(s) == 0:
        return 0.0
    if len(t) == 0:
        return 0.0
  
    # create two work vectors of integer distances
    #int[] v0 = new int[t.Length + 1];
    #int[] v1 = new int[t.Length + 1];
    v0 = []
    v1 = []
  
    # initialize v0 (the previous row of distances)
    # this row is A[0][i]: edit distance for an empty s
    # the distance is just the number of characters to delete from t
    # for (int i = 0; i < v0.Length; i++)
    # v0[i] = i;
    for i in range(len(t)+1):
        v0.append(i)
        v1.append(0)
 
    for i in range(len(s)): 
        # calculate v1 (current row distances) from the previous row v0
        # first element of v1 is A[i+1][0]
        # edit distance is delete (i+1) chars from s to match empty t
        v1[0] = i + 1
  
        # use formula to fill in the rest of the row
        for j in range(len(t)):
            cost = 1 - wordtovec.cosine_similarity(s[i],t[j])
            v1[j + 1] = min(v1[j]+1, v0[j+1]+1, v0[j]+cost)
  
        # copy v1 (current row) to v0 (previous row) for next iteration
        for j in range(len(t)+1):
            v0[j] = v1[j]
  
    return v1[len(t)] / float(max([len(s),len(t)]))

def iTextGen(model, w2v, P, word_bank, filler_words = 4):
    mxsentlen = model.mxsentlen
    wordtoidx = w2v.w2i
    idxtoword = w2v.reverse_dictionary
    Z = np.zeros((1,mxsentlen, len(idxtoword)), dtype = np.bool)
    n,l,w = np.shape(Z)
    sentence,support = [],[]
    Z[0,-1,:] = idxtoonehot(wordtoidx["SENTENCE_START"], len(idxtoword),'bool')
    key_idxs = np.array([wordtoidx[word] for word in word_bank])
    j = 1
    not_satisfied = True
    while not_satisfied:
        prbarray = model.model.predict(Z)
        prbs = prbarray[0]
        a = np.argsort(prbs)[::-1]
        current_word_bank,prb_bank = {},{}
        print("Current Sentence: "+' '.join(sentence)+"\n")
        print("Filler Words:")
        for i in range(filler_words):
            print("Word: "+idxtoword[a[i]]+"\tProbability: "+str(prbs[a[i]]))
            current_word_bank[idxtoword[a[i]]] = a[i]
            prb_bank[idxtoword[a[i]]] = prbs[a[i]]
        kprbs = prbs[key_idxs]
        W = idxtoword[key_idxs]
        a = np.argsort(kprbs)[::-1]
        print("Word Bank:")
        for i in range(len(word_bank)):
            print("Word: "+W[a[i]]+"\tProbability: "+str(kprbs[a[i]]))
            current_word_bank[W[a[i]]] = a[i]
            prb_bank[W[a[i]]] = kprbs[a[i]]
        current_word_bank[''] = None
        current_word_bank['POP_WORD'] = None
        print("")
        new_word = "A_WORD_WHICH_DOESNT_EXIST"
        while new_word not in current_word_bank:
            new_word = input("Pick Next Word: ")
        if new_word == '':
            not_satisfied = False
        elif new_word == 'POP_WORD':
            sentence.pop()
            Z[0,l-j:l,:] = Z[0,l-j-1:l-1,:]
            Z[0,l-j-1,:] = 0.0
            support.pop()
            j-=1
        else:
            sentence.append(new_word)
            support.append(prb_bank[new_word])
            newvec = idxtoonehot(current_word_bank[new_word],
                                 len(idxtoword),'bool')
            Z[0,l-j-1:l-1,:] = Z[0,l-j:l,:]
            Z[0,-1,:] = newvec
            j+=1
    print("Maximum support probability: "+str(min(support)))
    print("Standard Deviation: "+(str(np.std(support))))
    matched_support,supporting_sentences = percent_support(sentence, P, w2v)
    print("Support: "+str(np.mean(matched_support)))
    print("+/-: "+str(np.std(matched_support)))
    return supporting_sentences,matched_support

def percent_support(sentence, P, w2v):
    sentences = sentenizer(P, separate_by_P = True)
    support = []
    supporting_sentences = []
    for o in range(len(sentences)):
        org_supports = []
        for un_broken_sent in sentences[o]:
            sent = nltk.word_tokenize(un_broken_sent)
            for i in range(len(sent)):
                sent[i] = sent[i].lower()
            org_supports.append(1-fastMemLev(sentence, sent, w2v))
        a = np.argmax(org_supports)
        support.append(org_supports[a])
        supporting_sentences.append(sentences[o][a])
    return support,supporting_sentences

class document_mining:
    def __init__(self, filename = None):
        self.filename,self.P = None,None
        self.V,self.count,self.w2i,self.i2w = None, None, None, None
        self.sentences,self.osent,self.w = None,None,None
        if filename != None:
            self.filename = filename
            self.P = import_data(filename)
            self.V,self.count,self.w2i,self.i2w,self.sentences = \
            get_sentence_and_vocabulary(self.P)
            osent = sentenizer(self.P,True,True,False,True)
            for o in range(len(osent)):
                for i in range(len(osent[o])):
                    words = nltk.word_tokenize(osent[o][i])
                    for j in range(len(words)):
                        words[j] = words[j].lower()
                    osent[o][i] = words
            self.osent = osent
            self.w = len(self.i2w)
        self.w2v = None
        self.gru = None
        self.word_list,self.mxsentlen,self.samples = None,None,None
        self.unjoined_sents,self.open_cases = [],1
        self.generated_sentences = []
        self.support_avg,self.support_std = [],[]
        self.all_supports = []
        self.fillers,self.thresh,self.mxsent_cut = None,None,None
        self.supporting_sentence_indexes = []
        self.within_sentences,self.within_support = [],[]
        self.within_avg_support,self.within_std_support = [],[]
    def add_w2v(self, w2v):
        self.w2v = w2v
    def add_gru(self, gru):
        self.gru = gru
        self.word_list,self.mxsentlen,self.samples = \
        prepare_word_list(self.sentences)
    def prepare(self, words):
        self.unjoined_sents.append(words)
        print('\r'+str(len(self.unjoined_sents))+' sentences found. '+\
              '\t Open cases: '+str(self.open_cases), end = '')
        sys.stdout.flush()
    def update_support(self, words):
        self.generated_sentences.append(' '.join(words))
        support,supporting_indexes = [],[]
        for org in self.osent:
            org_supports = []
            for sent in org:
                org_supports.append(1-fastMemLev(words,sent,self.w2v))
            a = np.argmax(org_supports)
            support.append(org_supports[a])
            supporting_indexes.append(a)
        self.support_avg.append(np.mean(support))
        self.support_std.append(np.std(support))
        self.all_supports.append(support)
    def recursive_mine(self, Z, words, level):
        p = self.gru.model.predict(Z)
        a = np.argsort(p[0])[::-1][:self.fillers]
        p2 = p[0][a]
        diffs = [p2[i] - p2[i+1] for i in range(len(p2)-1)]
        ka,i = [a[0]],0
        while (i < len(diffs)) and (diffs[i] <= self.thresh):
            ka.append(a[i+1])
            i += 1
        n,l,w = np.shape(Z)
        self.open_cases += len(ka)
        for i in range(len(ka)):
            word = self.i2w[ka[i]]
            if word == "SENTENCE_END":
                self.open_cases -= 1
                self.prepare(words)
            elif (level + 1) < self.mxsent_cut:
                Z2 = np.zeros((n,l,w), dtype = np.bool)
                Z2[0,l-level-1:l-1,:] = Z[0,l-level:l,:]
                Z2[0,-1,:] = idxtoonehot(ka[i], self.w, 'bool')
                self.recursive_mine(Z2, np.append(words, word), level+1)
            else:
                self.open_cases -= 1
    def mine_for_sentences(self, fillers = 8, thresh = 'max_init_diff',
                           mxsent_cut = 'Q3'):
        Z = np.zeros((1,self.mxsentlen, self.w), dtype = np.bool)
        n,l,w = np.shape(Z)
        Z[0,-1,:] = idxtoonehot(self.w2i["SENTENCE_START"], self.w, 'bool')
        if thresh == 'max_init_diff':
            init_pred = self.gru.model.predict(Z)
            a = np.argsort(init_pred[0])[::-1][:fillers]
            p = init_pred[0][a]
            diffs = [p[i] - p[i+1] for i in range(len(p)-1)]
            thresh = np.max(diffs)
            print("Threshold set to: "+str(thresh))
        if type(mxsent_cut) is str:
            lns = []
            for org in self.osent:
                for snt in org:
                    lns.append(len(snt))
            if mxsent_cut == 'Q3':
                mxsent_cut = np.percentile(lns, 75)
            elif (mxsent_cut == 'med') or (mxsent_cut == 'Q2'):
                mxsent_cut = np.median(lns)
            elif mxsent_cut == 'avg':
                mxsent_cut = np.mean(lns)
            else:
                print("Choosing best middle for max sentence cut.")
                mxsent_cut = max([np.mean(lns),np.median(lns)])
            print("Max Sentence Cut set to: "+str(mxsent_cut))
        self.mxsent_cut = mxsent_cut
        self.fillers = fillers
        self.thresh = thresh
        self.recursive_mine(Z, [], 1)
        print("\nMeaning: "+str(len(self.unjoined_sents)*\
                              len(self.sentences))+" iterations to perform!")
        amt = len(self.unjoined_sents)
        for words in self.unjoined_sents:
            print('\r'+str(amt)+' left to run editdistance on.', end = '')
            sys.stdout.flush()
            self.update_support(words)
            amt -= 1
    def save(self, filename):
        save_document_mine_results(self, filename)
    def load(self, filename):
        self = load_document_mine_results(self, filename)
        
def main(w2v, gru, filename = 'IRB_paragraph_data_modified.csv',
         fill = 8, t = 0.02, mxsent = 'Q3'):
    # IRB: t = 0.015, mxsent = 'med' and KL2. PFP and TL1: default.
    doc_mine = document_mining(filename)
    doc_mine.add_w2v(w2v)
    doc_mine.add_gru(gru)
    doc_mine.mine_for_sentences(fill, t, mxsent)
    return doc_mine

def existing_sentence_mine(doc_mine, mxsent_cut = 'med'):
    if type(mxsent_cut) is str:
        lns = []
        for org in doc_mine.osent:
            for snt in org:
                lns.append(len(snt))
        if mxsent_cut == 'Q3':
            mxsent_cut = np.percentile(lns, 75)
        elif (mxsent_cut == 'med') or (mxsent_cut == 'Q2'):
            mxsent_cut = np.median(lns)
        elif mxsent_cut == 'avg':
            mxsent_cut = np.mean(lns)
        else:
            print("Choosing best middle for max sentence cut.")
            mxsent_cut = max([np.mean(lns),np.median(lns)])
        print("Max Sentence Cut set to: "+str(mxsent_cut))
    good_sentences = []
    sent_org_indxes = []
    for o in range(len(doc_mine.osent)):
        for snt in doc_mine.osent[o]:
            if len(snt) <= mxsent_cut:
                good_sentences.append(snt)
                sent_org_indxes.append(o)
    print("Amount of sentences found: "+str(len(good_sentences)))
    all_support,avg_support,std_support,sentences = [],[],[],[]
    amt = len(good_sentences)
    for i in range(len(good_sentences)):
        sent = good_sentences[i]
        print('\r'+str(amt)+' of editdistances left to perform!',end='')
        amt -= 1
        sys.stdout.flush()
        sentences.append(' '.join(sent))
        support = []
        for o in range(len(doc_mine.osent)):
            if o == sent_org_indxes[i]:
                support.append(1.0)
                continue
            org_support = []
            for sent2 in doc_mine.osent[o]:
                org_support.append(1-fastMemLev(sent,sent2,doc_mine.w2v))
            support.append(np.max(org_support))
        all_support.append(support)
        avg_support.append(np.mean(support))
        std_support.append(np.std(support))
    doc_mine.within_sentences = sentences
    doc_mine.within_support = all_support
    doc_mine.within_avg_support = avg_support
    doc_mine.within_std_support = std_support

def save_document_mine_results(doc_mine, filename):
    attributes = [doc_mine.within_sentences,doc_mine.within_support,
                  doc_mine.within_avg_support,doc_mine.within_std_support,
                  doc_mine.generated_sentences,doc_mine.all_supports,
                  doc_mine.support_avg,doc_mine.support_std]
    with open(filename, 'wb') as output:
        pickler = pickle.Pickler(output, -1)
        pickler.dump(attributes)
    output.close()

def load_document_mine_results(doc_mine, filename):
    with open(filename, 'rb') as load:
        attributes = pickle.load(load)
    [doc_mine.within_sentences,doc_mine.within_support,
     doc_mine.within_avg_support,doc_mine.within_std_support,
     doc_mine.generated_sentences,doc_mine.all_supports,
     doc_mine.support_avg,doc_mine.support_std] = attributes
    load.close()
    return doc_mine

def support_plot(doc_mine, title):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    w = ax.scatter(doc_mine.within_avg_support,doc_mine.within_std_support,
               color='blue')
    g = ax.scatter(doc_mine.support_avg,doc_mine.support_std,color='orange')
    Artists,Labels = [w,g],['Pre-existing','Generated']
    ax.set_xlabel("Average Support", fontsize=14)
    ax.set_ylabel("Standard Deviation of Support", fontsize = 14)
    ax.legend(Artists, Labels)
    ax.set_title(title, fontsize=14)
    plt.show()
