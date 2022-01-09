from google.cloud import *
from inverted_index_gcp import *
from pathlib import Path
from contextlib import closing
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import os
import re
from operator import itemgetter
import functools
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from builtins import *
import pickle


def read_pickle(file_name):
    stream = open(f'postings_gcp/{file_name}.pkl')
    pick = pickle.load(stream)
    stream.close()
    print(f'{file_name} loaded')
    return pick

def get_title_by_doc_id(doc_id):
    try:
        return doc_id_to_title_dic[doc_id]
    except:
        return "Invalid Title!"
        
print("Start loading...")
inverted_title = read_pickle("title2")
inverted_anchor = read_pickle("anchor_fix")
page_rank_dict = read_pickle("page_rank_dict")
page_view_dict = read_pickle("page_view")
inverted_body = read_pickle("text2")
doc_id_norm_dict = read_pickle("doc_id_norm_dict")
doc_id_to_title_dic = read_pickle("doc_id_to_title_dict")
print("Done!")


TUPLE_SIZE = 6       
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
def read_posting_list(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list


nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]  # TODO: calculate the corups stop words words
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stemmer = PorterStemmer()
all_stopwords = english_stopwords.union(corpus_stopwords)

def tokenize(text, stem=False):
  """
    This function aims in tokenize a text into a list of tokens.
    Moreover:
    * filter stopwords.
    * change all to lowwer case.
    * use stemmer
    
    Parameters:
    -----------
    text: string , represting the text to tokenize.    
    
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
  clean_text = []

  text = text.lower()
  tokens = [token.group() for token in RE_WORD.finditer(text)]
  for token in tokens:
    if token not in all_stopwords:
      if stem:
        token = stemmer.stem(token)
      clean_text.append(token)
  return clean_text

import math
from itertools import chain

def get_posting_gen(index, query):
    """
    This function returning the generator working with posting list.
    
    Parameters:
    ----------
    index: inverted index    
    """
    words = []
    pls = []
    for term in query:
        try:
            pls = read_posting_list(index, term)
        except:
            pls = []
        yield term, pls

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.    
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self,index,k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.AVGDL = sum(index.DL.values())/self.N 

    def calc_idf(self,list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        
        Returns:
        -----------
        idf: dictionary of idf scores. As follows: 
                                                    key: term
                                                    value: bm25 idf score
        """        
        idf = {}        
        for term in list_of_tokens:            
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass                             
        return idf
        

    def search(self, query,N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query. 
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        
        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        query = tokenize(query)
        candidate = set([])
        for w, pls in get_posting_gen(self.index, query):
            for doc_id, frq in pls:
                candidate.add(doc_id)
        self.idf = self.calc_idf(query)
        results = self._score(query, candidate)
        query_top_n = sorted(results, key = lambda x: x[1], reverse=True)[:N]
        return query_top_n
        # YOUR CODE HERE

    def _score(self, query, candidates):
        """
        This function calculate the bm25 score for given query and document.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        
        Returns:
        -----------
        score: float, bm25 score.
        """
        score_ret = {}
        for w, pls in get_posting_gen(self.index, query):
            term_frequencies = dict(pls)
            for doc_id in candidates:
                score = 0.0
                doc_len = self.index.DL[doc_id]
                if doc_id in term_frequencies:
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[w] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
                score_ret[doc_id] = score_ret.get(doc_id, 0) + score
        return list(score_ret.items())

def merge_results(title_scores,body_scores,title_weight=0.5,text_weight=0.5,N = 100):    
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body). 

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows: 
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
                
    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows: 
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function. 
    
    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score). 
    """
    temp_dict = {}
    for doc_id, score in title_scores:
        temp_dict[doc_id] = title_weight * score
    for doc_id, score in body_scores:
        if temp_dict.get(doc_id) is None:
            temp_dict[doc_id] = text_weight * score
        else:
            temp_dict[doc_id] += text_weight * score
    return temp_dict

import threading
import queue

w_title = 0.25
w_text = 1 - w_title

bm_weight = 0.5
page_view_weight = 0.25
page_rank_weight = 0.25

def thread_bm25(index, query, queue):
    bm25_t = BM25_from_index(index)
    bm25_queries_score_train_t = bm25_t.search(query, N=50)
    queue.put(bm25_queries_score_train_t)

def get_bm25(query):
    queue_title = queue.Queue()
    queue_body = queue.Queue()
    t_title = threading.Thread(target=thread_bm25, args=(inverted_title, query, queue_title))
    t_body = threading.Thread(target=thread_bm25, args=(inverted_body, query, queue_body))
    t_title.start()
    t_body.start()
    t_title.join()
    t_body.join()
    bm25_queries_score_train_title = queue_title.get()
    bm25_queries_score_train_body = queue_body.get()
    BM25_score = merge_results(bm25_queries_score_train_title,bm25_queries_score_train_body,w_title,w_text)
    return BM25_score

def add_page_rank_and_view(dic):
    max_bm25 = 0
    max_page_rank = 0
    max_page_view = 0
    for key in dic:
        bm = dic[key]
        page_rank = page_rank_dict[key][0]
        page_view = page_view_dict[key]
        if bm > max_bm25:
            max_bm25 = bm
        if page_rank > max_page_rank:
            max_page_rank = page_rank
        if page_view > max_page_view:
            max_page_view = page_view
            
    for key in dic:
        bm = dic[key]
        page_rank = page_rank_dict[key][0]
        page_view = page_view_dict[key]
        dic[key] = round((bm * bm_weight / max_bm25) + (page_rank * page_rank_weight / max_page_rank) + (page_view * page_view_weight / max_page_view), 5)
    return dic

def search_procedure(query):
    try:
        BM25 = get_bm25(query)
        calculated = add_page_rank_and_view(BM25)
        sor = list(sorted([(doc_id, calculated[doc_id]) for doc_id in calculated], key = lambda x: x[1], reverse=True)[:100])
        res = map(lambda x: (x[0], get_title_by_doc_id(x[0])), sor)
        return list(res)
    except Exception as e:
        print(f'Error - {e}')
        return []

from builtins import *
import math
  
def get_top_n(sim_dict,N=3):
    """ 
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores 
   
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3
    
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    lst = [(doc_id,round(score,5)) for doc_id, score in sim_dict.items()]
    srot = sorted(lst, key = lambda x: x[1],reverse=True)
    return srot[:N]

def get_posting_gen(index, query):
    """
    This function returning the generator working with posting list.
    
    Parameters:
    ----------
    index: inverted index    
    """
    words = []
    pls = []
    for term in query:
        try:
            pls = read_posting_list(index, term)
        except:
            pls = []
        yield term, pls

def norm_query(query_counter):
    c = 0
    for key in query_counter:
        c+=query_counter[key]**2
    return (1/math.sqrt(c))

def similarity(query_to_search,index,N=3):
    new_query = tokenize(query_to_search)
    query_counter={}
    for i in new_query:
        if i not in query_counter:
            query_counter[i] = 0
        query_counter[i]+=1
    generator = get_posting_gen(index, list(set(new_query)))
    sim_dict = {}
    for word,pls in generator:
        for doc_id,weight in pls:
            if doc_id not in sim_dict:
                sim_dict[doc_id] = 0
            sim_dict[doc_id] = sim_dict[doc_id] +query_counter[word]*weight
    for key in sim_dict:
        sim_dict[key] = sim_dict[key]*(norm_query(query_counter))*(doc_id_norm_dict[doc_id])
    return get_top_n(sim_dict,N)

def search_body_procedure(query):
    try:
        cos = similarity(query,inverted_body,N = 100)
        cos = map(lambda x: (x[0], get_title_by_doc_id(x[0])), cos)
        return list(cos)
    except Exception as e:
        print(f'Error!!! - {e}')
        return []

def search_title_procedure(query):
    query = tokenize(query)
    results = []
    for term in query:
        try:
            results.append(read_posting_list(inverted_title, term))
        except:
            print("Term not in inverted_title: " + term)
            pass
    
    if len(results) != 0:
        results = functools.reduce(lambda a, b: a+b, results)
        results = map(lambda x: x[0], results)
        counter = Counter()
        counter.update(results)
        results = map(lambda x: (x[0], get_title_by_doc_id(x[0])), counter.most_common())

    return list(results)


def search_anchor_procedure(query):
    query = tokenize(query)
    results = []
    for term in query:
        try:
            results.append(read_posting_list(inverted_anchor, term))
        except:
            print("Term not in inverted_anchor: " + term)
            pass
        
    if len(results) != 0:
        results = functools.reduce(lambda a, b: a+b, results)
        results = map(lambda x: x[0], results)
        counter = Counter()
        counter.update(results)
        results = map(lambda x: (x[0], get_title_by_doc_id(x[0])), counter.most_common())
    return list(results)


def page_rank_procedure(wiki_ids):
    res = []
    for doc_id in wiki_ids:
        try:
            res.extend(page_rank_dict[doc_id])
        except:
            print("doc_id not in page_rank_dict: " + str(doc_id))
            res.append(0)
    return res


def page_view_procedure(lst):
    res = []
    for doc_id in lst:
        try:
            res.append(page_view_dict[doc_id])
        except:
            print("doc_id not in page_view: " + str(doc_id))
            res.append(0)
    return res


from flask import Flask, request, jsonify

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = search_procedure(query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = search_body_procedure(query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = search_title_procedure(query)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = search_anchor_procedure(query)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json(force=True, silent=True, cache=False)
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res =  page_rank_procedure(wiki_ids)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json(force=True, silent=True, cache=False)
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = page_view_procedure(wiki_ids)
    # END SOLUTION
    return jsonify(res)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)