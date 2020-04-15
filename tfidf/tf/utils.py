import sys
sys.path.append('../..')
from tfidf.processing.utils import tokenize_preprocess

def simple_tf(passage):
    terms = tokenize_preprocess(passage)
    tf = {}
    for term in terms:
        if term not in tf.keys():
            tf[term] = terms.count(term)/len(terms)
    return tf

def weighted_tf(scores_passages):
    tf = {}
    for weight, passage in scores_passages.items():
        for term, score in simple_tf(passage).items():
            if term not in tf.keys():
                tf[term] = weight * score
            else:
                tf[term] += weight * score
    
    total = sum(tf.values(), 0.0)
    tf = {k: v / total for k, v in tf.items()}
    return tf
            

