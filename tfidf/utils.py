import sys
sys.path.append('..')
from tfidf.idf.utils import Idf
from tfidf.tf.utils import weighted_tf

def get_tfidf(scores_passages, input='tfidf/idf/words_frequency.json', num_words_per_docs=100):

    tf = weighted_tf(scores_passages)
    idf = Idf(input)

    tfidf = {}
    for term in tf.keys():
        tfidf[term] = tf[term] * idf.get(term, num_words_per_docs)

    return tfidf
    
