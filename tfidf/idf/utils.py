import sys
sys.path.append('../..')
import json
import numpy as np
from tfidf.processing.utils import preprocess


def create_words_frequency(input='words_frequency', output='words_frequency.json'):
    with open(input, 'r') as f:
        data = f.read()

    lines = data.split('\n')

    words_frequency = {}
    for line in lines:
        elems = line.split('\t')
        word = elems[1].split(' ')[0]
        word = preprocess(word, True)
        frequency = elems[2].split(' ')[0]
        if word in words_frequency.keys():
            words_frequency[word] += int(frequency)
        else:
            words_frequency[word] = int(frequency)
        
    with open(output, 'w') as fp:
        json.dump(words_frequency, fp)

def count_words(input='words_frequency.json'):
    with open(input, 'r') as f:
        words_frequency = json.load(f)
    return len(words_frequency)

class Idf():

    def __init__(self, input='words_frequency.json'):
        with open(input, 'r') as f:
            self.words_frequency = json.load(f)
        self.frequencies = np.array(list(self.words_frequency.values()))
        self.frequencies_sum = np.sum(self.frequencies)
        self.frequencies_min = np.min(self.frequencies)

    def get(self, word, num_words_per_docs=100):
        if word in self.words_frequency.keys():
            frequency = self.words_frequency[word]
        else:
            frequency = self.frequencies_min
        prob = min(1., num_words_per_docs * frequency / self.frequencies_sum)
        return np.log(1/prob)