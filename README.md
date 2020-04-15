# Keywords extraction using customized version of TF-IDF

The objective is to extract keywords from ranked passages. To obtain ranked passages refer to [this repository](https://github.com/airKlizz/MsMarco) (in the bonus notebook see ``best_passage`` function and change the ``return`` line with ``return dict(zip_scores_passages)``).

## Usage

```python
'''
example.py
'''

# import
from tfidf.utils import get_tfidf

# parameters
passages_path = 'passages_example/livestream_during_quarantine'

# read examples passages and create scores_passages
scores_passages = {}
with open(passages_path, 'r') as f:
    for line in f.readlines():
        elems = line.split('\t')
        scores_passages[float(elems[0])] = elems[1]

# compute the customized TF-IDF
tfidf = get_tfidf(scores_passages)

# display the top 10 keywords with tf-idf score
for i, k in enumerate(sorted(tfidf, key=tfidf.get, reverse=True)):
    if i > 10: break
    print('{} - tfidf: {:.4f}'.format(k, tfidf[k]))
    
'''
output:
quarantine - tfidf: 0.1889
artist - tfidf: 0.1315
livestream - tfidf: 0.1275
facebook - tfidf: 0.0977
concert - tfidf: 0.0785
fan - tfidf: 0.0745
stream - tfidf: 0.0645
twitch - tfidf: 0.0633
live - tfidf: 0.0623
est - tfidf: 0.0615
'''
```


## Customized TF-IDF

This customized version of TF-IDF allows to find words which are more present in the passages than in other english texts. 

### TF

TF is the weighted average of the frequency value of the term in the passage by the ranking score of the passage.

<img src="https://latex.codecogs.com/gif.latex?tf(t)&space;=&space;\frac{\sum_{p&space;\in&space;P}s_p*\frac{f_{t,&space;p}}{\sum_{t'&space;\in&space;p}&space;f_{t',&space;p}}}{\sum_{p&space;\in&space;P}s_p}" title="tf(t) = \frac{\sum_{p \in P}s_p*\frac{f_{t, p}}{\sum_{t' \in p} f_{t', p}}}{\sum_{p \in P}s_p}" />

*with P all passages, s<sub>p</sub> the ranking score of the passage p, f<sub>t, p</sub> the number of term t in the passage p.*

### IDF

IDF is the inverse frequency of words in english texts from [wikipedia](https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists#TV_and_movie_scripts).

<img src="https://latex.codecogs.com/gif.latex?idf(t)&space;=&space;\log(\frac{\sum_{t'&space;\in&space;T}f_{t'}}{f_t}*\sharp_{words\_per\_passage})" title="idf(t) = \log(\frac{\sum_{t' \in T}f_{t'}}{f_t}*\sharp_{words\_per\_passage})" />

*with T all english terms, f<sub>t</sub> the frequency of t, #<sub>words_per_passage</sub> the average number of words per passage (this value makes it possible to compute the probability that the term t appears in the passage).*

### TF-IDF

Finally the customized TF-IDF is the multiplication of TF and IDF.

## License
[MIT](https://choosealicense.com/licenses/mit/)
