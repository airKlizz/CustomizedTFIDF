from tfidf.utils import get_tfidf

passages_path = 'passages_example/livestream_during_quarantine'

scores_passages = {}
with open(passages_path, 'r') as f:
    for line in f.readlines():
        elems = line.split('\t')
        scores_passages[float(elems[0])] = elems[1]

tfidf = get_tfidf(scores_passages)

for i, k in enumerate(sorted(tfidf, key=tfidf.get, reverse=True)):
    if i > 40: break
    print('{} - tfidf: {:.4f}'.format(k, tfidf[k]))