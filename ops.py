import numpy as np

'''
Simple operations with vectors
'''

def sorted_similar_words(emb, vocab, word, top=0):
    if word not in vocab:
        return None
    if top <= 0:
        top = emb.shape[0]
    dd = distance_dict(emb, vocab, word)
    return sorted(dd.items(), key=lambda x: -x[1])[1:top+1]

def distance_dict(emb, vocab, word):
    if word not in vocab:
        return None
    vec = emb[vocab[word]]
    prods = np.dot(emb, vec)
    return {w: prods[vocab[w]] for w in vocab}
