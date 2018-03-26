from __future__ import print_function
import ops
from w2v_import import import_w2v

n_results = 20
vocab_size = 200000

# For python 2/3 compatibility
try:
    input = raw_input
except:
    pass

emb, vocab, freq = import_w2v('./GoogleNews-vectors-negative300.bin', stop_after=vocab_size)

while True:
    word = input('Enter a word: ')
    if not len(word): break
    if word in vocab:
        print()
        for w, s in ops.sorted_similar_words(emb, vocab, word, n_results):
            print(w + ((30-len(w))*' ') + str(s))
    else:
        print('Word not in vocabulary.')
    print('\n\n')
