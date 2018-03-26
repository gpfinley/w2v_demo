from __future__ import print_function
import codecs
import numpy as np
import struct
import re

ENCODING='latin_1'

def import_w2v(vectors_path, vocab_path=None, filter_re=None, stop_after=None, normalize=True):

    with codecs.open(vectors_path, 'r', encoding=ENCODING) as f:
        header = ''
        c = ''
        while c != '\n':
            c = f.read(1)
            header += c
        header = header.split()
        nwords = int(header[0])
        dim = int(header[1])
        if stop_after is not None:
            print('reading', stop_after, 'of', nwords, 'in bin file, with', dim, 'dimensions...')
        else:
            print('reading', nwords, 'in bin file, with', dim, 'dimensions...')
        dic = {}
        if stop_after is None:
            emb = np.zeros((nwords, dim))
        else:
            emb = np.zeros((stop_after, dim))
        lastchar = ''
        nextind = 0
        for wordnum in range(nwords):
            if stop_after and nextind >= stop_after:
                break
            thisword = lastchar.strip()
            while True:
                nextchar = f.read(1)
                if nextchar == ' ': break
                thisword += nextchar
                if len(thisword) > 100:
                    print(thisword)
            vector = f.read(4 * dim)
            vector = np.transpose(np.array([struct.unpack('f', bytearray(vector[4 * i:4 * (i + 1)], encoding=ENCODING)) for i in range(dim)]))
            if not filter_re or re.match(filter_re, thisword):
                dic[thisword] = nextind
                emb[nextind] = vector
                nextind += 1
            # todo: fix this in case newlines are NOT in the file (like the pre-trained vectors)
            lastchar = f.read(1)
    emb = emb[:nextind,:]
    print('read', nextind, 'words')
    freq = None
    if vocab_path:
        freq = {}
        for line in open(vocab_path):
            fields = line.split()
            if len(fields) > 1:
                freq[fields[0]] = int(fields[1])
    if normalize:
        emb = normalize_all(emb)
    return emb, dic, freq

def normalize_all(emb):
    return (emb.T / np.linalg.norm(emb, axis=1)).T
