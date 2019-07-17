from typing import Any, Iterable, List, Optional, Set, Tuple
import time
from load import load_words
import math
import vectors as v
from vectors import Vector
from word import Word
import json
import multiprocessing as mp
import threading


def most_similar(base_vector: Vector, words: List[Word]) -> List[Tuple[float, Word]]:
    """Finds n words with smallest cosine similarity to a given word"""
    # distance=v.cosine_similarity_normalized(base_vector)
    words_with_distance = [(v.cosine_similarity_normalized(base_vector, w.vector), w) for w in words]
    # We want cosine similarity to be as large as possible (close to 1)
    sorted_by_distance = sorted(words_with_distance, key=lambda t: t[0], reverse=True)

    return sorted_by_distance


def check_key(key, dic):
    exist = False
    for k in dic:
        if k == key:
            exist = True
            break
    return exist


def load_test_set():  # key is the UUID, values are the concepts
    testdic = dict()
    with open("dicts\\en-es.5000-6500.txt", 'r', encoding='utf8') as f:
        line = f.readline()
        while line:
            wds = line.strip().split(' ')
            orgw = str(wds[0].strip())
            trgw = str(wds[1].strip())
            # print (orgw,trgw)
            if check_key(orgw, testdic):
                # print (check_key(orgw, testdic))
                testdic[orgw].add(trgw)
            else:
                # print (check_key(orgw, testdic))
                # testdic[orgw] = set()
                testdic[orgw] = {trgw}
            line = f.readline()
    return testdic


def List_most_similar(words: List[Word], text: str):
    print(text)
    base_word = find_word(text, words)
    if not base_word:
        # print(f"Uknown word: {text}")
        return
    print(f"Words related to {base_word.text}:")
    sorted_by_distance = [
        (dist, word.text) for (dist, word) in
        most_similar(base_word.vector, words)
        if word.text.lower() != base_word.text.lower()
    ]

    print(text, sorted_by_distance[:10])
    with open("similarword-ENES-clusterring50k.txt", 'a', encoding='utf8') as out:
        out.write("{0}\t{1}\n".format(text, sorted_by_distance[:10]))
    # return sorted_by_distance[:5]


def find_word(text: str, words: List[Word]) -> Optional[Word]:
    try:
        return next(w for w in words if text == w.text)
    except StopIteration:
        return None


if __name__ == "__main__":
    t0 = time.time()
    words = load_words('data\\final_en-es_clustering_50k.vec')
    print("######################## Analogies ##################")
    t1 = time.time()
    print("loading the model takes %f seconds" % (t1 - t0))
    testDic = load_test_set()
    originwords = list([*testDic])
    print(len(originwords))

    for wt in originwords:
        List_most_similar(words,wt)

    """
    t2 = time.time()
    threads = []
    for i in range(len(originwords)):
        p = threading.Thread(target=List_most_similar, args=(words, originwords[i],))
        threads.append(p)
        p.start()
    for one_thread in threads:
        one_thread.join()"""


    t3 = time.time()
    print("rest of the program takes %f seconds" % (t3 - t2))
    print("Done!")

    count = 0
    indexdic = []
