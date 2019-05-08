import json
import pandas as pd
import numpy as np
import re
from utility import get_desc
from nltk.metrics import *

# def compute_jaccard_distance(tar,cand):
#     """
#     Compute jaccard distance between the words used in titles of one target and one candidate
#     """
#

def compute_year_distance(tar,cand):
    """
    Compute distance between two years, normalized the distance to 0-1
    """

    pass

def word_of_bag(str):
    """
    Turn a title in string format into a list of words
    """
    word_list = re.sub("[^\w]", " ",  str).split()
    return word_list

def metadata_method(tar,k):
    """
    Return the k-closest artworks' index based on textual meaning from descriptions
    """
    # read dataset
    with open('moma_artworks_filter.json') as json_file:
        data = json.load(json_file)

    print("total length: "+str(len(data)))
    # print(data.keys())
    # print(data['2'])
    distance = []
    tar_desc = tar["desc"]
    tar_list = word_of_bag(tar_desc)
    for cand in data:
        cand_desc = cand["desc"]
        cand_list = word_of_bag(cand_desc)
        dis = jaccard_distance(tar_list, cand_list)
        distance.append(dis)

    dis_array = np.array(distance)
    idx_dis = np.argsort(dis_array)[:k]
    return idx_dis

if __name__ == "__main__":
    test_url = "https://www.moma.org/collection/works/800"
    test_str = get_desc(test_url)
    list = word_of_bag(test_str)
    print(list)
