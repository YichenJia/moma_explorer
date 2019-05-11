import json
import pandas as pd
import numpy as np
from scipy import spatial

def encoding_method(tar_image,all_images,k):
    print("---- EXTRACT FEATURE VECTORS FROM PIXEL COLORS ----")
    all_cdis = []
    for cand_image in all_images:
        all_cdis_for_i = []
        # print(np.array(cand_image).shape)
        for i in range(len(cand_image)):
            for j in range(len(cand_image[0])):
                cdis = spatial.distance.cosine(tar_image[i][j], cand_image[i][j])
                all_cdis_for_i.append(cdis)

        avg_cdis = sum(all_cdis_for_i)/len(all_cdis_for_i)
        all_cdis.append(avg_cdis)

    cdis_array = np.array(all_cdis)
    idx_baseline = np.argsort(cdis_array)[:k]
    # print(idx_baseline)
    return idx_baseline
