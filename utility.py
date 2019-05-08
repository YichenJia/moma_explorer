import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re

def filter_data():
    """
    Helper function that helps filter the data based on classification and if there
    is valid URLs. This is function is only called when preparing dataset for training.
    """

    with open('moma_artworks_s.json') as json_file:
        data = json.load(json_file)
    print("total length: "+str(len(data)))
    df = pd.DataFrame(data)
    # print(df.head())
    # df = df.groupby('Classification')['ObjectID'].nunique()
    # print(df)
    # df = df.loc[df['Classification'] == 'Installation']

    # filter classifications
    filtered_list = ['Audio', 'Design', 'Furniture and Interior', 'Installation',
        'Media', 'Multiple', 'Performance', 'Product Design', 'Sculpture', 'Software',
        'Textile', 'Vehicle', 'Video']
    filtered_df = df[~df['Classification'].isin(filtered_list)]

    # filter null thumbnail and URLs
    filtered_df = df[df['ThumbnailURL'].notnull()]
    filtered_df = df[df['URL'].notnull()]

    print(filtered_df.shape)
    filtered_data = filtered_df.to_dict('records')
    # filtered_df.to_json("moma_artworks_filter.json",orient='index')

    with open('moma_artworks_filter.json','w') as fp:
        json.dump(filtered_data,fp)

def download_image_url():
    """
    Helper function that downloads images through urls in the dataset. This function only
    run once at the beginning to download data.
    """

def get_desc(url):
    page = urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    div = soup.find('div', class_="main-content")
    desc = div.text
    # print(repr(desc))
    return desc

def download_descriptions():
    """
    Helper function that scrape descriptions of artworks through url. This function only
    run once at the beginning to construct the dataset.
    """
    with open('moma_artworks_filter.json') as json_file:
        data = json.load(json_file)
    print(data[0])
    for i in range(len(data)):
        print(i)
        if i <= 100:
            item = data[i]
            url = item['URL']
            desc = get_desc(url)
            item["desc"] = desc

    with open('moma_artworks_desc.json','w') as fp:
        json.dump(data,fp)

if __name__ == "__main__":
    # filter_data()
    # test_url = "https://www.moma.org/collection/works/800"
    # get_desc(test_url)
    # filter_data()
    download_descriptions()
