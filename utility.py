import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import os

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

def download_descriptions_from_file_names(folder_path):
    all_names = []
    for img in os.listdir(folder_path):
        if img.split('.')[1] == 'jpg':
            name = img.split('.')[0]
            all_names.append(name)

    with open('moma_artworks_filter.json') as json_file:
        data = json.load(json_file)

    print('number of files inside the folder:')
    print(len(all_names))

    new_data = []
    null_data = []
    for i in range(23300,len(data)):
        item = data[i]
        # print(item['ObjectID'])
        if str(item['ObjectID']) in all_names:
            url = item['URL']
            desc = get_desc(url)
            if desc.startswith('\nIf you would like to reproduce an image of'):
                print('null: '+str(item['ObjectID']))
                null_data.append(item['ObjectID'])
            elif desc.startswith('\nMoMA collaborated with Google Arts & Culture Lab on a project'):
                print('null: '+str(item['ObjectID']))
                null_data.append(item['ObjectID'])
            elif desc == "404":
                print('404: '+str(item['ObjectID']))
                null_data.append(item['ObjectID'])
            elif len(desc) > 0:
                print('valid:     '+str(item['ObjectID'])+ ' at '+ str(round(i/len(data)*100)) + '%')
                new_item = {}
                new_item['ObjectID'] = item['ObjectID']
                new_item['URL'] = item['URL']
                new_item['ThumbnailURL'] = item['ThumbnailURL']
                new_item['desc'] = desc
                new_data.append(new_item)

            #save checkpoint
            if i/100 == i//100:
                print("saving at index" + str(i))
                with open('filtered_moma_desc_checkpoint.json','w') as fp:
                    json.dump(new_data,fp)
                print('----')
                print("file written as: filtered_moma_desc_checkpoint.json")
                print("valid items: "+str(len(new_data)))
                with open('filtered_moma_desc_null_checkpoint.json','w') as fp:
                    json.dump(null_data,fp)
                print("null items: "+str(len(null_data)))
                print('----')

    print(new_data[0].keys())
    print(len(new_data))

    with open('filtered_moma_desc.json','w') as fp:
        json.dump(new_data,fp)
    print("file written as: filtered_moma_desc.json")

    with open('filtered_moma_desc_null.json','w') as fp:
        json.dump(null_data,fp)
    return null_data

def get_desc(url):
    try:
        page = urlopen(url)
        soup = BeautifulSoup(page, 'html.parser')
        div = soup.find('div', class_="main-content")
        desc = div.text
    except:
        desc = "404"
        # print(repr(desc))
    return desc

def batch_delete_files(folder_path,null_data):
    for name in null_data:
        path = folder_path + '/' + str(name) + '.jpg'
        if os.path.exists(path):
            os.remove(path)
    print("finish removing files")

def download_descriptions():
    """
    Helper function that scrape descriptions of artworks through url. This function only
    run once at the beginning to construct the dataset.
    """
    with open('moma_artworks_filter.json') as json_file:
        data = json.load(json_file)
    print(data[0])

    new_data = []
    for i in range(len(data)):
        # print(i)
        if i <= 100:
            print(i)
            item = data[i]
            url = item['URL']
            desc = get_desc(url)
            if desc.startswith('\nIf you would like to reproduce an image of'):
                print('null')
            else:
                new_item = {}
                new_item['ObjectID'] = item['ObjectID']
                new_item['URL'] = item['URL']
                new_item['ThumbnailURL'] = item['ThumbnailURL']
                new_item['desc'] = desc
                new_data.append(new_item)

    print("filtering data")
    # col_dropped = ['Artist','AccessionNumber','BeginDate','Classification','Circumference (cm)','Date','DateAcquired',
    #                  'Department','Depth (cm)','Diameter (cm)','Duration (sec.)','EndDate','Gender',
    #                 'Height (cm)','Length (cm)','Medium','Nationality','ThumbnailURL','URL']
    # df = pd.DataFrame(data)
    # df.drop(['Artist'],axis=1)
    # df = df[df['desc'].notnull()]
    # print(df.head())
    # filtered_data = df.to_dict('records')
    print(new_data[0].keys())
    print(len(new_data))

    with open('moma_artworks_desc.json','w') as fp:
        json.dump(new_data,fp)
    print("file written as: moma_artworks_desc.json")

def desc_process():
    with open('filtered_moma_desc.json') as json_file:
        data = json.load(json_file)

    result = {}
    for item in data:
        result[item['ObjectID']] = item

    with open('filtered_moma_desc_dict.json','w') as fp:
        json.dump(result,fp)

if __name__ == "__main__":
    # filter_data()
    # test_url = "https://www.moma.org/collection/works/800"
    # get_desc(test_url)
    # filter_data()
    # folder_path = "MOMA_filtered"
    # null_data = download_descriptions_from_file_names(folder_path)
    # with open('filtered_moma_desc_null_checkpoint_14000.json') as json_file:
    #     null_data = json.load(json_file)
    # batch_delete_files(folder_path,null_data)
    desc_process()
