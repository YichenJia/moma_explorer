from cnn import *
from metadata import *
from baseline import *
import json
import matplotlib.pyplot as plt

def plot_neighbors(folder_path,image_name):
    all_images,images_info = load_img_from_folder(folder_path)
    print("number of candidates:")
    print(len(images_info))

    image_path = folder_path + "/" + image_name
    tar_image = load_single_img(image_path)
    all_paths = convert_name_to_path(folder_path,images_info)

    model = load_model()

    k = 10
    idx_l2, idx_cosine = model_predict_method(model,tar_image,all_images,k)

    idx_loss = style_feature_method(image_path,all_paths,k)
    print(idx_loss)

    l2_cand = []
    cosine_cand = []
    style_cand = []

    for i in range(k):
        l2_cand.append(images_info[idx_l2[i]])
        cosine_cand.append(images_info[idx_cosine[i]])
        style_cand.append(images_info[idx_loss[i]])

    # print("l2_cand: ")
    # print(l2_cand)
    # print("cosine_cand: ")
    # print(cosine_cand)
    # print("style_cand: ")
    # print(style_cand)

    result = {}
    neighbors = {}
    neighbors["l2"] = l2_cand
    neighbors["cosine"] = cosine_cand
    neighbors["style"] = style_cand
    result[image_name] = neighbors

    print("RESULT:")
    print(result)

    # write results to a json file
    with open(image_name.split('.')[0] + '_result.json','w') as fp:
        json.dump(result,fp)

def plot_neighbors_for_folder(folder_path,json_file_path,k):
    all_images,images_info = load_img_from_folder(folder_path)
    print("number of candidates:")
    print(len(images_info))
    all_paths = convert_name_to_path(folder_path,images_info)
    # path here is in different order than the files in folder, use this order as index for all the following code
    result = {}

    for i in range(len(all_images)):
        print(i)
        image_id = images_info[i]['name']
        image_name = image_id+'.jpg'
        print("running function on "+image_name)
        image_path = folder_path + "/" + image_name
        tar_image = load_single_img(image_path)

        idx_l2, idx_cosine = model_predict_method(tar_image,all_images,k)
        idx_loss = style_feature_method(image_path,all_paths,k)
        idx_desc = metadata_method(image_id,images_info,json_file_path,k)
        idx_baseline = encoding_method(tar_image,all_images,k)

        l2_cand = []
        cosine_cand = []
        style_cand = []
        desc_cand = []
        baseline_cand = []

        for i in range(k):
            l2_cand.append(images_info[idx_l2[i]])
            cosine_cand.append(images_info[idx_cosine[i]])
            style_cand.append(images_info[idx_loss[i]])
            desc_cand.append(images_info[idx_desc[i]])
            baseline_cand.append(images_info[idx_baseline[i]])

        neighbors = {}
        neighbors["l2"] = l2_cand
        neighbors["cosine"] = cosine_cand
        neighbors["style"] = style_cand
        neighbors["desc"] = desc_cand
        neighbors["baseline"] = baseline_cand

        print(image_id)
        print(neighbors)

        result[image_id] = neighbors
        print(result)

    print("SAVING RESULT...")
    # print(len(result))
    print(result)
    # write results to a json file
    with open('recommender_result_5_selected.json','w') as fp:
        json.dump(result,fp)

    print('file written as recommender_result.json')

    def plot_result(result_dict):
        pass

if __name__ == "__main__":
    folder_path = "selected_MOMA_5"
    json_file_path = "filtered_moma_desc_checkpoint"
    k = 5
    plot_neighbors_for_folder(folder_path,json_file_path,k)
