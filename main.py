from cnn import *
import json
import matplotlib.pyplot as plt

def plot_neighbors(folder_path,image_name):
    all_images,images_info = load_img_from_folder(folder_path)
    print("number of candidates:")
    print(len(images_info))

    image_path = folder_path + "/" + image_name
    tar_image = load_single_img(image_path)
    all_paths = read_all_path_from_folder(folder_path)

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

    #plot all the images
    # w=10
    # h=10
    # fig=plt.figure(figsize=(8, 8))
    # columns = 4
    # rows = 5
    # for i in range(1, columns*rows +1):
    #     img = np.random.randint(10, size=(h,w))
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(img)
    # plt.show()

if __name__ == "__main__":
    folder_path = "test_MOMA"
    image_name = "179238.jpg"
    plot_neighbors(folder_path,image_name)
