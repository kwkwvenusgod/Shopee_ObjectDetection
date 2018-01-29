import load_artelab_data
import load_deepfashion_data


def load(dataset_dict):
    all_imgs = []

    classes_count = {}

    class_mapping = {}


    for type in dataset_dict.keys():
        if type == 'artelab':
            all_imgs_tmp, classes_count_tmp, class_mapping_tmp = load_artelab_data.load(dataset_dict[type])
            all_imgs.extend(all_imgs_tmp)
            classes_count.update(classes_count_tmp)
            class_mapping.update(class_mapping_tmp)
        elif type == 'deepfashion':
            all_imgs_tmp, classes_count_tmp, class_mapping_tmp = load_deepfashion_data.load(dataset_dict[type])
            all_imgs.extend(all_imgs_tmp)
            classes_count.update(classes_count_tmp)
            class_mapping.update(class_mapping_tmp)

    return all_imgs, classes_count, class_mapping


def get_n_batch(n, data_gen):
    X = []
    Y = []
    image_aug = []

    for i in range(n):
        Xtmp, Ytmp, img_data_tmp = next(data_gen)
        X.append(Xtmp)
        Y.append(Ytmp)
        image_aug.append(img_data_tmp)

    return X, Y, image_aug
