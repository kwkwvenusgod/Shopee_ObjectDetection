import load_artelab_data
import load_deepfashion_data


def load(dataset_dict, size_lim=None):
    all_imgs = []

    classes_count = {}

    class_mapping = {}


    for type in dataset_dict.keys():
        if type == 'artelab':
            all_imgs_tmp, classes_count_tmp, class_mapping_tmp = load_artelab_data.load(dataset_dict[type],size_lim=size_lim)
            all_imgs.extend(all_imgs_tmp)
            classes_count.update(classes_count_tmp)
            class_mapping.update(class_mapping_tmp)
        elif type == 'deepfashion':
            all_imgs_tmp, classes_count_tmp, class_mapping_tmp = load_deepfashion_data.load(dataset_dict[type],size_lim=size_lim)
            all_imgs.extend(all_imgs_tmp)
            classes_count.update(classes_count_tmp)
            class_mapping.update(class_mapping_tmp)

    return all_imgs, classes_count, class_mapping
