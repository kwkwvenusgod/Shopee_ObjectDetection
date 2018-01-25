import os
import cv2


def load(root_path_list=['deepfashion/consumer-to-shop']):
    all_imgs = []

    classes_count = {}

    class_mapping = {}

    trainval_files = []
    test_files = []

    for root_path in root_path_list:
        annotation_path = root_path + '/Anno'
        bbox_path = annotation_path + '/list_bbox_consumer2shop.txt'
        with open(bbox_path, 'rb') as bbox_file:
            bbox_list = bbox_file.readlines()
            bbox_list = [bbox.rstrip('\n') for bbox in bbox_list]

        eval_path = root_path + '/Eval/list_eval_partition.txt'
        with open(eval_path, 'rb') as eval_file:
            eval_list = eval_file.readlines()
            eval_list = [eval_elem.rstrip('\n') for eval_elem in eval_list]

        train_test_list = parse_train_test_list(eval_list)

        for i in range(2, int(bbox_list[0])):

            bbox_item = bbox_list[i]
            bbox_item = bbox_item.split()

            image_path = os.path.join(root_path, bbox_item[0])
            bbox_category = bbox_item[0].split('/')[2]

            x1 = int(round(float(bbox_item[3])))
            y1 = int(round(float(bbox_item[4])))
            x2 = int(round(float(bbox_item[5])))
            y2 = int(round(float(bbox_item[6])))

            im = cv2.imread(os.path.join(image_path))
            im_height, im_width = im.shape[:2]

            annotation_data = {'filepath': image_path, 'width': im_width,
                               'height': im_height, 'bboxes': []}

            if train_test_list[bbox_item[0]] == 'test':
                annotation_data['imageset'] = 'test'
                test_files.append(image_path)
            else:
                annotation_data['imageset'] = 'trainval'
                trainval_files.append(image_path)

            annotation_data['bboxes'].append(
                {'class': bbox_category, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': False})

            if bbox_category not in classes_count:
                classes_count[bbox_category] = 1
            else:
                classes_count[bbox_category] += 1

            if bbox_category not in class_mapping:
                class_mapping[bbox_category] = len(class_mapping)

            all_imgs.append(annotation_data)
    return all_imgs, classes_count, class_mapping


def parse_train_test_list(eval_list):
    res = {}
    for i in range(2, len(eval_list)):
        eval_item = eval_list[i].split()

        if len(eval_item) == 4:
            decision = eval_item[3]
            res.update({eval_item[0]:decision})
            res.update({eval_item[1]:decision})
    return res
