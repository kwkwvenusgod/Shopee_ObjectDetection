import os


def load(root_path='deepfashion/consumer-to-shop'):
    annotation_path = root_path + '/Anno'
    bbox_path = annotation_path + '/list_bbox_consumer2shop.txt'
    with open(bbox_path, 'rb') as bbox_file:
        bbox_list = bbox_file.readlines()
        bbox_list = [bbox.rstrip('\n') for bbox in bbox_list]

    eval_path = root_path + '/Eval/list_eval_partition.txt'
    with open (eval_path, 'rb') as eval_file:
        eval_list = eval_file.readlines()
        eval_list = [eval_elem.rstrip('\n') for eval_elem in eval_list]

    all_imgs = []

    classes_count = {}

    class_mapping = {}

    trainval_files = []
    test_files = []

    for i in range(2, bbox_list[0]):
        bbox_item = bbox_list[i]
        bbox_item = bbox_item.split()

        image_path = os.path.join(root_path, bbox_item[0])
        bbox_category = image_path.split('/')[2]

        x1 = bbox_item[3]
        y2 = bbox_item[4]
        x2 = bbox_item[5]
        y2 = bbox_item[6]













    return