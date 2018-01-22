import os
import xml.etree.ElementTree as ET

def load(data_path_list):

    all_imgs = []

    classes_count = {}

    class_mapping = {}

    trainval_files = []
    test_files = []

    visualise = False
    for data_path in data_path_list:
        annotaion_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')
        imgsets_path_train = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets', 'Main', 'test.txt')

        try:
            with open(imgsets_path_train) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.jpg')
        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip() + '.jpg')
        except Exception as e:
            print(e)

        annotations = [os.path.join(annotaion_path, s) for s in os.listdir(annotaion_path)]
        ind = 0
        for annotation in annotations:
            try:
                ind += 1
                et = ET.parse(annotation)
                element = et.getroot()

                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
                                       'height': element_height, 'bboxes': []}

                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'

                for obj in element_objs:
                    class_name = obj.find('name').text
                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1

                    if class_name not in class_mapping:
                        class_mapping[class_name] = len(class_mapping)

                    obj_bbox = obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    difficulty = int(obj.find('difficult').text) == 1
                    annotation_data['bboxes'].append(
                        {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
                    all_imgs.append(annotation_data)
            except Exception as e:
                print(e)


    return all_imgs, classes_count, class_mapping