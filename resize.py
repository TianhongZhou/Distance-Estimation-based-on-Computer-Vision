import os
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import xml.dom.minidom as xmldom
import numpy as np
import cv2

matplotlib.use('TkAgg')

image_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\faster_rcnn\data\VOCdevkit\VOC2007\JPEGImages"
image_resized_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\faster_rcnn\data\VOCdevkit\VOC2007\JPEGImages_1"
label_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\faster_rcnn\data\VOCdevkit\VOC2007\Annotations"
label_resized_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\faster_rcnn\data\VOCdevkit\VOC2007\Annotations_1"
image_root = os.walk(image_path)
label_root = os.walk(label_path)

new_width = 512
new_height = 512


def get_number(name):
    for i in range(len(name)):
        if name[i] == ".":
            return name[:i]


def image_resize(name):
    image = Image.open(image_path + "/" + name + ".jpg")
    image_matrix = np.array(image)
    h = image_matrix.shape[0]
    w = image_matrix.shape[1]
    image_resized = image.resize((new_width, new_height))
    image_resized_matrix = np.array(image_resized)
    cv2.imwrite(image_resized_path + "/" + name + ".jpg", image_resized_matrix)
    return h, w


def xml_rewrite(name, h, w):
    file_path = label_path + "/" + name + ".xml"
    xml_file = xmldom.parse(file_path)
    elements = xml_file.documentElement

    elements.getElementsByTagName("width")[0].firstChild.data = new_width
    elements.getElementsByTagName("height")[0].firstChild.data = new_height

    for node in elements.getElementsByTagName("xmin"):
        node.firstChild.data = float(node.firstChild.data) / w * new_width

    for node in elements.getElementsByTagName("xmax"):
        node.firstChild.data = float(node.firstChild.data) / w * new_width

    for node in elements.getElementsByTagName("ymin"):
        node.firstChild.data = float(node.firstChild.data) / h * new_height

    for node in elements.getElementsByTagName("ymax"):
        node.firstChild.data = float(node.firstChild.data) / h * new_height

    save_path = label_resized_path + "/" + name + ".xml"
    with open(save_path, 'wb') as f:
        f.write(xml_file.toprettyxml(encoding='utf-8'))


for path, dir_list, file_list in image_root:
    for file_name in file_list:
        image_name = get_number(file_name)
        height, width = image_resize(image_name)
        xml_rewrite(image_name, height, width)
