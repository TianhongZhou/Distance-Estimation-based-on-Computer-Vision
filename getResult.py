import xml.etree.ElementTree as ET
import os, cv2
import numpy as np

test_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\yolov5-master\my_data\ImageSets\Main" \
            r"\test.txt "
origin_image_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\yolov5-master\my_data\images"
label_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\yolov5-master\my_data\Annotations"
new_image_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\yolov5-master\runs\detect\exp2"
write_path = r"F:\document\University\Research\2022-IEMP-Bjorn\research\project\yolov5-master\result_compare"

with open(test_path) as f:
    lines = f.readlines()

for line in lines:
    line = line.replace('\n', '').replace('\r', '')
    xml_file = label_path + "/" + line + ".xml"
    tree = ET.parse(xml_file)
    root = tree.getroot()
    imgfile = origin_image_path + "/" + line + ".jpg"
    im = cv2.imread(imgfile)

    for o in root.findall("object"):
        object_name = o.find("name").text
        xmin = int(o.find("bndbox").find("xmin").text)
        ymin = int(o.find("bndbox").find("ymin").text)
        xmax = int(o.find("bndbox").find("xmax").text)
        ymax = int(o.find("bndbox").find("ymax").text)
        color = (0, 0, 255)
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, object_name, (xmin, ymin - 7), font, 1, color, 4)

    img2file = new_image_path + "/" + line + ".jpg"
    im2 = cv2.imread(img2file)
    final = np.hstack([im, im2])

    cv2.imwrite(write_path + "/" + line + ".jpg", final)
