import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yolov5
from PIL import Image

from faster_rcnn.frcnn import FRCNN
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

image_root = os.walk(r"F:\document\University\Research\2022-IEMP-Bjorn\research\images")


def get_number(name):
    for i in range(len(name)):
        if (name[i] == " ") | (name[i] == "."):
            return int(name[:i])


def get_name(name):
    for i in range(len(name)):
        if name[i] == ".":
            return int(name[:i])

        if name[i] == "(":
            for j in range(0, 10):
                if name[i + j] == ")":
                    return int(name[:i - 1]) + 0.001 * int(name[i + 1:i + j])


def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image


def get_result(result):
    if len(result[0]) == 0:
        return [], []

    max_one_index = -1
    max_zero_index = -1
    max_one = 0
    max_zero = 0
    boxes = []
    categories = []
    for i in range(len(result[0])):

        if result[0][i][-1] == 1:
            if result[0][i][-2] > max_one:
                max_one = result[0][i][-2]
                max_one_index = i

        if result[0][i][-1] == 0:
            if result[0][i][-2] > max_zero:
                max_zero = result[0][i][-2]
                max_zero_index = i

    if max_one_index != -1:
        categories.append(1)
        boxes.append(result[0][max_one_index][0:4])

    if max_zero_index != -1:
        categories.append(0)
        boxes.append(result[0][max_zero_index][0:4])

    return boxes, categories


def get_Data(graph, graph_model, dist_model, file_name):
    graph = Image.open(graph)
    image_shape = np.array(np.shape(graph)[0:2])
    input_shape = get_new_img_size(image_shape[0], image_shape[1])
    image = cvtColor(graph)
    image_data = resize_image(image, [input_shape[1], input_shape[0]])
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()
        roi_cls_locs, roi_scores, rois, _ = graph_model.net(images)
        results = graph_model.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape,
                                         nms_iou=graph_model.nms_iou, confidence=graph_model.confidence)

    boxes, categories = get_result(results)
    # detect_result = graph_model(graph)
    # predictions = results.pred[0]
    # boxes = predictions[:, :4]
    # categories = predictions[:, 5]

    lst = [0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    lst[-2] = get_number(file_name) / 50
    # lst[-2] = get_number(file_name)
    lst[-3] = get_name(file_name)

    if len(categories) == 0:
        for x in range(len(lst) - 2):
            lst[x] = -1

    elif len(categories) == 1:
        xmin = boxes[0][0].item()
        xmax = boxes[0][2].item()
        ymin = boxes[0][1].item()
        ymax = boxes[0][3].item()

        if categories[0] == 0:
            lst[0] = -1
            lst[1] = -1
            lst[2] = -1
            lst[3] = -1
            lst[4] = -1
            lst[5] = -1
            lst[6] = -1
            lst[7] = -1
            lst[8] = xmin
            lst[9] = xmax
            lst[10] = ymin
            lst[11] = ymax
            lst[12] = xmax - xmin
            lst[13] = ymax - ymin
            lst[14] = (xmax - xmin) * (ymax - ymin)
            lst[15] = (xmax - xmin) / (ymax - ymin)
            lst[16] = -1
            lst[17] = -1
            lst[18] = -1
            lst[19] = -1
            lst[20] = -1
            lst[21] = -1
            lst[22] = -1
        else:
            lst[0] = xmin
            lst[1] = xmax
            lst[2] = ymin
            lst[3] = ymax
            lst[4] = xmax - xmin
            lst[5] = ymax - ymin
            lst[6] = (xmax - xmin) * (ymax - ymin)
            lst[7] = (xmax - xmin) / (ymax - ymin)
            lst[8] = -1
            lst[9] = -1
            lst[10] = -1
            lst[11] = -1
            lst[12] = -1
            lst[13] = -1
            lst[14] = -1
            lst[15] = -1
            lst[16] = -1
            lst[17] = -1
            lst[18] = -1
            lst[19] = -1
            lst[20] = -1
            lst[21] = -1
            lst[22] = -1

    else:
        if categories[0] == 0:
            hunter_index = 0
            monster_index = 1
        else:
            hunter_index = 1
            monster_index = 0

        h_xmin = boxes[hunter_index][0].item()
        h_xmax = boxes[hunter_index][2].item()
        h_ymin = boxes[hunter_index][1].item()
        h_ymax = boxes[hunter_index][3].item()
        m_xmin = boxes[monster_index][0].item()
        m_xmax = boxes[monster_index][2].item()
        m_ymin = boxes[monster_index][1].item()
        m_ymax = boxes[monster_index][3].item()

        lst[0] = m_xmin
        lst[1] = m_xmax
        lst[2] = m_ymin
        lst[3] = m_ymax
        lst[4] = m_xmax - m_xmin
        lst[5] = m_ymax - m_ymin
        lst[6] = (m_xmax - m_xmin) * (m_ymax - m_ymin)
        lst[7] = (m_xmax - m_xmin) / (m_ymax - m_ymin)
        lst[8] = h_xmin
        lst[9] = h_xmax
        lst[10] = h_ymin
        lst[11] = h_ymax
        lst[12] = h_xmax - h_xmin
        lst[13] = h_ymax - h_ymin
        lst[14] = (h_xmax - h_xmin) * (h_ymax - h_ymin)
        lst[15] = (h_xmax - h_xmin) / (h_ymax - h_ymin)
        lst[16] = abs(m_xmin - h_xmin)
        lst[17] = abs(m_xmax - h_xmax)
        lst[18] = abs(m_ymin - h_ymin)
        lst[19] = abs(m_ymax - h_ymax)
        lst[20] = (h_xmax - h_xmin) / (m_xmax - m_xmin)
        lst[21] = (h_ymax - h_ymin) / (m_ymax - m_ymin)
        lst[22] = ((h_xmax - h_xmin) * (h_ymax - h_ymin)) / ((m_xmax - m_xmin) * (m_ymax - m_ymin))

    if len(categories) >= 2:
        y_pred = dist_model.predict([lst[:-3]])
        lst[-1] = y_pred[0]
    else:
        lst[-1] = -1 / 50
        # lst[-1] = -1

    print(str(lst[-1]) + ", " + str(lst[-2]))
    return lst


def delta(threshold, real, predict):
    count_true = 0
    count_total = 0

    for index, values in real.items():
        if real[index] == 0:
            real[index] -= 0.0000000000000000000000000000000000000000001
        if predict[index] == 0:
            predict[index] -= 0.0000000000000000000000000000000000000000001

        res = max(real[index] / predict[index], predict[index] / real[index])
        if res < threshold:
            count_true += 1
        count_total += 1

    return count_true / count_total


def squa_rel(real, predict):
    count = 0
    total = 0

    for index, values in real.items():
        square = abs(real[index] - predict[index])
        curr = square / max(abs(real[index]), abs(predict[index]))
        total += curr
        count += 1

    return total / count


# yolo_model = yolov5.load("models/best_yolov3.pt")
# yolo_model.conf = 0.25
# yolo_model.iou = 0.45
# yolo_model.agnostic = False
# yolo_model.multi_label = False
# yolo_model.max_det = 1000
model = FRCNN()
# model.net.load_state_dict(torch.load("models/best_frcnn.pth"))
pretrained_dict = torch.load("models/best_frcnn.pth")
model_dict = model.net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'head.cls_loc' not in k)}
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'head.score' not in k)}
model_dict.update(pretrained_dict)
model.net.load_state_dict(model_dict)

with open("./models/adaboost_model.pkl", 'rb') as file:
    dist_estimate_model = pickle.load(file)

header = ["m_xmin", "m_xmax", "m_ymin", "m_ymax", "m_width", "m_height", "m_size", "m_wh_ratio",
          "h_xmin", "h_xmax", "h_ymin", "h_ymax", "h_width", "h_height", "h_size", "h_wh_ratio",
          "x_min_diff", "x_max_diff", "y_min_diff", "y_max_diff", "width_ratio", "height_ratio", "size_ratio",
          "name", "real_dist", "pred_dist"]
df = pd.DataFrame(columns=header)

count_image = 0
for path, dir_list, file_list in image_root:
    for file_name in file_list:
        curr_lst = get_Data(os.path.join(path, file_name), model, dist_estimate_model, file_name)
        df.loc[len(df.index)] = curr_lst
        count_image += 1
        print(count_image)

df.to_csv("./detect_and_estimate_data.csv", index=False)
# df = pd.read_csv("./detect_and_estimate_data.csv")
# df = df.sort_values("real_dist", ascending=True, inplace=False, kind='quicksort', ignore_index=True)

df = df.drop(df[(df["real_dist"] == -0.02) & (df["pred_dist"] != -0.02)].index)
df = df.drop(df[(df["real_dist"] != -0.02) & (df["pred_dist"] == -0.02)].index)
delta1 = delta(1.5, df["real_dist"], df["pred_dist"])
print("delta1: " + str(delta1))
delta2 = delta(2, df["real_dist"], df["pred_dist"])
print("delta2: " + str(delta2))
delta3 = delta(2.5, df["real_dist"], df["pred_dist"])
print("delta3: " + str(delta3))
r2 = r2_score(df["real_dist"], df["pred_dist"])
print("R2: " + str(r2))
mse = mean_squared_error(df["real_dist"], df["pred_dist"])
print("MSE: " + str(mse))
squarel = squa_rel(df["real_dist"], df["pred_dist"])
print("Squa Rel: " + str(squarel))
