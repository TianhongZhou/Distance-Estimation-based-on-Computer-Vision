import os
import xml.dom.minidom as xmldom
import pandas as pd

label_root = os.walk(r"F:\document\University\Research\2022-IEMP-Bjorn\research\labels")


def parse_xml(file_path):
    xml_file = xmldom.parse(file_path)
    elements = xml_file.documentElement
    length = elements.getElementsByTagName("object").length
    lst = [0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    name = elements.getElementsByTagName("filename")[0].firstChild.data
    lst[-1] = get_number(name)
    if int(get_number(name)) < 0:
        lst[-2] = -1
    else:
        lst[-2] = (int(get_number(name))) / 50
    lst[-3] = name

    if length == 0:
        for x in range(len(lst)):
            lst[x] = -1

    elif length == 1:
        xmin = float(elements.getElementsByTagName("xmin")[0].firstChild.data)
        xmax = float(elements.getElementsByTagName("xmax")[0].firstChild.data)
        ymin = float(elements.getElementsByTagName("ymin")[0].firstChild.data)
        ymax = float(elements.getElementsByTagName("ymax")[0].firstChild.data)

        if elements.getElementsByTagName("name")[0].firstChild.data == "hunter":
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
        if elements.getElementsByTagName("name")[0].firstChild.data == "hunter":
            hunter_index = 0
            monster_index = 1
        else:
            hunter_index = 1
            monster_index = 0

        h_xmin = float(elements.getElementsByTagName("xmin")[hunter_index].firstChild.data)
        h_xmax = float(elements.getElementsByTagName("xmax")[hunter_index].firstChild.data)
        h_ymin = float(elements.getElementsByTagName("ymin")[hunter_index].firstChild.data)
        h_ymax = float(elements.getElementsByTagName("ymax")[hunter_index].firstChild.data)
        m_xmin = float(elements.getElementsByTagName("xmin")[monster_index].firstChild.data)
        m_xmax = float(elements.getElementsByTagName("xmax")[monster_index].firstChild.data)
        m_ymin = float(elements.getElementsByTagName("ymin")[monster_index].firstChild.data)
        m_ymax = float(elements.getElementsByTagName("ymax")[monster_index].firstChild.data)

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

    return lst


def get_number(name):
    for i in range(len(name)):
        if (name[i] == " ") | (name[i] == "."):
            return int(name[:i])


header = ["m_xmin", "m_xmax", "m_ymin", "m_ymax", "m_width", "m_height", "m_size", "m_wh_ratio",
          "h_xmin", "h_xmax", "h_ymin", "h_ymax", "h_width", "h_height", "h_size", "h_wh_ratio",
          "x_min_diff", "x_max_diff", "y_min_diff", "y_max_diff", "width_ratio", "height_ratio", "size_ratio",
          "name", "scaled_dist", "dist"]
df = pd.DataFrame(columns=header)

for path, dir_list, file_list in label_root:
    for file_name in file_list:
        curr_lst = parse_xml(os.path.join(path, file_name))
        df.loc[len(df.index)] = curr_lst

df.to_csv("./data.csv", index=False)
