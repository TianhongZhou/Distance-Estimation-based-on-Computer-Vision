import pickle
import gui
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt
import tkinter
from PIL import ImageGrab
import sys
import pandas as pd
import yolov5
import cgitb


def screenshot():
    win = tkinter.Tk()
    width = win.winfo_screenwidth()
    height = win.winfo_screenheight()
    img = ImageGrab.grab(bbox=(0, 0, width, height))
    img.save("./current_screen.jpg")
    return img


def click(input_ui, graph_model, dist_model):
    graph = screenshot()
    detect_result = graph_model(graph)
    predictions = detect_result.pred[0]
    boxes = predictions[:, :4]

    if (graph.size[0] != 1080) | (graph.size[1] != 1920):
        for box in boxes:
            box[0].item = box[0].item() / graph.size[1] * 1920
            box[2].item = box[0].item() / graph.size[1] * 1920
            box[1].item = box[0].item() / graph.size[0] * 1080
            box[3].item = box[0].item() / graph.size[0] * 1080

    categories = predictions[:, 5]

    lst = [0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0]

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

    if lst[0] == -1:
        input_ui.label.setText("No Monster Detected")
        return

    if lst[9] == -1:
        input_ui.label.setText("No Hunter Detected")
        return

    y_pred = dist_model.predict([lst])
    input_ui.label.setText("Current Distance: " + str(float(y_pred[0])))


if __name__ == "__main__":
    cgitb.enable(format="text")
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = gui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
    MainWindow.show()

    if ui.comboBox.currentText() == "Great Sword: Strongarm Stance -> True Charged Slash":
        ui.label_2.setText(ui.label_2.text() + "    1.4 to 2.1")

    yolo_model = yolov5.load("models/best_yolov5_sgd.pt")
    yolo_model.conf = 0.25
    yolo_model.iou = 0.45
    yolo_model.agnostic = False
    yolo_model.multi_label = False
    yolo_model.max_det = 1000

    with open("./models/random_forest_model.pkl", 'rb') as file:
        dist_estimate_model = pickle.load(file)

    ui.pushButton.clicked.connect(lambda: click(ui, yolo_model, dist_estimate_model))

    sys.exit(app.exec())
