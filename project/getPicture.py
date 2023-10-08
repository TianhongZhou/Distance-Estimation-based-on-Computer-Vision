import cv2


def read_video(path):
    video_captured = cv2.VideoCapture(path)
    return video_captured


def get_video_info(video_captured):
    fps = video_captured.get(cv2.CAP_PROP_FPS)
    video_size = (int(video_captured.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video_captured.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = video_captured.get(cv2.CAP_PROP_FRAME_COUNT)
    return fps, video_size, frame_count


def save_image(video_captured, step, name):
    count = 0
    status, frame = video_captured.read()

    while status:
        if count % step == 0:
            cv2.imwrite("F:/document/University/Research/2022-IEMP-Bjorn/research/images/" + str(name) + ".jpg", frame)
            name = name + 1
        count = count + 1
        status, frame = video_captured.read()

    video_captured.release()


video = read_video(r"F:\document\University\Research\2022-IEMP-Bjorn\research\video\150.mp4")
# print(get_video_info(video))
# save_image(video, 20, 5075)
