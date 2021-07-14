import numpy as np
import cv2

def calc_hist(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=1)

    q1 = ((0 < ang) & (ang <= 45)).sum()
    q2 = ((45 < ang) & (ang <= 90)).sum()
    q3 = ((90 < ang) & (ang <= 135)).sum()
    q4 = ((135 < ang) & (ang <= 180)).sum()
    q5 = ((180 < ang) & (ang <= 225)).sum()
    q6 = ((225 <= ang) & (ang <= 270)).sum()
    q7 = ((270 < ang) & (ang <= 315)).sum()
    q8 = ((315 < ang) & (ang <= 360)).sum()

    hist = [q1, q2, q3, q4, q5, q6, q7, q8]

    return (hist)


def process_video(fn):
    video_hist = []
    hog_list = []
    sum_desc = []
    bins_n = 8

    cap = cv2.VideoCapture(fn)
    ret, prev = cap.read()

    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()

    while True:

        ret, img = cap.read()

        if not ret: break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prevgray = gray

        bins = np.hsplit(flow, bins_n)

        out_bins = []
        for b in bins:
            out_bins.append(np.vsplit(b, bins_n))

        frame_hist = []
        for col in out_bins:

            for block in col:
                frame_hist.append(calc_hist(block))

        video_hist.append(np.matrix(frame_hist))

    # average per frame
    sum_desc = video_hist[0]
    for i in range(1, len(video_hist)):
        sum_desc = sum_desc + video_hist[i]

    ave = sum_desc / len(video_hist)
    maxx = np.amax(video_hist, 0)
    maxx = np.matrix(maxx)


    ave_desc = np.asarray(ave)
    a_desc = []
    a_desc.append(np.asarray(ave_desc, dtype=np.uint8).ravel())
    label=1
    max_desc = np.asarray(maxx)
    m_desc = []
    m_desc = np.asarray(max_desc, dtype=np.uint8).ravel()
    print(a_desc)
    print(m_desc)
    #return a_desc, label, m_desc


if __name__ == '__main__':
    process_video('./louder.mp4')


