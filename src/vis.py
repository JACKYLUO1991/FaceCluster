from __future__ import print_function, division

import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import time

import face_recognition
from imutils import build_montages
from imutils import paths

import sys
sys.path.append("../")
import baseline


if __name__ == "__main__":
    data = []

    for imagePath in tqdm(list(paths.list_images("../dataset"))):
        # img = cv.imread(imagePath)
        # rgb = img[:, :, ::-1]
        rgb = face_recognition.load_image_file(imagePath)
        boxes = face_recognition.face_locations(
            rgb, model='hog')  # 'cnn' or 'hog'
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        d = [{'imagePath': imagePath, 'loc': box, 'encoding': enc}
             for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

    data = np.array(data)
    encodings = [d['encoding'] for d in data]

    # Method Choice
    s = time.time()
    # labels = baseline.dbscan(encodings, eps=0.5, min_samples=3)
    labels = baseline.chinese_whispers(encodings, threshold=0.5)
    print("[INFO] time: {}".format((time.time() - s) * 1000))

    labelIDs = np.unique(labels)
    # print(labelIDs)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    # print(numUniqueFaces)

    for labelID in labelIDs:
        idxs = np.where(labels == labelID)[0]
        faces = []

        for i in idxs:
            bgr = cv.imread(data[i]['imagePath'])
            (top, right, bottom, left) = data[i]['loc']
            face = cv.resize(bgr[top: bottom, left: right], (128, 128))
            faces.append(face)
        montage = build_montages(faces, (128, 128), (2, 5))[0]
        title = "Face ID #{}".format(labelID)
        title = "Unknown Faces" if labelID == -1 else title
        cv.imshow(title, montage)
        cv.waitKey(-1)
