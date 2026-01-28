import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

def blobs(img, x, y, r):
    x2 = np.random.randint(x - 200, x + 200)
    x3 = np.random.randint(x - 200, x + 200)
    x4 = np.random.randint(x - 200, x + 200)

    y2 = np.random.randint(y - 200, y + 200)
    y3 = np.random.randint(y - 200, y + 200)
    y4 = np.random.randint(y - 200, y + 200)

    r2 = np.random.randint(r - 50, r + 50)
    r3 = np.random.randint(r - 50, r + 50)
    r4 = np.random.randint(r - 50, r + 50)

    x_lst, y_lst, r_lst = [x, x2, x3, x4], [y, y2, y3, y4], [r, r2, r3, r4]

    for i in range(4):
        cv.circle(img, (x_lst[i], y_lst[i]), r_lst[i], (255, 0, 0), -1)

width, height = 2560, 1790

def generate_images(n_images):
    for k in range(n_images):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        n_blobs = np.random.randint(1, 10)
        x_circle = np.random.randint(0, width, size=n_blobs)
        y_circle = np.random.randint(0, height, size=n_blobs)
        r_circle = np.random.randint(100, 200, size=n_blobs)

        n_plates = np.random.randint(10, 30)
        x_plates = np.random.randint(0, width, size=n_plates)
        y_plates = np.random.randint(0, height, size=n_plates)
        major_axis_lenght = np.random.randint(100, 1500, size=n_plates)
        minor_axis_lenght = np.random.randint(100, 1500, size=n_plates)
        angle = np.random.randint(0, 360, size=n_plates)
        start_angle = np.random.randint(0, 360, size=(n_plates))
        end_angle = np.random.randint(start_angle, start_angle + 3, size=n_plates)

        for j in range(n_plates):
            cv.ellipse(img, (x_plates[j], y_plates[j]), (major_axis_lenght[j], minor_axis_lenght[j]), angle[j], start_angle[j], end_angle[j], (0, 0, 255), -1)

        for i in range(n_blobs):
            blobs(img, x_circle[i], y_circle[i], r_circle[i])

        im = Image.fromarray(img)

        im.save(rf"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\label\{k}.png")

        raw = img.mean(axis=2).astype(np.uint8)

        raw = ((raw / raw.max()) * 255).astype(np.uint8)

        im2 = Image.fromarray(raw)

        im2.save(rf"C:\Users\magfa\Documents\Master\Masteroppgave\synthetic_dataset\raw\{k}.png")

generate_images(100)