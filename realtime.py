# // * Copyright (C) 2023 Matthew Favela - All Rights Reserved
# // * You may use, distribute and modify this code under the
# // * terms of the MIT license, which unfortunately won't be
# // * written for another century.
# // *
# // * You should have received a copy of the MIT license with
# // * this file. If not, please write to: Chewy42
# // * @author Matthew Favela
# // * @version 1.0
# // * @since 2023-03-06
# // */


import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from datetime import datetime
import io
import os
import base64

model = torch.hub.load('ultralytics/yolov5', 'custom', './models/gsd.pt')
print("model loaded")
model.eval()

detection_threshold = 0.5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    display_sale_count = 0
    quantity_sale_count = 0
    price_reduction_count = 0
    total_sale_count = 0
    confidence_level = 0

    _, frame = cap.read()
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: invalid image.")

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #get and preprocess the image
    _, buffer = cv2.imencode('.png', frame)
    image_bytes = base64.b64encode(buffer)
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img, size=416)
    data = results.pandas().xyxy[0]
    data_json = data.to_json(orient="records")

    #get detections
    for index, row in data.iterrows():
        if row["class"] == 2:
            cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                          (0, 255, 0), 2)
            quantity_sale_count += 1
            total_sale_count += 1
        elif row["class"] == 1:
            cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                          (0, 0, 255), 2)
            price_reduction_count += 1
            total_sale_count += 1
        else:
            cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                          (255, 0, 0), 2)
            display_sale_count += 1
            total_sale_count += 1
        confidence_level += row["confidence"]

    #display the image
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #calculate confidence level
    confidence_level = (confidence_level / 10) * 100

    #print results
    print("------------------------------------------")
    print("DISPLAY SALES")
    print(f"TOTAL: {display_sale_count}")
    print("------------------------------------------")
    print("QUANTITY SALES")
    print(f"TOTAL: {quantity_sale_count}")
    print("------------------------------------------")
    print("PRICE REDUCTION SALES")
    print(f"TOTAL: {price_reduction_count}")
    print("------------------------------------------")
    print("TOTAL SALES")
    print(f"TOTAL: {total_sale_count}")
    print("------------------------------------------")
    print("CONFIDENCE LEVEL")
    print(f"TOTAL: {round(confidence_level, 2)}")
    print("------------------------------------------")

cap.release()
cv2.destroyAllWindows()