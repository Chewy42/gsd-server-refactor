# // * Copyright (C) 2023 Matthew Favela - All Rights Reserved
# // * You may use, distribute and modify this code under the
# // * terms of the MIT license, which unfortunately won't be
# // * written for another century.
# // *
# // * You should have received a copy of the MIT license with
# // * this file. If not, please write to: Chewy42
# // * @author Matthew Favela
# // * @version 1.0
# // * @since 2023-02-25
# // */


import torch
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template
import cv2
from datetime import datetime
import argparse
import io
import os
import base64
import requests
# from pymongo import MongoClient
# import dotenv
import geopandas as gpd
import logging

app = Flask(__name__)

DETECTION_URL = "/gsd-inference-server/models/v1"
detection_threshold = 0.5

@app.route(DETECTION_URL, methods=['POST'])
def predict():
    logging.info('Received request: %s', request)
    if not request.method == "POST":
        return
    if request.files.get("image") and request.files.get("store") and request.files.get("products") == None:
        display_sale_count = 0
        quantity_sale_count = 0
        price_reduction_count = 0
        confidence_level = 0
        now = datetime.now()
        gdf = gpd.tools.geocode('Orange, CA')

        # get the data from the POST request.
        request.files.get("image")
        data = request.files["image"]
        time_sent = now.strftime("%d%m%Y%H%M%S")

        #get and preprocess the image
        image_file = data
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=416)
        data = results.pandas().xyxy[0]
        data_json = data.to_json(orient="records")

        file_bytes = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        #get detections
        for index, row in data.iterrows():
            if row["class"] == 2:
                cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                              (0, 255, 0), 2)
                quantity_sale_count += 1
            elif row["class"] == 1:
                cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                              (0, 0, 255), 2)
                price_reduction_count += 1
            else:
                cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                              (255, 255, 0), 2)
                display_sale_count += 1

        total_sales_count = display_sale_count + quantity_sale_count + price_reduction_count

        #make directory
        if not os.path.exists(f"./image_submissions/"):
            os.makedirs(f"./image_submissions/")

        #save image
        cv2.imwrite(f"./image_submissions/result_{time_sent}.png", frame)

        #calculate confidence level
        confidence_level = (total_sales_count / 10) * 100

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
        print(f"TOTAL: {total_sales_count}")
        print("------------------------------------------")
        print("COORDINATES OF IMAGE")
        print("Latitude:", gdf['geometry'][0].x, "Longitude:", gdf['geometry'][0].y)
        print("------------------------------------------")
        print("CONFIDENCE LEVEL")
        print(f"TOTAL: {round(confidence_level, 2)}")
        print("------------------------------------------")
        print("TIME SENT")
        print(f"TOTAL: {time_sent}")
        print("------------------------------------------")


        image = open(f"./image_submissions/result_{time_sent}.png", "rb")
        image_read = image.read()
        image_64_encode = base64.encodebytes(image_read)
        image_64_encode = image_64_encode.decode("utf-8")
        image.close()

        #insert into db collection
        # collection.insert_one({
        #     "time_sent": float(time_sent),
        #     "latitude": float(gdf['geometry'][0].y),
        #     "longitude": float(gdf['geometry'][0].x),
        #     "drone_count": float(drone_count),
        #     "confidence_level": round(confidence_level, 2),
        #     "image": image_64_encode,
        #     "data": data_json
        # })

        return jsonify({"display_sales_detected": display_sale_count, "quantity_sales_detected": quantity_sale_count,"price_reduction_sales_detected": price_reduction_count,"confidence_level": confidence_level})
    elif request.files.get("image") and request.files.get("store") and request.files.get("products"):
        display_sale_count = 0
        quantity_sale_count = 0
        price_reduction_count = 0
        product_list = request.form.get("products").split("/")
        confidence_level = 0
        now = datetime.now()
        gdf = gpd.tools.geocode('Orange, CA')

        # get the data from the POST request.
        request.files.get("image")
        data = request.files["image"]
        time_sent = now.strftime("%d%m%Y%H%M%S")

        # get and preprocess the image
        image_file = data
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        results = model(img, size=416)
        data = results.pandas().xyxy[0]
        data_json = data.to_json(orient="records")

        file_bytes = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # get detections
        for index, row in data.iterrows():
            if row["class"] == 2:
                cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                              (0, 255, 0), 2)
                quantity_sale_count += 1
            elif row["class"] == 1:
                cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                              (0, 0, 255), 2)
                price_reduction_count += 1
            else:
                cv2.rectangle(frame, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                              (255, 255, 0), 2)
                display_sale_count += 1

        total_sales_count = display_sale_count + quantity_sale_count + price_reduction_count

        # make directory
        if not os.path.exists(f"./image_submissions/"):
            os.makedirs(f"./image_submissions/")

        # save image
        cv2.imwrite(f"./image_submissions/result_{time_sent}.png", frame)

        # print results
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
        print(f"TOTAL: {total_sales_count}")
        print("------------------------------------------")
        print("COORDINATES OF IMAGE")
        print("Latitude:", gdf['geometry'][0].x, "Longitude:", gdf['geometry'][0].y)
        print("------------------------------------------")
        print("CONFIDENCE LEVEL")
        print(f"TOTAL: {round(confidence_level, 2)}")
        print("------------------------------------------")
        print("TIME SENT")
        print(f"TOTAL: {time_sent}")
        print("------------------------------------------")


        image = open(f"./image_submissions/result_{time_sent}.png", "rb")
        image_read = image.read()
        image_64_encode = base64.encodebytes(image_read)
        image_64_encode = image_64_encode.decode("utf-8")
        image.close()

        # insert into db collection
        # collection.insert_one({
        #     "time_sent": float(time_sent),
        #     "latitude": float(gdf['geometry'][0].y),
        #     "longitude": float(gdf['geometry'][0].x),
        #     "drone_count": float(drone_count),
        #     "confidence_level": round(confidence_level, 2),
        #     "image": image_64_encode,
        #     "data": data_json
        # })

        return jsonify({"display_sales_detected": display_sale_count, "quantity_sales_detected": quantity_sale_count,
                        "price_reduction_sales_detected": price_reduction_count, "confidence_level": confidence_level})
    else:
        return jsonify({"error": "no image or store or products"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    # dotenv.load_dotenv()
    # client = MongoClient(os.getenv("MONGO_URI"))
    # db = client.chapmandassh
    # collection = db.detections



    model = torch.hub.load('ultralytics/yolov5', 'custom', './models/gsd.pt')
    print("model loaded")
    model.eval()
    app.run(host="0.0.0.0", port=args.port, debug=True)