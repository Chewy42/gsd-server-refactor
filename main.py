from flask import Flask, request, jsonify
import os
from PIL import Image
import cv2
from datetime import datetime
import torch
import pandas as pd
import base64

app = Flask(__name__)

@app.route('/api/models/v1', methods=['POST'])
def api():
    if request.method == 'POST':

        #sale counts
        display_sale_count = 0
        quantity_sale_count = 0
        price_reduction_count = 0

        #time
        now = datetime.now()
        time_sent = now.strftime("%d%m%Y%H%M%S")
        
        #get image from json
        data = request.get_json()
        image_base64 = data["image"]

        #save image
        image_data = base64.b64decode(image_base64)
        with open("image.jpg", "wb") as f:
            f.write(image_data)
        
        #load image
        img = Image.open("image.jpg")

        #detect sales
        results = model(img, size=416)
        data = results.pandas().xyxy[0]
        data_json = data.to_json(orient="records")
        

        print("No products specified by user")
        print(f"{all_sales} discounts detected in image")
        print("------------------------------------------")
        print(f"{display_sale_count} display sales")
        print(f"{price_reduction_count} price reduction sales")
        print(f"{quantity_sale_count} quantity sales")
        return jsonify("api endpoint hit!!")
    else:
        return jsonify("404 Server Error!")
    
if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'gsd.pt')
    print("Model successfully loaded!")
    model.eval()
    app.run(host=os.environ.get('HOST', '0.0.0.0'),
            port=int(os.environ.get('PORT', 5000)),
            debug=True)
