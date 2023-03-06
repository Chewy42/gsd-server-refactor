import pprint
import requests

DETECTION_URL = "http://localhost:4242/gsd-inference-server/models/v1"


image_location = f"test.png"

image_data = open(image_location, "rb").read()
store_name = "albertsons"

data = {
    "image": image_data,
    "store": store_name,
    "products": None,
}

response = requests.post(DETECTION_URL, files=data)

pprint.pprint(response)