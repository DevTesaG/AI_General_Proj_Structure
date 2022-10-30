import requests
from PIL import Image
import numpy as np

ENDPOINT_URL = "https://ml-gen-struct.herokuapp.com/infer"

def infer():
    image = np.asarray(Image.open('assets/poodle.jpg')).astype(np.float32)
    data = { 'image': image.tolist() }
    response = requests.post(ENDPOINT_URL, json = data)
    response.raise_for_status()
    print(response)

if __name__ =="__main__":
    infer()