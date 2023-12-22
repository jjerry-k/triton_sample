import requests
import cv2
from api.utils import img_to_base64

# Test Mnist
img_path = "test_image/mnist_sample.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_enc = img_to_base64(img)
req = {"image": img_enc}
response = requests.post("http://localhost:8080/mnist/predict", json=req)
if response.status_code != 200:
    print(f"Check status!! (Code: {response.status_code}, Content: {response.content})")
else:
    response = response.json()
    print(f"Mnist Result: {response}")
    

# Test ResNet
img_path = "test_image/kitten.jpg"
img = cv2.imread(img_path)[..., :3]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
img_enc = img_to_base64(img)
req = {"image": img_enc}
response = requests.post("http://localhost:8080/resnet50/predict", json=req)
if response.status_code != 200:
    print(f"Check status!! (Code: {response.status_code}, Content: {response.content})")
else:
    response = response.json()
    print(f"ResNet50 Result: {response}")

# Test Yolo8n
img_path = "test_image/kitten.jpg"
img = cv2.imread(img_path)[..., :3]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))
img_enc = img_to_base64(img)
req = {"image": img_enc}
response = requests.post("http://localhost:8080/yolov8n/predict", json=req)
if response.status_code != 200:
    print(f"Check status!! (Code: {response.status_code}, Content: {response.content})")
else:
    response = response.json()
    print(f"ResNet50 Result: {response}")