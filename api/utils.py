import base64
import cv2
import numpy as np

def img_to_base64(image: np.ndarray) -> str:
    return base64.b64encode(image.tobytes()).decode("utf8")


def base64_to_img(img: str, mode="rgb") -> np.ndarray:
    img = np.frombuffer(base64.urlsafe_b64decode(img), dtype=np.uint8)
    img_dec = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE if mode=="gray" else cv2.IMREAD_COLOR)
    is_encoded = img_dec is not None
    return img_dec if is_encoded else img