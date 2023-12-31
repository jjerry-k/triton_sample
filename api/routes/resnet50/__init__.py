import uuid
import json
import traceback
from fastapi import APIRouter, HTTPException, status

from settings import logger, grpcclient, TRITON_CLIENT, IMAGENET_INDEX
from utils import np, base64_to_img

from .model import Request, Response

router = APIRouter(
    prefix="/resnet50",
    tags=["Image Classification"]
)

outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]
mean = [[[0.485]], [[0.456]], [[0.406]]]
std = [[[0.229]], [[0.224]], [[0.225]]]
        
@router.post("/predict", response_model=Response)
async def predict(data: Request) -> Response:
    request_id = uuid.uuid4().hex
    logger.info(f"Start Requst (ID: {request_id})")
    try:
        # Preprocessing
        img = base64_to_img(data.image) / 255.
        # (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
        img = img.reshape(256, 256, 3)
        img = img.transpose([2, 0, 1])
        img = (img - mean) / std
        img = img[np.newaxis]
        
        # Inference
        inputs = [grpcclient.InferInput("INPUT__0", img.shape, "FP32")]
        inputs[0].set_data_from_numpy(img.astype(np.float32))
        results = await TRITON_CLIENT.infer(
                                            model_name="resnet50",
                                            inputs=inputs,
                                            outputs=outputs
                                        )

        # Postprocessing
        output = results.as_numpy("OUTPUT__0")
        top_1 = output.argmax()
        logger.info(f"Success Requst (ID: {request_id})")
        return Response(prediction=IMAGENET_INDEX[str(top_1)][1])
        
    except Exception as e:
        logger.error(f"Error Requst (ID: {request_id}), {traceback.format_exc()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail=str(traceback.format_exc()))