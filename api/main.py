from fastapi import FastAPI, APIRouter

from routes import mnist, resnet50

app = FastAPI()

router = APIRouter(
    prefix="",
    tags=["Basic"]
)

@router.get("/")
def health() -> bool:
    return {"success": True}

app.include_router(router)
app.include_router(mnist.router)
app.include_router(resnet50.router)