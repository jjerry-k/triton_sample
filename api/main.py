from fastapi import FastAPI, APIRouter

app = FastAPI()

router = APIRouter(
    prefix="",
    tags=["Basic"]
)

@router.get("/")
def health() -> bool:
    return {"success": True}

app.include_router(router)

# Import endpoints in routes
import routes
for k, v in routes.__dict__.items():
    if "router" in dir(v):
        exec(f"app.include_router({v.router})")