from fastapi import APIRouter
from fastapi.responses import FileResponse
from src.core.experiment_runner import run_capture_cycle
from src.database.repository import get_statistics, get_photos

router = APIRouter()

@router.get("/")
async def index():
    return FileResponse("panel/dist/index.html")


@router.get("/take_photo")
async def take_photo():
    path = run_capture_cycle(manual=True)
    return {"message": "Foto tomada", "path": path}


@router.get("/statistic")
async def statistics():
    return get_statistics()


@router.get("/getPhotos")
async def photos():
    return get_photos()
