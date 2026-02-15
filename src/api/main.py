from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import router
from src.orchestration.scheduler import start_scheduler, stop_scheduler

app = FastAPI(title="BioHarvest Acquisition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

#app.mount("/photos", StaticFiles(directory="photos"), name="photos")
#app.mount("/static", StaticFiles(directory="panel/dist"), name="static")


@app.on_event("startup")
def startup():
    start_scheduler()


@app.on_event("shutdown")
def shutdown():
    stop_scheduler()
