from fastapi import FastAPI
from app.routes.app_routes import router

app = FastAPI()


app.include_router(router)
