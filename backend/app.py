from fastapi import FastAPI

from backend.routes import router

app = FastAPI(title="ChordEdit")
app.include_router(router)
