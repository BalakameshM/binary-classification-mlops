# src/api/main.py
from fastapi import FastAPI
from src.api.router import router

app = FastAPI(title="Cereal MLOps API")
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
