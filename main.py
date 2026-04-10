import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import modules
from user_management import user_mgmt_router
from gmail import gmail_router
from settings import settings_router
from upload import upload_router
from performance import perf_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Seedfolio API", root_path="/api/v1")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_mgmt_router)
app.include_router(gmail_router)
app.include_router(settings_router)
app.include_router(upload_router)
app.include_router(perf_router)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7071))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
