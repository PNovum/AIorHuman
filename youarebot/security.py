from fastapi import Depends, HTTPException, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os

security = HTTPBasic()

try:
    VALID_USERNAME = os.environ["BASIC_USER"]
    VALID_PASSWORD = os.environ["BASIC_PASS"]
except KeyError as e:
    missing = e.args[0]
    raise RuntimeError(
        f"Environment variable {missing}"
    )

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != VALID_USERNAME or credentials.password != VALID_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

async def protect_docs(request: Request, call_next):
    protected_paths = {"/docs", "/redoc", "/openapi.json"}
    if request.url.path in protected_paths:
        auth = request.headers.get("Authorization")
        if not auth or not auth.startswith("Basic "):
            return Response(headers={"WWW-Authenticate": "Basic"}, status_code=401)
    return await call_next(request)
