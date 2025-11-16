import time
from fastapi import Request

async def logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    path = request.url.path
    method = request.method
    status = response.status_code
    print(f"{method} {path} -> {status} ({duration:.1f} ms)")
    return response
