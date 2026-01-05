# src/utils/decorators.py
import time
from functools import wraps
from src.utils.logger import get_logger

log = get_logger(__name__)

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = fn(*args, **kwargs)
        ms = (time.time() - start) * 1000
        log.info(f"{fn.__name__} took {ms:.2f} ms")
        return out
    return wrapper
