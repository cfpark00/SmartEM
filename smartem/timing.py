from pathlib import Path
from functools import wraps
import time

time_on = False


def timing(f):
    if not time_on:
        return f
    else:
        @wraps(f)
        def wrap(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()
            # print(f"func:{f.__name__} args:[{args},{kw}] took: {te-ts:2.4f} sec")
            file = Path(str(f.__code__.co_filename))
            print(f"func:{f.__name__} @ {file.stem} took: {te-ts:2.4f} sec")
            return result

        return wrap
