def profile_step(step_name, func, *args, **kwargs):
import time
start = time.time()
result = func(*args, **kwargs)
print(f"{step_name} took {time.time() - start:.2f} seconds")
return result
