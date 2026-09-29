import json
import numpy as np

mae = np.float64(1.23456)
val = round(mae, 4)
print("Type of val:", type(val))

try:
    json.dumps({"val": val})
    print("JSON serialization SUCCESS")
except Exception as e:
    print("JSON serialization FAILED:", type(e).__name__, e)
