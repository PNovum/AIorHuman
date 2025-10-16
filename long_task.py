import time

import datetime

with open("output.txt", "w") as f:
    for i in range(1, 120):
        f.write(f"{i}: Hello from server\n")
        f.write(f"{datetime.datetime.now()}")
        f.flush()
        time.sleep(1)
