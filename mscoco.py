import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

for i in range(60):
    if i < 10:
        name = 0000+ str(i)
    else:
        name = 000+ str(i)

    cmd = 'tar xopf ' + name + '.tar'
    print(cmd)
    os.run(cmd)




















