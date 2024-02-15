import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

for i in range(32):
    name = 'views_subdir' + str(i) + '.zip'

    cmd = 'unzip ' + name + ' -d views'
    print(cmd)
    os.system(cmd)




















