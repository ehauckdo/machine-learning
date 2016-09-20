#!  /usr/bin/env python

import pandas as pd
import numpy as np
import sys
import gc
from Tools.estimators import *
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("input/train.csv", keep_default_na=False)
print(df.head(5))
