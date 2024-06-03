from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
loadboston=load_digits()

x,y=load_digits(return_X_y=True)

print(y)

print(loadboston.feature_names)

