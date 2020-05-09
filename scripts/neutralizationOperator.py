import pandas as pd
import numpy as np
def neutralize_series(series, by, proportion=1.0):
   scores = series.values.reshape(-1, 1)
   exposures = by.values.reshape(-1, 1)

   # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
   exposures = np.hstack(
       (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

   correction = proportion * (exposures.dot(
       np.linalg.lstsq(exposures, scores)[0]))
   corrected_scores = scores - correction
   neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
   return neutralized