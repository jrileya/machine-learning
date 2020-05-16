#!/usr/bin/env python

"""
This file creates the meta.json and numpy dataset.txt files from the
energy.csv file that was downloaded from the UCI machine learning repo.
"""

import os
import json
import pandas as pd
import numpy as np
from operator import itemgetter

DIRNAME   = os.path.dirname(__file__)
DATAPATH  = os.path.join(DIRNAME, "energy.csv")
OUTPATH   = os.path.join(DIRNAME, "dataset.txt")

FEATURES  = {
    "X1": "relative compactness",
    "X2": "surface area",
    "X3": "wall area",
    "X4": "roof area",
    "X5": "overall height",
    "X6": "orientation",
    "X7": "glazing area",
    "X8": "glazing area distribution"
}

LABEL_MAP = {
    "Y1": "heating load",
    "Y2": "cooling load",
}

if __name__ == "__main__":
    # Import the data frame
    df = pd.read_csv(DATAPATH)

    # Dump the numpy array
    np.savetxt(OUTPATH, df, fmt='%3.2f')

    print "Wrote dataset of %i instances and %i attributes to %s" \
          % (df.shape + (OUTPATH,))

    features = [v for (k, v) in sorted(FEATURES.items(), key=itemgetter(0))]

    with open('meta.json', 'w') as f:
        meta = {'feature_names': features, 'target_names': LABEL_MAP}
        json.dump(meta, f, indent=4)
