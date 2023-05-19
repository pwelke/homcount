import pandas as pd
import numpy as np


# parse and pretty print some results
infiles = ['2022-12-20_as-wl3.csv']
    

for infile in infiles:
    df = pd.read_csv(infile, header=None, delimiter=' ')
    print(df.groupby([5,10], ).agg(mean=(8, np.mean), std=(8, np.std)).to_latex())