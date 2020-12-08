"""

    Set the fname to the file you want to convert to fixed with format.
    
    The skript will output the same file but formatet as fixed with file.

    Not you need the package tabulate. 

"""

from tabulate import tabulate
import pandas as pd

def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)

pd.DataFrame.to_fwf = to_fwf

# convert to fixed with file
fname = "ao-360-400_combine"

df = pd.read_csv(fname + ".txt", sep = '\s+')
df.to_fwf(fname + ".txt")

print("Converted formating! Exported to: ", fname + "_fwf.txt")