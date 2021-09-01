import pandas as pd

# LOAD COLVAR FILES 
def load_colvar(filename='COLVAR',folder='./',sep=' '):
    skip_rows=1
    headers = pd.read_csv(folder+filename,sep=' ',skipinitialspace=True, nrows=0).columns[2:]  
    colvar = pd.read_csv(folder+filename,sep=sep,skipinitialspace=True,
                         header=None,skiprows=range(skip_rows),names=headers,comment='#')
    
    return colvar