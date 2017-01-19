
# coding: utf-8

# In[ ]:

'''
Convert aligned file (.xlsx) to metaboanalyst (.csv) ready file.

In MetaboAnalyst choose option: Samples in columns(unpaired).
'''


# In[ ]:

import pandas as pd
import tkinter as tk
from tkinter import filedialog


# In[ ]:

root = tk.Tk()
root.withdraw()
file = filedialog.askopenfilename(filetypes = [("EXCEL","*.xlsx")])
print ('Opening file:\n'+file.split('/')[-1])


# In[ ]:

df = pd.read_excel(file)


# In[ ]:

df.drop('Use', axis=1, inplace=True)
df.drop('Samples', axis=1, inplace=True)


# In[ ]:

df


# In[ ]:

df.to_csv(file[:-5]+'_metaboanalyst.csv', index=False, header=True)


# In[ ]:

temp = []
with open(file[:-5]+'_metaboanalyst.csv') as f:
    for x in f:
        if x.startswith('m/z'):
            temp.append('Sample,'+x.split(',',1)[1])
            temp.append('Label,I,I,I,Mock,Mock,Mock\n')
        else:
            temp.append(x)
with open(file[:-5]+'_metaboanalyst.csv', 'w') as f:
    f.write("".join(temp))


# In[ ]:

print ('Converted to MetaboAnalyst')

