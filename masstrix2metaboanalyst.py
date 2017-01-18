
# coding: utf-8

# In[ ]:

'''
Remove all the non-compound peaks.
Takes as input a aligned file (.xlsx) and a masstrix file (.tsv) and as output a metaboanalyst (.cvs) ready file.
(only with the compound found in masstrix).

In MetaboAnalyst choose option: Samples in columns(unpaired).
'''


# In[ ]:

import pandas as pd
import tkinter as tk
from tkinter import filedialog


# In[ ]:

root = tk.Tk()
root.withdraw()
print ('A abrir ficheiro...')
file = filedialog.askopenfilename(filetypes = [("Excel","*.xlsx")])
file2 = filedialog.askopenfilename(filetypes = [("TSV","*.tsv")])


# In[ ]:

df = pd.read_excel(file)


# In[ ]:

df['m/z'] = df['m/z'].round(7)


# In[ ]:

if 'Use' in list(df):
    df = df.drop('Use', axis=1)
if 'Samples' in list(df):
    df = df.drop('Samples', axis=1)


# In[ ]:

l=[]
with open(file2) as f:
    for x in f:
        try:
            l.append(float(x.split('\t')[0]))
        except ValueError:
            continue


# In[ ]:

col = list(df)[1:]
dfp = df[df['m/z'].isin(l)]


# In[ ]:

dfp.to_csv(file[:-5]+'_metaboanalyst.csv', index=False, header=True)


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


# In[ ]:



