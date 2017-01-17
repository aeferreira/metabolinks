import pandas as pd
import tkinter as tk
from tkinter import filedialog

def merge(file, sheet, error):
    
    '''
    Merge and align m/z data from an MS-office Excel file.
    
    Parameters:
        file : File path
        sheet : Name of sheet of the DataFrame
        error : Error of aligment in ppm
    '''   
    
    df = pd.read_excel(file, sheet)
    df = df.dropna(axis=1, how='all')
    a = list(df.columns.values)
    t = df[[a[0], a[1]]]
    t.insert(2, 'Tubo', a[1])
    t.columns = ['Mass', 'Intensity', 'Tubo']
    for x in range(2,len(a),2):
        tempdf = df[[a[x], a[x+1]]]
        tempdf.insert(2, 'Tubo', a[x+1])
        tempdf.columns = ['Mass', 'Intensity', 'Tubo']
        t = pd.concat([t,tempdf])
    t = t.sort_values('Mass')
    t = t.reset_index()
    t = t.drop('index', axis=1)
    
    dfp = pd.DataFrame()
    mass = []
    d = {}
    last=0
    
    for index, row in t.iterrows():
        if index == 0:
            last = row.Mass
            d[row.Tubo] = row.Intensity
            mass.append(row.Mass)        
            continue
        elif row.Tubo in d:
            c = sum(mass)/len(mass)
            df_temp = pd.DataFrame(d, index=[c])
            dfp = pd.concat([dfp,df_temp])
            mass = []
            d = {}
            last=row.Mass
            d[row.Tubo] = row.Intensity
            mass.append(row.Mass)
        elif abs((last-row.Mass)*1000000)/last <= error:
            d[row.Tubo] = row.Intensity
            mass.append(row.Mass)
        else:
            c = sum(mass)/len(mass)
            df_temp = pd.DataFrame(d, index=[c])
            dfp = pd.concat([dfp,df_temp])
            mass = []
            d = {}
            last=row.Mass
            d[row.Tubo] = row.Intensity
            mass.append(row.Mass)
    dfp = dfp[pd.notnull(dfp.index)]
    
    dfp.insert(len(dfp.columns), 'Samples', dfp.notnull().sum(axis=1))
    dfp.insert(len(dfp.columns), 'temp', dfp.isnull().sum(axis=1))
    dfp.insert(len(dfp.columns), 'Use', dfp.Samples>dfp.temp)
    dfp = dfp.drop('temp', 1)
    
    return dfp

if __name__=='__main__':
##     root = tk.Tk()
##     root.withdraw()
##     file = filedialog.askopenfilename(filetypes = [("EXCEL","*.xlsx")])
##     print ('Opening file:\n'+file.split('/')[-1])
    
    exp_filename = "merge_example_data/1 - Cabernet Sauvignon_Fev2016.xlsx"


    xl = pd.ExcelFile(exp_filename)
    l = xl.sheet_names
    print ('\nStarting alingment of file {}...\n'.format(exp_filename))


    writer = pd.ExcelWriter(exp_filename[:-5]+'_aligned.xlsx')
    for x in l:
        merge(exp_filename, x, 1).to_excel(writer, index=True, header=True, 
                                           index_label='m/z', sheet_name=x)
        print (x +' aligned')
    writer.save()


    print('\nAll done!')



