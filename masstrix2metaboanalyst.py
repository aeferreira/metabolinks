'''Takes as input a aligned file (.xlsx) and a masstrix file (.tsv) and as output a metaboanalyst .cvs ready file.
(only with the compound found in masstrix).'''


import pandas as pd
import tkinter as tk
from tkinter import filedialog


def round_up(number):
    '''
    Try to correct the float number round "problem".
    To use only with floats with 8 decimals places and with the intent to round up to 7 decimals places.
    To learn more go to -> https://docs.python.org/3.6/tutorial/floatingpoint.html
    '''
    if len(str(number).split('.')[1])==8 and str(number).endswith('5'):
        return(number+0.00000005)
    else:
        return round(number,7)

if __name__=='__main__':
##	root = tk.Tk()
##	root.withdraw()
##	print ('A abrir ficheiro...')
##	file = filedialog.askopenfilename(filetypes = [("Excel","*.xlsx")])
##	file2 = filedialog.askopenfilename(filetypes = [("TSV","*.tsv")])


	df = pd.read_excel(file)
	df['m/z'] = df['m/z'].apply(round_up)
	df['m/z'] = df['m/z'].round(7)


	if 'Use' in list(df):
		df = df.drop('Use', axis=1)
	if 'Samples' in list(df):
		df = df.drop('Samples', axis=1)


	l=[]
	with open(file2) as f:
		for x in f:
			try:
				l.append(float(x.split('\t')[0]))
			except ValueError:
				continue


	col = list(df)[1:]
	dfp = df[df['m/z'].isin(l)]


	dfp.to_csv(file[:-5]+'_metaboanalyst_clean.csv', index=False, header=True)


	temp = []
	with open(file[:-5]+'_metaboanalyst_clean.csv') as f:
		for x in f:
			if x.startswith('m/z'):
				temp.append('Sample,'+x.split(',',1)[1])
				temp.append('Label,I,I,I,Mock,Mock,Mock\n')
			else:
				temp.append(x)
	with open(file[:-5]+'_metaboanalyst_clean.csv', 'w') as f:
		f.write("".join(temp))


	if len(l) == len(dfp):
		print ('Converted to MetaboAnalyst')
	else:
		print('Converted with some missing m/z')
		print("Not in Mastrix file:")
		for x in l:
			if x not in a:
				print (x)