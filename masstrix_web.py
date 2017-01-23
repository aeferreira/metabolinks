import requests
import pandas as pd
from io import StringIO
import time
import tkinter as tk
from tkinter import filedialog


def load_mass_list():
	'''
	Load mass list from .txt file.
	File must have one mass per line.
	'''
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(filetypes = [("TXT","*.txt")])
    l=[]
    with open(file) as f:
        for x in f:
            l.append(x.strip('\n'))
    return l


def save_as_tsv(title='Save file as'):
	'''
	Get the file path to save the file.
	'''
    ftypes = [('TSV', '.tsv'), ('All files', '*')]
    filename = filedialog.asksaveasfilename(filetypes=ftypes, title=title,
                                            defaultextension='.tsv')
    return filename


def masstrix_job(mass_list, mode, ppm='2.0', natrium=False, potassium=False, chlorine=False, bromine=False, name='no job-id', save=False):
    '''
    Web request to MassTRIX server.
    Return pandas dataframe with the results.
    
    Parameters:
        mass_list : list of masses to send to MassTrix server
        mode : 'pos' (positive) or 'neg' (negative) scan mode
        ppm : Max. error relative to input mass, default='2.0'
        natrium : ('on'/'undef') use Natrium adduct, default: 'on' in positive mode, 'off' in negative
        potassium : ('on'/'undef') use Potassium adduct, default: 'on' in positive mode, 'off' in negative
        chlorine : ('on'/'undef') use Chlorine adduct, default: 'on' in negative mode, 'off' in positive
        bromine : ('on'/'undef') use Bromine adduct, default: 'off' in negative mode, 'off' in positive
        name : job identifier name, default='no job-id'
        save : (True/False) save the file to .tsv
        
    In masstrix 'undef' = 'off' and/or None.
    '''
    
#     Set default values in positive and negative modes
    if natrium:
        natrium='on'
    elif not natrium and mode=='pos':
        natrium='on'
    else:
        natrium='undef'


    if potassium:
        potassium='on'
    elif not potassium and mode=='pos':
        potassium='on'
    else:
        potassium='undef'        


    if chlorine:
        chlorine='on'
    elif not chlorine and mode=='neg':
        chlorine='on'
    else:
        chlorine='undef' 


    if bromine:
        bromine='on'
    else:
        bromine='undef'

#     Print settings
    
    print('Number of masses = '+str(len(mass_list)))
    print('Mode = '+mode)
    print('Natrium = '+natrium)
    print('Potassium = '+potassium)
    print('Chlorine = '+chlorine)
    print('Bromine = '+bromine)
    print('Name = '+name)
    
#     Requests payload
    
    payload = {'TASK':'NEW',
               'PASTEMASSLIST':'\n'.join(mass_list),
               'MODE':mode,
               'natrium':natrium,
               'potassium':potassium,
               'bromine':bromine,
               'chlorine':chlorine,
               'PPM':ppm,
               'ABSOLUTE_M_Z':'undef',
               'DB':'KEGGlipidnoiso',
               'ORG':'vvi',
               'JOBID':name,
               'AFFYCHIPTYPE':'undef',
               'foldChange':'0.5',
               'foldcolorup': 'red',
               'foldcolordown':'blue',
               'foldcolornorm':'no'
              }
    r = requests.post("http://masstrix3.helmholtz-muenchen.de/masstrix3/run.cgi", data=payload)
    for x in r.text.splitlines():
        if '<A HREF=run.cgi?ID=' in x:
            job=x.split('ID=')[1].split('>')[0]
            print ('Job ID =', job)
    q = requests.get('http://masstrix3.helmholtz-muenchen.de/masstrix3/users/'+job)
    while True:
        q = requests.get('http://masstrix3.helmholtz-muenchen.de/masstrix3/users/'+job)
        if 'masses.annotated.reformat.tsv' in q.text:
            print('\nMassTrix finished')
            break
        else:
            print('Waiting for MassTrix...')
            time.sleep(30)
    u = requests.get('http://masstrix3.helmholtz-muenchen.de/masstrix3/users/'+job+'/masses.annotated.reformat.tsv')
    l=[]
    for x in u.content.decode().splitlines():
        l.append(x)
    l2 = [l[-1]]
    l2.extend(l[:-1])
    df = pd.read_csv(StringIO('\n'.join(l2)), sep='\t')
    if save:
        filename = save_as_tsv()
        with open(filename, "wb") as f:
            f.write(u.content)
        
    return df


# Examples:
# masstrix_job(load_mass_list(),'pos')
# masstrix_job(load_mass_list(),'pos',ppm='1.0',save=True)
# masstrix_job(load_mass_list(),'neg',ppm='1.0',bromine='on',chlorine='on',name='teste_name',save=True)