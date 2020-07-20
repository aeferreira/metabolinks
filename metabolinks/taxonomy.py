from __future__ import print_function
import time
import os

import requests

import pandas as pd
# ------------- util to clean up MassTRiX columns

def cleanup_cols(df, isotopes=True, uniqueID=True, columns=None):
    """Removes the 'uniqueID' and the 'isotope presence' columns."""
    col_names = []
    if uniqueID:
        col_names.append("uniqueID")
    if isotopes:
        iso_names = (
            "C13",
            "O18",
            "N15",
            "S34",
            "Mg25",
            "Mg26",
            "Fe54",
            "Fe57",
            "Ca44",
            "Cl37",
            "K41",
        )
        col_names.extend(iso_names)
    if columns is not None:
        col_names.extend(columns)
    return df.drop(col_names, axis=1)

# ------------- Local database loading ----------------------------------

class LocalDBs(object):
    def __init__(self,
                 kegg_db,
                 lm_df,
                 hmdb2kegg_dict,
                 lipidmaps2kegg_dict,
                 kegg2lipidmaps_dict):

        self.kegg_db = kegg_db
        self.lm_df = lm_df
        self.hmdb2kegg_dict = hmdb2kegg_dict
        self.lipidmaps2kegg_dict = lipidmaps2kegg_dict
        self.kegg2lipidmaps_dict = kegg2lipidmaps_dict
        self.new_kegg_records = []

_local_dbs = None
_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dbs'))
        
def load_local_dbs():
    print ('\nLoading local DBs')
    
    kegg_df_fname = os.path.join(_DB_DIR, 'kegg_db.txt')
    kegg_db = load_local_kegg_db(kegg_df_fname)
    
    lm_db_fname = (os.path.join(_DB_DIR, 'lm_metadata.txt'))
    lm_df = load_local_lipidmaps_db(lm_db_fname)    
    
    trans_hmdb2kegg_fname = os.path.join(_DB_DIR, 'trans_hmdb2kegg.txt')
    hmdb2kegg_dict = get_trans_id_table(trans_hmdb2kegg_fname)

    lipidmaps2kegg_dict = lm_df['KEGG_ID'].dropna().to_dict()

    kegg2lipidmaps_dict = {}
    for k, v in lipidmaps2kegg_dict.items():
        kegg2lipidmaps_dict[v] = k
    
    return LocalDBs(kegg_db, lm_df, hmdb2kegg_dict,
                    lipidmaps2kegg_dict, kegg2lipidmaps_dict)


def load_local_kegg_db(db_fname):
    """Load Kegg compound records from a local text DB."""
    
    kegg = {}
    db = open(db_fname)
    curr_c = None
    curr_txt = None
    for line in db:
        if len(line.strip()) == 0:
            continue
        if line.startswith('---- Compound:'):
            if curr_c is not None:
                whole_rec = ''.join(curr_txt)
                kegg[curr_c] = whole_rec
            curr_c = line.split(':')[1].strip()
            curr_txt = []
        elif curr_c is not None:
            curr_txt.append(line)
    whole_rec = ''.join(curr_txt)
    kegg[curr_c] = whole_rec
    db.close()
    return kegg


def load_local_lipidmaps_db(db_fname):
    """Load LIPIDMAPS records from a local text DB."""
    df = pd.read_table(db_fname, index_col=0, dtype={'CHEBI_ID':str})
    return df
    
# ------------- Code related to Kegg compound records -------------------

def get_kegg_section(k_record, sname, whole_section=False):
    """Get the section with name _sname_ from a kegg compound record.
    
       Returns a str, possibly empty."""
    
    in_section = False
    section = []
    
    for line in k_record.splitlines():
        if line.startswith(sname):
            in_section = True
            section.append(line)
        elif in_section and line.startswith(' '):
            section.append(line)
        elif in_section and not line.startswith(' '):
            break

    if whole_section:
        sectionlines = section
    else:
        sectionlines = [line[12:] for line in section]
    return '\n'.join(sectionlines)

def request_kegg_record(c_id, localdict, local_dbs):
    if c_id in localdict:
        return localdict[c_id]
    else:
        record = requests.get('http://rest.kegg.jp/get/' + c_id).text
        outrecord = '---- Compound: {}\n'.format(c_id) + record
        local_dbs.new_kegg_records.append(outrecord)
        return record

# Load ID translation tables as dicts
# IMPORTANT: Use local files, (fetched by fetch_dbs.py)
def get_trans_id_table(fname):
    d = {}
    with open(fname) as f:
        # hmdb:HMDB00002 \t cpd:C00986 \t equivalent
        for line in f:
            if len(line) == 0:
                continue
            foreign, cpd, _ = line.split('\t')
            foreign = foreign.split(':')[1].strip()
            cpd = cpd.split(':')[1].strip()
            d[foreign] = cpd
    return d


class Progress(object):
    """A progress reporter of % done for a maximum (int) of self.total"""
    def __init__(self, total=1):
        self.reset(total)
    def tick(self):
        self.count +=1
        print(str(round((1-((self.total-self.count)/self.total))*100)) + "% done")
    def reset(self, total=1):
        self.total = total
        self.count = 0


def classes_from_brite(krecord, brite_blacklist=None, trace=False):
    """Parses classes from the BRITE section of a kegg record."""
    classes = [[], [], [], []]

    # replace word BRITE by spaces
    krecord = ' '* 5 + krecord[5:]

    # split by 11 spaces (and discard ":" ), remove trailing \n
    cstrings = [c.rstrip() for c in krecord.split('           ')[1:]]

    brite_id = ''
    for x in cstrings:
##         print('*******************************')
##         print('|{}|'.format(x))
##         print('*******************************')
        
        if x.startswith(' ') and not x.startswith('  '):
            if '[BR:' in x:
                brite_id = x.split('[BR:')[1].split(']')[0]
            elif '[br' in x:
                brite_id = 'br' + x.split('[br')[1].split(']')[0]
            else:
                brite_id = 'br?????'
                
            classes[0].append(x[1:])
        elif x.startswith('  ') and not x.startswith('   '):
            classes[1].append(x[2:])
        elif x.startswith('   ') and not x.startswith('    '):
            classes[2].append(x[3:])
        elif x.startswith('    ') and not x.startswith('     '):
            classes[3].append(x[4:])
    if brite_blacklist is not None:
        if brite_id in brite_blacklist:
            classes = [['null'], [brite_id], [], []]

    classes = ['#'.join(c) for c in classes]
    if trace:
        print ('from BRITE (with id {}):'.format(brite_id))
        print(tuple(classes))
    return tuple(classes)


def classes_from_lipidmaps(lm_id, lm_df, trace=False):
    if trace:
        print('LIPIDMAPS id: {}'.format(lm_id))
##     f = requests.get('http://www.lipidmaps.org/rest/compound/lm_id/' + lm_id + '/all/json').text
##     s = json.loads(f)
##     if f == '[]':
##         return 'null', 'null', 'null', 'null'
##     mm = 'Lipids [LM]'
##     cc = s['core'] if s['core'] is not None else 'null'
##     ss = s['main_class'] if s['main_class'] is not None else 'null'
##     tt = s['sub_class'] if s['sub_class'] is not None else 'null'
    if lm_id not in lm_df.index:
        cc = ['Lipids [LM]', 'null', 'null', 'null']
    else:
        record = lm_df.loc[lm_id]
        cc = ['Lipids [LM]']
        for field in 'CATEGORY', 'MAIN_CLASS', 'SUB_CLASS':
            cc.append(record[field] if pd.notnull(record[field]) else 'null')
    a = tuple(cc)
    if trace:
        print(a)
    return a 


def annotate_compound(compound_id, c_counter, already_fetched, 
                      trace=False, local_dbs=None, brite_blacklist=None):
    c_id = None
    lm_id = None
    hmdb_id = None
    ks_id = None
    
    brite = ''
    dblinks = ''
    in_plants = ''
    
    trans_kegg = compound_id
    trans_lipidmaps = compound_id
    
    if compound_id.startswith('C'):
        c_id = compound_id
        if c_id in local_dbs.kegg2lipidmaps_dict:
            lm_id = local_dbs.kegg2lipidmaps_dict[c_id]
            trans_lipidmaps = lm_id
    
    elif compound_id.startswith('LM'):
        lm_id = compound_id
        if lm_id in local_dbs.lipidmaps2kegg_dict:
            c_id = local_dbs.lipidmaps2kegg_dict[lm_id]
            trans_kegg = c_id
    
    elif compound_id.startswith('HMDB'):
        hmdb_id = compound_id
        if hmdb_id in local_dbs.hmdb2kegg_dict:
            c_id = local_dbs.hmdb2kegg_dict[hmdb_id]
            trans_kegg = c_id
            if c_id in local_dbs.kegg2lipidmaps_dict:
                lm_id = local_dbs.kegg2lipidmaps_dict[c_id]
                trans_lipidmaps = lm_id

    else:
        pass # does MASSTrix use any other DBs for IDs?
    
    if trace:
        print('\n---- compound: {} ({}, {})'.format(compound_id, 
                                             trans_kegg, 
                                             trans_lipidmaps))
    
    if c_id in already_fetched or lm_id in already_fetched:
        if trace:
            print('already looked up')
        return (trans_kegg, trans_lipidmaps, '', '', '', '', '')
        
    if c_id is not None:
        krecord = request_kegg_record(c_id, local_dbs.kegg_db, local_dbs)
        if trace:
            print('(look up {})'.format(c_id))
        c_counter.inc_looks()
        already_fetched.append(c_id)
    
        brite = get_kegg_section(krecord, 'BRITE', whole_section=True)
        dblinks = get_kegg_section(krecord, 'DBLINKS')
        
        if len(dblinks) > 0:
            # find if there are LIPIMAPS or KNApSAcK Xrefs in DBLINKS section
            if 'LIPIDMAPS:' in dblinks:
                for line in dblinks.splitlines():
                    if 'LIPIDMAPS:' in line:
                        lm_id = line.split(':')[1].strip()
                        trans_lipidmaps = lm_id
            if 'KNApSAcK' in dblinks:
                for line in dblinks.splitlines():
                    if 'KNApSAcK:' in line:
                        ks_id = line.split(':')[1].strip()
    
    # get compound taxonomy
    
    # first from BRITE section
    if (c_id is not None) and len(brite) > 0:
        classes = classes_from_brite(brite, 
                                     brite_blacklist=brite_blacklist,
                                     trace=trace)
        if classes[0] == 'null':
            if trace:
                print('BRITE class {} in blacklist. Skipped'.format(classes[1]))
            classes = ('', '', '', '')
    
    # then from LIPIDMAPS
    elif (lm_id is not None):
        if trace:
            print('(look up {})'.format(lm_id))
        classes = classes_from_lipidmaps(lm_id,
                                         local_dbs.lm_df,
                                         trace=trace)
        c_counter.inc_looks()
        already_fetched.append(lm_id)

    else:
        classes = ('', '', '', '')
        if trace:
            print ('No BRITE, no LIPIDMAPS')
    
    # find if present in plants
    in_plants = ''
    if ks_id is not None:
        r = requests.post('http://kanaya.naist.jp/knapsack_jsp/information.jsp?sname=C_ID&word=' + ks_id)
        c_counter.inc_looks()
        if 'Plantae' in r.text:
            in_plants = 'Plantae'
            if trace:
                print(in_plants)
    
    result = [trans_kegg, trans_lipidmaps]
    result.extend(list(classes))
    result.append(in_plants)
    return tuple(result)

class _count_compounds(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.looks_count = 0
    def inc(self, n=1):
        self.count += n
    def inc_looks(self, n=1):
        self.looks_count += n
    def get_count(self):
        return self.count
    def get_looks_count(self):
        return self.looks_count


def annotate_all(peak, c_counter, progress=None, trace=False,
                 local_dbs=None, brite_blacklist=None):
    """Create Pandas Series with compound class annotations."""
    
    data = [[], [], [], [], [], [], []]
    
    if trace:
        print('\n++++++++ PEAK +++++++++++++')
    
    already_fetched = []
    
    for compound_id in peak.split('#'):
        c_counter.inc()
        
        d_compound = annotate_compound(compound_id, 
                                       c_counter, 
                                       already_fetched, 
                                       trace, local_dbs,
                                       brite_blacklist)
        
        for d, i in zip(data, d_compound):
            if len(i) > 0: # don't include empty strings
                d.append(i)
    
    if progress is not None:
        progress.tick()
    
    compressed_data = []
    for i, d in enumerate(data):
        if i < 2:
            compressed_data.append(d)
        else:
            compressed_data.append(set(d))
    
    hash_data = ['#'.join(d) for d in compressed_data]    

    col_names = ['trans_KEGG_Ids',
                 'trans_LipidMaps_Ids',
                 'Major Class', 
                 'Class', 
                 'Secondary Class', 
                 'Tertiary Class',
                 'KNApSAcK']
    
    
    if trace:
        print('\nDATA:')
        for n, d in zip(col_names, hash_data):
            print('{:25s} : {}'.format(n, d))
    
    return pd.Series(hash_data, index=col_names)


def insert_taxonomy(df, brite_blacklist=None, trace=False):
    """Apply annotate_all to kegg or LIPIDMAPS ids."""
    
    progress = Progress(total=len(df['raw_mass']))
    c_counter = _count_compounds()
    local_dbs = load_local_dbs()
    
    if brite_blacklist is not None:
        with open(brite_blacklist) as f:
            brite_blacklist = f.read().splitlines()
    
    df = pd.concat([df, df['KEGG_cid'].apply(annotate_all, 
                                             args=(c_counter, progress,
                                             trace,
                                             local_dbs,
                                             brite_blacklist))], axis=1)
    
    frmt = '\nDone! {} ids processed. {} DB lookups'.format
    print(frmt(c_counter.get_count(), c_counter.get_looks_count()))
    print('updating local Kegg DB')
    kegg_df_fname = os.path.join(_DB_DIR, 'kegg_db.txt')
    with open(kegg_df_fname, 'a') as kf:
        kf.write('\n')
        kf.write('\n'.join(local_dbs.new_kegg_records))
    print('\nDone')
    return df

if __name__ == '__main__':

    print ('\nStarting...\n')
    #import six
    #from metabolinks.dataio import read_MassTRIX
    from metabolinks import datasets

    start_time = time.time()

    df = datasets.demo('masstrix_output')
    #df = read_MassTRIX(six.StringIO(datasets.MassTRIX_output()))
    print ("Data was read\n")

    # Clean up uniqueId and "isotope" cols
    cdf = cleanup_cols(df)

    # check result
    cdf.info()
    assert len(cdf.columns) == 12 # there are 24 - 12 = 12 columns
    assert len(cdf.index) == 15 # there are still 15 peaks

    print ('Starting annotations...')

    # Call the main driver function.
    results = insert_taxonomy(cdf, trace=True)

    elapsed_time = time.time() - start_time
    m, s = divmod(elapsed_time, 60)
    print ("Finished in " + "%02dm%02ds" % (m, s))
    print('------------------------------------')

    results.info()
    
    # Export the annotated dataframe into a MS-Excel file
    # Name it with the same name as the .tsv, replacing tail with '_raw.xlsx'
    
    out_fname = 'results_comptaxa.xlsx'
    results.to_excel(out_fname, header=True, index=False)
    print (f"File {out_fname} written")
