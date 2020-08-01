from collections import namedtuple
import time
import os

import requests

import pandas as pd

# ------------- Local database loading ----------------------------------

LocalDBs = namedtuple('LocalDBs', ['kegg_db', 'lm_df',
                                   'hmdb2kegg', 'lm2kegg',
                                   'kegg2lm', 'kegg2brite',
                                   'new_kegg_records',])

_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dbs'))

def load_local_dbs():
    print ('\nLoading local DBs')
    
    kegg_df_fname = os.path.join(_DB_DIR, 'kegg_db.txt')
    kegg_db = load_local_kegg_db(kegg_df_fname)

    lm_db_fname = (os.path.join(_DB_DIR, 'lm_metadata.txt'))
    lm_df = pd.read_csv(lm_db_fname, index_col=0, dtype={'CHEBI_ID':str}, sep='\t')
    #lm2kegg = lm_df['KEGG_ID'].dropna() # a pandas Series

    trans_hmdb2kegg_fname = os.path.join(_DB_DIR, 'trans_hmdb2kegg.txt')
    table = pd.read_csv(trans_hmdb2kegg_fname, sep='\t', index_col=None)
    table.columns = ['hmdb', 'kegg', 'equiv']
    table = table.drop(columns='equiv')
    table['hmdb'] = table['hmdb'].str.split(':').str.get(1)
    table['kegg'] = table['kegg'].str.split(':').str.get(1)
    hmdb2kegg = table.set_index('hmdb') # a pandas Series

    trans_lm2kegg_fname = os.path.join(_DB_DIR, 'trans_lm2kegg.txt')
    table = pd.read_csv(trans_lm2kegg_fname)

    lm2kegg = table.set_index('lm')['kegg'] # a pandas Series
    #kegg2lm = table.set_index('kegg')['lm'] # a pandas Series
    kegg2lm = table.drop_duplicates(subset='kegg', keep='first').set_index('kegg')['lm']

    new_kegg_records = []

    kegg2brite_fname = os.path.join(_DB_DIR, 'kegg2brite.csv')
    kegg2brite = pd.read_csv(kegg2brite_fname)
    kegg2brite = kegg2brite.set_index('KeGG CId')
    kegg2brite = kegg2brite.sort_index()

    return LocalDBs(kegg_db, lm_df, hmdb2kegg,
                    lm2kegg, kegg2lm,
                    kegg2brite,
                    new_kegg_records,)


def load_local_kegg_db(db_fname):
    """Load Kegg compound records from a local text DB."""

    with open(db_fname) as whole_file:
        records = whole_file.read().split('---- Compound:')
    kegg = {}
    records = [r.lstrip() for r in records]
    records = [r for r in records if r]
    kegg = {}
    for block in records:
        firstline, rest = block.split('\n', 1)
        kegg[firstline.strip()] = rest
    return pd.Series(kegg)

# ------------- KeGG-compound record parsing -------------------

def get_kegg_section(k_record, sname, whole_section=False):
    """Get the section with name `sname` from a kegg compound record.

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

def request_kegg_record(c_id, local_dbs):
    if c_id in local_dbs.kegg_db:
        return local_dbs.kegg_db[c_id]
    else:
        record = requests.get('http://rest.kegg.jp/get/' + c_id).text
        outrecord = '---- Compound: {}\n'.format(c_id) + record
        local_dbs.new_kegg_records.append(outrecord)
        return record

# ------------- compound annotation -------------------

def get_identifiers_from_id(compound_id, local_dbs, already_fetched, trace=False):
    c_id = None
    lm_id = None
    hmdb_id = None
    ks_id = None
    primary_id = None

    if trace:
        print('\n------------ {}'.format(compound_id))

    if compound_id.startswith('C'):
        c_id = compound_id
        primary_id = c_id
        lm_id = local_dbs.kegg2lm.get(c_id)

    elif compound_id.startswith('LM'):
        lm_id = compound_id
        primary_id = lm_id
        c_id = local_dbs.lm2kegg.get(lm_id)

    elif compound_id.startswith('HMDB'):
        hmdb_id = compound_id
        primary_id = hmdb_id
        c_id = local_dbs.hmdb2kegg.get(hmdb_id)
        if c_id is not None:
            lm_id = local_dbs.kegg2lm.get(c_id)

    else:
        raise ValueError('Cannot resolve compound Id {}'.format(compound_id))

    # if type(c_id) == pd.Series:
    #     print('Series c_id', c_id)
    # if type(lm_id) == pd.Series:
    #     print('Series lm_id', lm_id)
    if c_id in already_fetched or lm_id in already_fetched:
        if trace:
            print('{} already looked up before, skipping'.format(primary_id))
            cformat = '---- compound: KeGG={} LM={}, HMDB={}'.format
            print(cformat(c_id, lm_id, hmdb_id))
        return None

    # if c_id is not None:
    #     krecord = request_kegg_record(c_id, local_dbs)
    #     if trace:
    #         print('(retrieving KNApSAcK from {})'.format(c_id))

    #     dblinks = get_kegg_section(krecord, 'DBLINKS')

    #     if len(dblinks) > 0:
    #         if 'KNApSAcK' in dblinks:
    #             for line in dblinks.splitlines():
    #                 if 'KNApSAcK:' in line:
    #                     ks_id = line.split(':')[1].strip()
        
    #     # find if present in plants
    #     if ks_id is not None:
    #         r = requests.post('http://kanaya.naist.jp/knapsack_jsp/information.jsp?sname=C_ID&word=' + ks_id)
    #         if 'Plantae' in r.text:
    #             ks_id = ks_id + ' in Plantae'
    #             if trace:
    #                 print(ks_id)

    if trace:
        print('Primary identifier: {}'.format(primary_id))
        # cformat = '---- compound: KeGG={} LM={}, HMDB={}, KNApSAcK={}'.format
        cformat = '---- compound: KeGG={} LM={}, HMDB={}'.format
        print(cformat(c_id, lm_id, hmdb_id))

    for db_id in (c_id, lm_id, hmdb_id):
        if db_id is not None:
            already_fetched.append(db_id)

    returndict = {'kegg_id': c_id, 
                  'lm_id': lm_id,
                  'hmdb_id': hmdb_id,
                  'primary': primary_id,}
    return returndict


def get_identifiers(compound_ids, local_dbs, trace=False):
    already_fetched = []
    result = []
    for compound_id in compound_ids:
        rd = get_identifiers_from_id(compound_id, local_dbs, already_fetched, trace=trace)
        if rd is not None:
            result.append(rd)
    return pd.DataFrame(result).set_index('primary')


def get_LM_taxonomy(lm_id, local_dbs):
    row = local_dbs.lm_df.loc[lm_id]
    name = row['NAME']
    sname = row['SYSTEMATIC_NAME']
    abbrev = row['ABBREVIATION']
    if pd.isnull(name):
        name = sname
    if pd.isnull(name):
         name = abbrev
    row = row[['CATEGORY', 'MAIN_CLASS', 'SUB_CLASS']]
    row = pd.Series([name, 'Lipids'] + list(row.values),
                     index=['Names', 'Ontology', 'Category', 'Main class', 'Subclass'])
    return row

def substitute_lipids_ontology(identifier, primary, ontologies, local_dbs):
    if not identifier in local_dbs.lm_df.index:
        return ontologies
    lmtax = get_LM_taxonomy(identifier, local_dbs)
    lmtax.name = primary
    lmtax = lmtax.to_frame().T
    if len(ontologies.index) == 0:
        return lmtax
    elif 'Lipids' in ontologies['Ontology'].values:
        ontologies.loc[ontologies['Ontology'] == 'Lipids'] = lmtax
    else:
        ontologies = pd.concat([ontologies, lmtax])
    return ontologies

def get_taxonomy_from_identifiers(identifier_table, local_dbs, skip_ontologies=None, trace=False):
    if skip_ontologies is None:
        skip_ontologies = tuple()
    if isinstance(skip_ontologies, str):
        skip_ontologies = [skip_ontologies]
    alltax = []

    for primary, row in identifier_table.iterrows():
        if trace:
            print('---------- {}'.format(primary))
        ontologies = pd.DataFrame()

        if primary.startswith('LM'):
            ontologies = substitute_lipids_ontology(primary, primary, ontologies, local_dbs)
        else:
            if primary.startswith('C'):
                table = local_dbs.kegg2brite
                ontologies = table.iloc[table.index.isin([primary])]
                ontologies = ontologies[['Names', 'Ontology', 'Category', 'Main class', 'Subclass']]

                if row.lm_id is not None:
                    ontologies = substitute_lipids_ontology(row.lm_id, primary, ontologies, local_dbs)

        if len(ontologies.index) > 0:
            ontologies = ontologies[~ontologies['Ontology'].isin(skip_ontologies)]

        if len(ontologies.index) > 0:
            if trace:
                print(ontologies)
            alltax.append(ontologies)

    return pd.concat(alltax)

def generate_taxonomy(identifiers, local_dbs=None, skip_ontologies=None, trace=False):
    if local_dbs is None:
        local_dbs = load_local_dbs()
    
    id_table = get_identifiers(identifiers, local_dbs, trace=trace)
    annotations = get_taxonomy_from_identifiers(id_table, local_dbs,
                                                skip_ontologies=skip_ontologies,
                                                trace=trace)

    # print('updating local Kegg DB')
    if len(local_dbs.new_kegg_records) > 0:
        kegg_df_fname = os.path.join(_DB_DIR, 'kegg_db.txt')
        with open(kegg_df_fname, 'a') as kf:
            kf.write('\n')
            kf.write('\n'.join(local_dbs.new_kegg_records))

    # concat extra columns with other ids
    #annotations.info()
    #id_table.info()
    annotations = annotations.merge(id_table, how='left', left_index=True, right_index=True)
    return annotations


if __name__ == '__main__':

    from metabolinks import datasets

    dbs = load_local_dbs()

    # for k, v in dbs._asdict().items():
    #     print(f'----- {k}')
    #     print(v)
    #     print('*'*30)

    df = datasets.demo('masstrix_output')
    print ("Demo data:\n")

    df.info()
    print('************************')
    print(df.KEGG_cid)
    print('************************')

    # get one identifier per row
    cids = df.KEGG_cid.str.split('#').explode()
    print('\n\n----- Identifier list')
    print(cids)
    print('-----------------------------------')

    # build identifier translation table
    identifiers = get_identifiers(cids, dbs, trace=False)

    print('\n\n----- Identifiers translation table')
    print(identifiers)
    print('-----------------------------------')

    annotations = get_taxonomy_from_identifiers(identifiers, dbs, trace=False)

    print('Annotations:\n')
    print(annotations)

    print('+++++++++++++++++++++++++++++++++++')
    annotations = generate_taxonomy(cids, skip_ontologies='Carcinogens')

    print('Annotations:\n')
    print(annotations)
    annotations.to_csv('demo_taxonomy.csv')
