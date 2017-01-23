from __future__ import print_function
import requests

print('Fetching ID translation table HMDB->Kegg from Kegg...', end=' ')
fname = 'trans_hmdb2kegg.txt'
h = requests.post('http://rest.genome.jp/link/compound/hmdb').text
with open(fname, 'w') as f:
    f.write(h)
print('Done\nFile {} created.\n'.format(fname))

print('Fetching ID translation table LIPIDMAPS->Kegg from Kegg...', end=' ')
fname = 'trans_lipidmaps2kegg.txt'
h = requests.get('http://rest.genome.jp/link/compound/lipidmaps').text
with open(fname, 'w') as f:
    f.write(h)
print('Done\nFile {} created.\n'.format(fname))

