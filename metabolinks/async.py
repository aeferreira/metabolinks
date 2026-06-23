""" Asynchronously fetch data bases and unzip the files """

import asyncio
import requests
from zipfile import ZipFile
import aiofiles, aiohttp
import logging
import time

async def fetch_db(url_list, **kwargs):
    # download the file contents in binary format
    print('Downloading...')
    for url in url_list:
        r = await requests.get(url)
        return r

def unzip(file_name, response):
    # open method to open a file on your system and write the contents
    with open(file_name, "wb") as f:
        f.write(response.content)

    # opening the zip file in READ mode 
    with ZipFile(file_name, 'r') as zip: 
        # printing all the contents of the zip file 
        zip.printdir() 
  
        # extracting all the files 
        print('Extracting all the files now...') 
        zip.extractall() 
        print('Done!')

async def main():
    start=time.time()

    urls= ['http://www.hmdb.ca/system/downloads/current/hmdb_metabolites.zip','http://www.lipidmaps.org/resources/downloads/LMSDFDownload12Dec17.zip']
    print('Fetching')
    asyncio.create_task( fetch_db(urls))
    unzip('hmbd_db.zip', fetch_db(urls))
    unzip('LIPIDMAPS_db.zip', fetch_db(urls))
    
    end=time.time()
    print('It took', end-start, 'seconds to fetch these data bases.')

asyncio.run(main())