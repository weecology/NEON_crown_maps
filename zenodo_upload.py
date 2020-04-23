### Zenodo upload
import requests
import os

def get_token():
    token = os.environ.get('ACCESS_TOKEN')
    return token

def upload(path):
    """Upload an item to zenodo"""
    
     # Get the deposition id from the already created record
    deposition_id = "3764116"
    data = {'name': os.path.basename(path)}
    files = {'file': open(path, 'rb')}
    r = requests.post('https://zenodo.org/api/deposit/depositions/%s/files' % deposition_id,
                      params={'access_token': ACCESS_TOKEN}, data=data, files=files)
    r.json()
    
if __name__== "__main__":
    ACCESS_TOKEN = get_token()
    BASE_DIR = "/orange/idtrees-collab/dataset/"
    
    files_to_upload = glob.glob(BASE_DIR + "*")
    for f in files_to_upload:
        upload(f)
