from azure.storage.blob import BlockBlobService
import tables
from io import StringIO


STORAGEACCOUNTNAME= 'recommend-test'
STORAGEACCOUNTKEY= 'eZ6rUu9dIV3I0ZZgyXDm2yAe+dJJ8m7C3YTlMuOUqHD7EWck9hLFVEamxJa9RQgIty81t32zNPjUie9Mt4rd9Q=='
LOCALFILENAME= 'data'
CONTAINERNAME= 'data'
BLOBNAME= 'Guest Recommender.csv'
print('HI')
#download from blob
# t1=time.time()
blob_service=BlockBlobService(account_name=STORAGEACCOUNTNAME,account_key=STORAGEACCOUNTKEY)
blobstring = blob_service.get_blob_to_text(CONTAINERNAME,BLOBNAME)
df = pd.read_csv(StringIO(blobstring))
print(df.head())