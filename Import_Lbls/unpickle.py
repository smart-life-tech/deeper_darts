'''
Download 'cropped_images.zip' from:
https://ieee-dataport.org/open-access/deepdarts-dataset
and use this script to unpickle it, and save it to a tab seperated format
'''

import pickle
import ssl; print(ssl.OPENSSL_VERSION)
# Load the pickled DataFrame
with open('c:\\Users\\USER\\Documents\\raspberrypi\\dart\\darts2\\deeper_darts\\datasets\\labels.pkl', 'rb') as f:
    data = pickle.load(f)

data.to_csv('all_labels.tsv', sep='\t', index=False)   # tab-separated format with index
