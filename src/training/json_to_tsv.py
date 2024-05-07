import json
import pandas as pd
from sklearn.model_selection import train_test_split

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Example usage
file_path = '../../blip_laion_cc_sbu_20k.json'
json_data = read_json_file(file_path)
df = pd.DataFrame(columns=['file','caption'])
df['file'] = json_data.keys()
df['caption'] = json_data.values()

# replace the file column with the full path
df['file'] = df['file'].apply(lambda x: '../../generated_images_20k/'+x+'.jpg')

# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.1)
train.to_csv('../../CC3M_data/train_with_cap.csv',index=False,sep='\t')
test.to_csv('../../CC3M_data/val_with_cap.csv',index=False,sep='\t')