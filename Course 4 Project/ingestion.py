import os
import json
import pandas as pd


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    final_df = pd.DataFrame()
    list_files = []
    filenames = os.listdir(os.path.join(os.getcwd(), input_folder_path))
    for filename in filenames:
        df = pd.read_csv(os.path.join(os.getcwd(), input_folder_path, filename))
        final_df = pd.concat([final_df, df], ignore_index=True)
        list_files.append(filename+'\n')

    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    final_df.drop_duplicates(inplace=True)
    final_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)

    with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        f.writelines(list_files)


if __name__ == '__main__':
    merge_multiple_dataframe()