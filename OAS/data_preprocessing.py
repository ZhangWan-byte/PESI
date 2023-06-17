import os
import glob
import pickle
import pandas as pd

from tqdm import tqdm


csvs = glob.glob("./*.csv")

df_list = []
for csv in tqdm(csvs):
    tmp = pd.read_csv(csv, sep=',', header=1)
    df_list.append(tmp)

df = pd.concat(df_list)
df = df.reset_index()

data = []
error_idx = []
for i in tqdm(range(len(df))):
    try:
        tmp = "/".join([df["cdr1_aa_heavy"].iloc[i], 
                        df["cdr2_aa_heavy"].iloc[i], 
                        df["cdr3_aa_heavy"].iloc[i], 
                        df["cdr1_aa_light"].iloc[i], 
                        df["cdr2_aa_light"].iloc[i], 
                        df["cdr3_aa_light"].iloc[i]])
    except:
        error_idx.append(i)
    data.append(tmp)

pickle.dump(data, open("./oas_data.pkl", "wb"))