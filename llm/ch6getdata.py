import urllib.request
import zipfile
import os
from pathlib import Path

import pandas as pd

###############################################
#
#
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

###############################################
#
#    
def read_data():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "data"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        print(f"Error {e}")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)        
            
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label","Text"])
    print(df)

    print(df["Label"].value_counts())

    return df

###############################################
#
#
def create_balanced_dataset(df):
    num_spam = df[ df["Label"] == "spam"  ].shape[0]

    ham_subset = df[ df["Label"] == "ham" ].sample(num_spam, random_state=123)

    balanced_df = pd.concat([ham_subset, df[ df["Label"] == "spam" ] ])

    print(balanced_df["Label"].value_counts())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    print(balanced_df)
        
    return balanced_df

###############################################
#
#
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df
    
###############################################
#
#
if __name__ == "__main__":
    df = read_data()
    b_df = create_balanced_dataset(df)

    train_df, validation_df, test_df = random_split(b_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
    
    
