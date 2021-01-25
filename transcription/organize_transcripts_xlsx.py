import os
from collections import defaultdict
import pandas as pd

base_folder_path = "brother_transcripts/transcript/"
folders = ["DeepSpeech_with_vad", "Wav2Letter", "Wav2Letter_with_vad"]
dic = defaultdict(dict)

for folder in folders:
    curr_path = os.path.join(base_folder_path, folder)
    for item in os.listdir(curr_path):
        full_text_path = os.path.join(curr_path, item)
        with open(full_text_path) as f:
            file_read = f.read()
            print("read file at", full_text_path)
            dic[item][folder] = file_read
print(dic)

df = pd.DataFrame(data=dic)
df = df.T
df.to_excel("dict1.xlsx")
