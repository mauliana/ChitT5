import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def clean_str(str):
    str = str.replace(" . ", ". ")
    str = str.replace(" , ", ", ")
    str = str.replace(" ? ", "? ")
    str = str.replace(" ' ", "'")
    str = str.replace(" ’ ", "'")
    return str

def split_save(data, path):
    # # split for training and evaluation
    train_df, rest_df = train_test_split(data, test_size=0.20)
    eval_df, test_df = train_test_split(rest_df, test_size=0.50)

    # cek output path
    if not os.path.exists(path):
        os.makedirs(path)

    # # Save data into file
    data.to_csv(path+"clean_all.tsv", "\t", index=False)
    train_df.to_csv(path+"train.tsv", "\t", index=False)
    eval_df.to_csv(path+"eval.tsv", "\t", index=False)
    test_df.to_csv(path+"test.tsv", "\t", index=False)

def preprocess_dailydialog(data_path, output_path):
    # Open file
    path = data_path
    f = open(path+'dailydialogues_text.txt', 'r')

    # read all data and store each line to array
    data = []
    for line in f:
        data.append(line)

    #  Format the data into input and target
    input_text = []
    target_text = []
    for line in data:
        split = line.split("__eou__")
        utterence_len = len(split)-1  # -1 because the last character is \n

        hist = ""
        for i in range(utterence_len-1):
            if i == 0:
                input = clean_str(split[0])
                target = clean_str(split[1])
                hist = input
            else:
                input = hist+clean_str(split[i])
                target = clean_str(split[i+1])
                hist = split[i-1]+split[i]
            
            input_text.append(input)
            target_text.append(target)

    # combine in one dataframe
    data_clean = pd.DataFrame({
        'input_text': input_text,
        'target_text': target_text,
    })

    split_save(data_clean, path=output_path)
    print('Preprocessing completed, check the result on: '+output_path)

def preprocess_small_talk(data_path, output_path):
    path = data_path
    f = open(path+'small-talk.txt', 'r')

    data = ''
    for line in f:
        data += line
        # print(line)
    dialog = data.split("###")

    input_text = []
    target_text = []
    for i in dialog:
        utt = i.split('\n')

        hist = ''    
        for j in range(len(utt)-2): # -2 because the last split is \n and empty
            input = " ".join([hist,utt[j]])
            target = utt[j+1]
            if j == 0:
                hist = utt[j]
            else:
                hist = utt[j-1]+utt[j]
            
            if input == ' ':
                continue

            # Check if the target end with question mark
            input_text.append(input)
            target_text.append(target)
            
    # combine in one dataframe
    data_clean = pd.DataFrame({
        'input_text': input_text,
        'target_text': target_text
    })
    
    split_save(data_clean, path=output_path)
    print('Preprocessing completed, check the result on: '+output_path)


def preprocess_coqa(data_path, output_path):
    path = data_path
    data=json.load((open(path+'coqa-train-v1.0.json')))

    input_text = []
    target_text = []
    for d in data['data']:
        que = []
        ans = []
        for q in d['questions']:
            que.append(q['input_text'])
        for a in d['answers']:
            ans.append(a['input_text'])

        length = len(que)
        dialog = ''
        for i in range(length):
            dialog += que[i]+'\n'+ans[i]+'\n'

        utt = dialog.split('\n')
        hist = ''    
        for j in range(len(utt)-2): # -2 because the last split is \n and empty
            if j == 0:
                input = utt[j]
                hist = input
            else:
                input = ' '.join([hist, utt[j]])
                hist = ' '.join([utt[j-1], utt[j]])
                
            target = utt[j+1]
            # hist = input
            
            if input == ' ' or target == ' ':
                continue

            # Check if the target end with question mark
            if target.endswith('?'):
                input_text.append(input)
                target_text.append(target)

    # combine in one dataframe
    data_clean = pd.DataFrame({
        'input_text': input_text,
        'target_text': target_text
    })

    split_save(data_clean, path=output_path)
    print('Preprocessing completed, check the result on: '+output_path)

def main():
    data_path = 'datasets/'
    preprocess_dailydialog(data_path, output_path=data_path+'dd/')
    preprocess_small_talk(data_path, output_path=data_path+'st/')
    preprocess_coqa(data_path, output_path=data_path+'coqa/')

if __name__ == "__main__":
    main()