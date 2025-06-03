#!/mnt/fast/nobackup/users/ta01123/eat_apr30/fairseqvenv/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import torchaudio

meta_data_dir = "/mnt/fast/nobackup/scratch4weeks/ta01123/SPASS/SPASS_META_DATA/"
market_csv = meta_data_dir + "1_MatrixLabelsMarket.csv"
park_csv = meta_data_dir + "2_MatrixLabelsPark.csv"
square_csv = meta_data_dir + "3_MatrixLabelsSquare.csv"
street_csv = meta_data_dir + "4_MatrixLabelsStreet.csv"
waterfront_csv = meta_data_dir + "5_MatrixLabelsWaterfront.csv"


main_path = "/mnt/fast/nobackup/scratch4weeks/ta01123/SPASS/"

train_percent = 0.75

# all_labels_set = set()

train_df = pd.DataFrame()
eval_df = pd.DataFrame()
# for csv_path,sub_path in [(market_csv,'Market audio files'), (park_csv,'Park audio files'), (square_csv,'Square audio files'),]:# (street_csv,'Street audio files'), (waterfront_csv,'Waterfront audio files')]:
# for csv_path,sub_path in  [(waterfront_csv,'Waterfront audio files')]:
for csv_path,sub_path in   [(street_csv,'Street audio files')]:

    sub_path = sub_path.replace(' ','_')

    manifest_dir_name = 'manifest_SPASS_'+(sub_path.split('_')[0]).upper() #### added later 

    # if os.path.exists(manifest_dir_name):
    #     shutil.rmtree(manifest_dir_name)
    # os.mkdir(manifest_dir_name)

    print(f"manifest_dir_name: {manifest_dir_name}")

    df = pd.read_csv(csv_path)
    df = df[['audio_filename','class']]
    df['audio_filename'] = sub_path + '/' + df['audio_filename']
    print(df.head())
    print("# of wav files: ", df['audio_filename'].nunique())
    print("# of classes: ", df['class'].nunique())



    #### 
    result = df.groupby('audio_filename')['class'].apply(lambda x: x.unique().tolist())

    wav_files_l = result.index.tolist()  # List of unique wav files
    labels_l = result.tolist()           # List of corresponding unique classes for each wav file
    num_classes_l = result.apply(len).tolist()  # List of the number of unique classes for each wav file

    temp_labels = set([item for sublist in labels_l for item in sublist])
    # if 'Vwater' in temp_labels:
    #     print(f"\nVwater in {sub_path}")
    all_labels_set = temp_labels



    train_split = int(len(wav_files_l)*train_percent)
    train_wav_files = wav_files_l[:train_split]
    eval_wav_files = wav_files_l[train_split:]

    ## no overlap between train and eval labels
    assert len(set(train_wav_files).intersection(set(eval_wav_files))) == 0

    train_labels = labels_l[:train_split]
    temp_train_labels = set([item for sublist in train_labels for item in sublist])
    eval_labels = labels_l[train_split:]
    temp_eval_labels = set([item for sublist in eval_labels for item in sublist])
    assert temp_eval_labels - temp_train_labels == set(), f"eval labels: {temp_eval_labels}, train labels: {temp_train_labels}"

    train_num_classes = num_classes_l[:train_split]
    eval_num_classes = num_classes_l[train_split:]

    assert len(train_wav_files) == len(train_labels) == len(train_num_classes)
    assert len(eval_wav_files) == len(eval_labels) == len(eval_num_classes)

    train_wav_labels = list(zip(train_wav_files, train_labels))
    eval_wav_labels = list(zip(eval_wav_files, eval_labels))

    # print("Train wav labels: ", train_wav_labels[:5], "Count: ", len(train_wav_labels))
    # print("Eval wav labels: ", eval_wav_labels[:5], "Count: ", len(eval_wav_labels))
    # print("All labels: ", all_labels_set, "Count: ", len(all_labels_set))

    ### label_descriptors.csv
    temp = pd.DataFrame()
    temp[0]=list(range(len(all_labels_set)))
    all_labels_l = list(all_labels_set)
    temp[1]=all_labels_l
    temp[2]=all_labels_l
    temp.to_csv(manifest_dir_name+'/'+'label_descriptors.csv', index=False, header=False)
    ########

    ## manifest format

    for split,wav_label_l in [('train',train_wav_labels), ('eval',eval_wav_labels)]:
        to_print = []
        for wav_file, labels in wav_label_l:
            sr = -1
            wav,sr = torchaudio.load(main_path + wav_file)
            assert wav.shape[0] == 1, f"wav shape: {wav.shape}"
            # print(f"duration: {wav.shape[1]/sr}")
            duration = wav.shape[1]/sr
            # assert abs(duration - 10) < 0.5, ### all passed except STREET
            if abs(duration - 10) > 0.5:
                print(f"duration: {duration}, wav_file: {wav_file}, sr: {sr}")
            
            tsv_to_print = (wav_file, sr)

            lbl_to_print = (wav_file.split(".wav")[0], ",".join(labels))

            to_print.append((tsv_to_print, lbl_to_print))
        
        results = to_print
        



        # You can write these results to your file as before
        with open(f'{manifest_dir_name}/{split}.tsv', 'w') as f1, open(f'{manifest_dir_name}/{split}.lbl', 'w') as f2:
            f1.write(main_path + '\n')
            for line in results:
                line1, line2 = line[0], line[1]
                f1.write('\t'.join([str(x) for x in line1]) + '\n')
                f2.write('\t'.join([str(x) for x in line2]) + '\n')

