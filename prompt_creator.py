import csv
import os
import argparse
import pandas as pd
from random import randint

def read_csv(csv_file_path):
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_content = [row for row in csv_reader]
    return csv_content

def write_to_txt(txt_file_path, text, newline_after=False):
    with open(txt_file_path, mode='w', encoding='utf-8') as txt_file:
        if newline_after:
            txt_file.write(text + '\n')
        else:
            txt_file.write(text)

def append_to_txt(txt_file_path, text, newline_after=False):
    with open(txt_file_path, mode='a', encoding='utf-8') as txt_file:
        if newline_after:
            txt_file.write(text + '\n')
        else:
            txt_file.write(text)

def create_dataframe(input_type, m, s, e, correctness):
    df = pd.DataFrame({'input_type': 10*[input_type]})
    if m == 0:
        df['movement'] = list(range(1, 11))
    else:
        df['movement'] = 10*[m]

    if s == 0:
        df['subject'] = list(range(1, 11))
    else:
        df['subject'] = 10*[s]

    if e == 0:
        df['episode'] = list(range(1, 11))
    else:
        df['episode'] = 10*[e]

    if correctness == 'correct':
        df['correctness'] = 10*[1]
    else:
        df['correctness'] = 10*[0]

    return df

def initialize_dataframe(input_type, m, s, e, k):

    # create separate dataframes for correct and incorrect examples
    df_cor = create_dataframe(input_type, m, s, e, 'correct')
    df_inc = create_dataframe(input_type, m, s, e, 'incorrect')

    # shuffle each dataframe to randomize demo samples
    df_cor = df_cor.sample(frac=1)
    df_inc = df_inc.sample(frac=1)

    # ensure demo sampled is balanced between labels
    k_cor = k // 2 # rounds down if k is odd
    k_inc = k - k_cor
    df_cor.insert(4, 'sample_type', k_cor*['demo'] + (10-k_cor)*['test'])
    df_inc.insert(4, 'sample_type', k_inc*['demo'] + (10-k_inc)*['test'])
    df = pd.concat([df_cor, df_inc], ignore_index=True)
    
    # shuffle sample order to mix cor/inc test samples
    df = df.sample(frac=1)
    df.sort_values('sample_type', ignore_index=True, inplace=True)

    # generate file names for ease of access
    cor_tag = ''
    df.insert(5, 'file', df.apply(lambda row: f'm{row["movement"]:02d}_s{row["subject"]:02d}_e{row["episode"]:02d}', axis=1))
    df['file'] = df['file'] + df['correctness'].apply(lambda x: cor_tag if x == 1 else '_inc')

    return df

def log_experiment(save_path_csv, df):
    df.to_csv(save_path_csv, index=False)

def one_sample_block(sample_num, csv_file_path, save_path_txt, correctness):
    '''
    Append one data sample block (from csv) to txt file
    Correctness: 'correct' or 'incorrect' or 'test'

    '''
    data_start = f"Data {sample_num+1}: '''"
    append_to_txt(save_path_txt, data_start, newline_after=True)
    data = read_csv(csv_file_path)
    for row in data:
        append_to_txt(save_path_txt, ','.join(row)+ '\n')

    data_end = f"'''\nLabel {sample_num+1}: {correctness}"
    if correctness == 'test':
        data_end = f"'''\nLabel {sample_num+1}: "
    append_to_txt(save_path_txt, data_end + '\n', newline_after=True)

def calculate_test_count(input_type, remaining):
    '''
    Splits remaining test samples into blocks of 3 or less
    to account for LLM input size limitations
    '''
    tests_per_file = []
    multiplier = 1

    if input_type == 'feat' or input_type == 'cot':
        multiplier = 3

    if remaining < 4*multiplier:
        return [remaining]
    while remaining > 3*multiplier:
        tests_per_file.append(3*multiplier)
        remaining -= 3*multiplier
    if remaining > 0:
        tests_per_file.append(remaining)

    return tests_per_file

def create_prompts_pos(df, total, k, save_dir_txt, dataname, device):
    # create LLM prompt 1 with k demos
    save_path_txt = save_dir_txt + f'/0_{k}demos1test.txt'
    out_format = "Desired output format: \"{Label}\""
    instructions = "Instructions: Identify the label (“correct” or “incorrect”) for the data sample below containing sequences of xyz positions of 22 body joints extracted from Kinect data of standing shoulder abduction exercise. Ensure the output adheres to the format provided." 
    write_to_txt(save_path_txt, out_format, newline_after=True)
    append_to_txt(save_path_txt, instructions, newline_after=True)

    sample_num = 0
    while sample_num < k: # for each demo
        file = df.loc[sample_num, 'file']
        correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
        print("DEMO FILE: ", file)
        csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/positions/{file}_positions.csv'
        one_sample_block(sample_num, csv_file_path, save_path_txt, correctness)
        sample_num += 1

    # add one test sample to the end of demos
    file = df.loc[sample_num, 'file']
    correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
    print("DEMO-TEST FILE: ", file)
    csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/positions/{file}_positions.csv'
    one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
    sample_num += 1
    print("file saved as: ", save_path_txt)

    # create LLM prompt files to cover remaining test examples
    tests_per_file = calculate_test_count('pos', total - sample_num)

    for file_num in range(len(tests_per_file)):
        print("------TEST FILE NUM: ", file_num)
        num_tests = tests_per_file[file_num]
        save_path_txt = save_dir_txt + f'/{file_num+1}_{num_tests}tests.txt'
        # out_format = "Desired output format: \"{Label}\""
        instructions = f"Instructions: Here are {num_tests} more samples (unlabelled), please maintain the desired output format, returning each label on a new line either \"correct\" or \"incorrect\"." 
        write_to_txt(save_path_txt, out_format, newline_after=True)
        append_to_txt(save_path_txt, instructions, newline_after=True)

        for i in range(num_tests):
            file = df.loc[sample_num, 'file']
            correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
            print("TEST FILE: ", file)
            csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/positions/{file}_positions.csv'
            one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
            sample_num += 1

    print("file saved as: ", save_path_txt)

def create_prompts_feat(df, total, k, save_dir_txt, dataname, device):
    # create LLM prompt 1 with k demos
    save_path_txt = save_dir_txt + f'/0_{k}demos1test.txt'
    out_format = "Desired output format: \"{Label}\""
    instructions = "Instructions: Identify the label (“correct” or “incorrect”) for the data sample below containing sequences of three features extracted from Kinect data of standing shoulder abduction exercise. Ensure the output adheres to the format provided."
    write_to_txt(save_path_txt, out_format, newline_after=True)
    append_to_txt(save_path_txt, instructions, newline_after=True)

    sample_num = 0
    while sample_num < k: # for each demo
        file = df.loc[sample_num, 'file']
        correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
        print("DEMO FILE: ", file)
        csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/features/{file}_features.csv'
        one_sample_block(sample_num, csv_file_path, save_path_txt, correctness)
        sample_num += 1

    # add one test sample to the end of demos
    file = df.loc[sample_num, 'file']
    correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
    print("DEMO-TEST FILE: ", file)
    csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/features/{file}_features.csv'
    one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
    sample_num += 1
    print("file saved as: ", save_path_txt)

    # create LLM prompt files to cover remaining test examples
    tests_per_file = calculate_test_count('feat', total - sample_num)

    for file_num in range(len(tests_per_file)):
        print("------TEST FILE NUM: ", file_num)
        num_tests = tests_per_file[file_num]
        save_path_txt = save_dir_txt + f'/{file_num+1}_{num_tests}tests.txt'
        # out_format = "Desired output format: \"{Label}\""
        instructions = f"Instructions: Here are {num_tests} more samples (unlabelled), please maintain the desired output format, returning each label on a new line either \"correct\" or \"incorrect\"." 
        write_to_txt(save_path_txt, out_format, newline_after=True)
        append_to_txt(save_path_txt, instructions, newline_after=True)

        for i in range(num_tests):
            file = df.loc[sample_num, 'file']
            correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
            print("TEST FILE: ", file)
            csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/features/{file}_features.csv'
            one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
            sample_num += 1

    print("file saved as: ", save_path_txt)


def create_prompts_cot(df, total, k, save_dir_txt, dataname, device):
    # create LLM prompt 1 with k demos
    save_path_txt = save_dir_txt + f'/0_{k}demos1test.txt'
    out_format = "Desired output format: \"{Label}: {Adjective} {noun} {body part and feature}\":  "
    instructions = "Instructions: Identify the label and provide the top 1 to 3 rationale for the corresponding label “correct” or “incorrect” for the data sample below containing sequences of three features extracted from Kinect data of standing shoulder abduction exercise. Ensure the output adheres to the format provided."
    write_to_txt(save_path_txt, out_format, newline_after=True)
    append_to_txt(save_path_txt, instructions, newline_after=True)

    sample_num = 0
    while sample_num < k: # for each demo
        file = df.loc[sample_num, 'file']
        correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
        print("DEMO FILE: ", file)
        csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/features/{file}_features.csv'
        one_sample_block(sample_num, csv_file_path, save_path_txt, correctness)
        sample_num += 1

    # add one test sample to the end of demos
    file = df.loc[sample_num, 'file']
    correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
    print("DEMO-TEST FILE: ", file)
    csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/features/{file}_features.csv'
    one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
    sample_num += 1
    print("file saved as: ", save_path_txt)

    # create LLM prompt files to cover remaining test examples
    tests_per_file = calculate_test_count('feat', total - sample_num)

    for file_num in range(len(tests_per_file)):
        print("------TEST FILE NUM: ", file_num)
        num_tests = tests_per_file[file_num]
        save_path_txt = save_dir_txt + f'/{file_num+1}_{num_tests}tests.txt'
        # out_format = "Desired output format: \"{Label}\""
        instructions = f"Instructions: Here are {num_tests} more unlabelled samples, please maintain the desired output format, returning each label on a new line either \"correct\" or \"incorrect\"." 
        write_to_txt(save_path_txt, out_format, newline_after=True)
        append_to_txt(save_path_txt, instructions, newline_after=True)

        for i in range(num_tests):
            file = df.loc[sample_num, 'file']
            correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
            print("TEST FILE: ", file)
            csv_file_path = f'dataset/{dataname}_generated/{correctness}/{device}/features/{file}_features.csv'
            one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
            sample_num += 1

    print("file saved as: ", save_path_txt)

'''
TERMINAL RUN COMMAND:
python prompt_creator.py --dataname UI-PRMD --device kinect --total 20 --k 0 --input_type pos --m 7 --s 0 --e 1
'''

if __name__ == '__main__':
    # define default parameters 
    parser = argparse.ArgumentParser("Create .txt prompts for LLM experiments")
    parser.add_argument('--dataname', type=str, default='UI-PRMD', help='name of dataset')
    parser.add_argument('--device', type=str, default='kinect', help='device used to capture data')
    parser.add_argument('--total', type=int, default=20, help='total number of samples in entire experiment')
    parser.add_argument('--k', type=int, default=0, help='k value for number of demos')
    parser.add_argument('--input_type', type=str, default='feat', help='type of input data: pos, feat, or cot')
    parser.add_argument('--m', type=int, default=7, help='if 0, generate random value. if 1-10, use const value')
    parser.add_argument('--s', type=int, default=1, help='if 0, generate random value. if 1-10, use const value')
    parser.add_argument('--e', type=int, default=0, help='if 0, generate random value. if 1-10, use const value')

    args = parser.parse_args()
    dataname = args.dataname
    device = args.device # don't need now, but to keep it generalizable for future
    total = args.total
    k = args.k
    input_type = args.input_type
    m = args.m
    s = args.s
    e = args.e

    num_tests = total - k # number of examples to include in each test prompt
    exp_name = f"{k}shot-{input_type}_m{m:02d}_s{s:02d}_e{e:02d}"

    demo_count, test_count = 0, 0

    save_dir_txt = f'dataset/{dataname}_prompts/{exp_name}'
    if not os.path.exists(save_dir_txt):
        os.makedirs(save_dir_txt)

    # initialize dataframe to store experiment file information
    df = initialize_dataframe(input_type, m, s, e, k)    
    print(df)

    if input_type == 'pos':
        create_prompts_pos(df, total, k, save_dir_txt, dataname, device)
    elif input_type == 'feat':
        create_prompts_feat(df, total, k, save_dir_txt, dataname, device)
    elif input_type == 'cot':
        create_prompts_cot(df, total, k, save_dir_txt, dataname, device)
    
    log_experiment(save_dir_txt + f'/log_{exp_name}.csv', df)