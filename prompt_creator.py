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

def append_to_txt(txt_file_path, text, newline_after=False):
    with open(txt_file_path, mode='a', encoding='utf-8') as txt_file:
        if newline_after:
            txt_file.write(text + '\n')
        else:
            txt_file.write(text)
            
def generate_filenames(num_demos, m, s, e):
    '''
    if m, s, or e is None, generate random values for them
    returns a list of strings of file names in the format 'mXX_sXX_eXX'
    note: specify the correctness of the files in the create_prompt() function
    '''
    filenames = []
    # TO DO: fix that RANDINT WORKS
    for i in range(num_demos):
        if m == None:
            m = randint(1, 10)
        if s == None:
            s = randint(1, 10)
        if e == None:
            e = randint(1, 10)
        filenames.append(f'm{m:02d}_s{s:02d}_e{e:02d}')
    return filenames

def create_prompt(filenames, save_dir_txt, dataname, input_type, device, correctness, exp_num, file_type, file_num, num_files):
    '''
    filenames: list of strings of file names in the format 'mXX_sXX_eXX'
    exp_num: experiment number
    file_type: 'demo' or 'test'
    file_num: file number of file_type
    num_files: total number of files to generate of file_type
    '''
    cor_tag = ''
    if correctness == 'incorrect':
        cor_tag = '_inc'

    save_path_txt = save_dir_txt + f'/exp{exp_num}_{file_type}{file_num}{cor_tag}.txt'
    log_path_txt = save_dir_txt + f'/exp{exp_num}_{file_type}{file_num}{cor_tag}_log.txt'
    ex_num = 1 + ((file_num-1)*num_files)

    
    for file in filenames:
        # print("CURRENT FILE: ", file)
        csv_file_path = f'dataset/{dataname}_features/{correctness}/{device}/{input_type}/{file}{cor_tag}_{input_type}.csv'
        if file_type == 'demo':
            header = f'# {correctness} example {ex_num}'
            append_to_txt(log_path_txt, correctness + '\t', newline_after=False)
        else:
            header = f'# unlabelled example {ex_num}'
            correctness_rand = 'correct' if randint(0, 1) == 1 else 'incorrect' # generate random correctness per test example
            print("correctness_rand: ", correctness_rand)
            if correctness_rand == 'correct':
                cor_tag = ''
            else:
                cor_tag = '_inc'
            csv_file_path = f'dataset/{dataname}_features/{correctness_rand}/{device}/{input_type}/{file}{cor_tag}_{input_type}.csv'
            append_to_txt(log_path_txt, correctness_rand + '\t', newline_after=False)

        data = read_csv(csv_file_path)
        append_to_txt(save_path_txt, header, newline_after=True) # write header to prompt file
        for row in data:
            append_to_txt(save_path_txt, ','.join(row)+ '\n') # write data to prompt file
        ex_num += 1
        append_to_txt(log_path_txt, file, newline_after=True) # write file name to log file

def initialize_dataframe(input_type, m, s, e, k):
    df = pd.DataFrame({'input_type': 20*[input_type]})
    if m == 0:
        df['movement'] = list(range(1, 11)) + list(range(1, 11))
    else:
        df['movement'] = 20*[m]

    if s == 0:
        df['subject'] = list(range(1, 11)) + list(range(1, 11))
    else:
        df['subject'] = 20*[s]

    if e == 0:
        df['episode'] = list(range(1, 11)) + list(range(1, 11))
    else:
        df['episode'] = 20*[e]

    df['correctness'] = 10*[1] + 10*[0]

    df = df.sample(frac=1) # shuffle order of rows
    # ensure half of the first k rows contain correctness = 1, random sampling won't work

    df.insert(0, 'sample', list(range(1, 21))) # add sample number column

    df['sample_type'] = k*['demo'] + (20-k)*['test']

    return df

'''
TERMINAL RUN COMMAND:
python prompt_creator.py --dataname UI-PRMD --device kinect --exp_num 1 --k 0 --input_type pos --m 7 --s 0 --e 1

'''
if __name__ == '__main__':
    # define default parameters 
    parser = argparse.ArgumentParser("Create .txt prompts for LLM experiments")
    parser.add_argument('--dataname', type=str, default='UI-PRMD', help='name of dataset')
    parser.add_argument('--device', type=str, default='kinect', help='device used to capture data')
    parser.add_argument('--exp_num', type=int, default=1, help='experiment number')
    parser.add_argument('--k', type=int, default=0, help='k value for number of demos')
    parser.add_argument('--input_type', type=str, default='feat', help='type of input data: pos, feat, or cot')
    parser.add_argument('--m', type=int, default=7, help='if 0, generate random value. if 1-10, use const value')
    parser.add_argument('--s', type=int, default=1, help='if 0, generate random value. if 1-10, use const value')
    parser.add_argument('--e', type=int, default=None, help='if 0, generate random value. if 1-10, use const value')

    args = parser.parse_args()
    dataname = args.dataname
    device = args.device # don't need now, but to keep it generalizable for future
    exp_num = args.exp_num
    k = args.k
    input_type = args.input_type
    m = args.m
    s = args.s
    e = args.e

    num_tests = 20 - k # number of examples to include in each test prompt
    exp_name = str(k) + "-shot-" + input_type
    if m != 0: 
        exp_name += f"_m{m:02d}"
    if s != 0:
        exp_name += f"_s{s:02d}"
    if e != 0:
        exp_name += f"_e{e:02d}"

    demo_count, test_count = 0, 0

    save_dir_txt = f'dataset/{dataname}_prompts/{exp_num}_{exp_name}'
    if not os.path.exists(save_dir_txt):
        os.makedirs(save_dir_txt)

    # initialize dataframe to store experiment file information
    df = initialize_dataframe(input_type, m, s, e, k)    
    print(df)

"""
    # generate demo prompts (labelled)
    for correctness in ['correct', 'incorrect']:
        for demo_num in range(1, 1+num_demo_files):
            demo_files = generate_filenames(num_demos, m, s, e)
            print("demo files: ", demo_files)
            create_prompt(demo_files, save_dir_txt, dataname, input_type, device, correctness, exp_num, 'demo', demo_num, num_demos)
            demo_count += 1
    
    for test_num in range(1, 1+num_test_files):
            test_files = generate_filenames(num_tests, m, s, e)
            # TO DO: make sure test files are different from demo files

            print("test files: ", test_files)
            create_prompt(test_files, save_dir_txt, dataname, input_type, device, correctness, exp_num, 'test', test_num, num_tests)
            test_count += 1

    print("num demo files saved:\t", demo_count)
    print("num test files saved:\t", test_count)
    print("TOTAL num files saved:\t", demo_count + test_count)
"""