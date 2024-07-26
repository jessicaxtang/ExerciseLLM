import csv
import os
from random import randint

def read_csv(csv_file_path):
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_content = [row for row in csv_reader]
    return csv_content

def write_to_txt(txt_file_path, text, newline_after=False):
    with open(txt_file_path, mode='a', encoding='utf-8') as txt_file:
        if newline_after:
            txt_file.write(text + '\n')
        else:
            txt_file.write(text)
            
def generate_train_files(num_files, m, s, e):
    '''
    if m, s, or e is None, generate random values for them
    returns a list of strings of file names in the format 'mXX_sXX_eXX'
    note: specify the correctness of the files in the create_training_prompt() function
    '''
    train_files = []
    for i in range(num_files):
        if m == None:
            m = randint(1, 11)
        if s == None:
            s = randint(1, 11)
        if e == None:
            e = randint(1, 11)
        train_files.append(f'm{m:02d}_s{s:02d}_e{e:02d}')
    return train_files

def create_training_prompt(train_files, save_dir_txt, dataname, input_type, device, correctness, exp_num, train_num):
    '''
    train_files: list of strings of file names in the format 'mXX_sXX_eXX'
    exp_num: experiment number
    train_num: training file number
    '''
    cor_tag = ''
    if correctness == 'incorrect':
        cor_tag = '_inc'

    save_path_txt = save_dir_txt + f'/exp{exp_num}_train{train_num}{cor_tag}.txt'
    ex_num = 1

    for file in train_files:
        csv_file_path = f'dataset/{dataname}_features/{correctness}/{device}/{input_type}/{file}{cor_tag}_{input_type}.csv'
        data = read_csv(csv_file_path)
        header = f'# {correctness} example {ex_num*train_num}'
        write_to_txt(save_path_txt, header, newline_after=True)
        for row in data:
            write_to_txt(save_path_txt, ','.join(row)+ '\n')
        ex_num += 1

if __name__ == '__main__':
    # define default parameters
    dataname = 'UI-PRMD'
    input_type = 'features'
    downsample_rate = 3
    device = 'kinect'
    correctness = 'correct'
    cor_tag = ''

    # None for random value
    m = 7
    s = 1
    e = None
    num_files = 10 # number of examples to include in each training prompt
    num_train_files = 2 # number of training prompts to create per correctness
    exp_num = 1

    count = 0

    save_dir_txt = f'dataset/{dataname}_prompts/exp{exp_num}/{correctness}/{device}/{input_type}'
    if not os.path.exists(save_dir_txt):
        os.makedirs(save_dir_txt)

    for correctness in ['correct', 'incorrect']:
        for train_num in range(1, 1+num_train_files):
            train_files = generate_train_files(num_files, m, s, e)
            print("Train files: ", train_files)
            create_training_prompt(train_files, save_dir_txt, dataname, input_type, device, correctness, exp_num, train_num)
            count += 1

    print("Total number of files saved: ", count)
