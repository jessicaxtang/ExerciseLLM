'''
Creates .txt files for LLM experiment prompts, to aggregate data samples and labels for faster web testing.
'''

import csv
import os
import pandas as pd
from tqdm import tqdm

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
    '''
    Create a dataframe with 10 samples of the same movement, subject, episode, and correctness
    If m, s, or e is 0, generate random values for each sample
    '''
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

def initialize_dataframe_20(input_type, m, s, e, k):
    '''
    Create a dataframe with 20 samples for the experiment
    '''

    # create separate dataframes for correct and incorrect examples
    df_cor = create_dataframe(input_type, m, s, e, 'correct')
    df_inc = create_dataframe(input_type, m, s, e, 'incorrect')

    # shuffle each dataframe to randomize demo samples
    df_cor = df_cor.sample(frac=1)
    df_inc = df_inc.sample(frac=1)

    # ensure demo sampled is balanced between labels
    df_cor.insert(4, 'sample_type', k*['demo'] + (10-k)*['test'])
    df_inc.insert(4, 'sample_type', k*['demo'] + (10-k)*['test'])
    df = pd.concat([df_cor, df_inc], ignore_index=True)
    
    # shuffle sample order to mix cor/inc test samples
    df = df.sample(frac=1)
    df.sort_values('sample_type', ignore_index=True, inplace=True)

    # generate file names for ease of access
    df.insert(5, 'file', df.apply(lambda row: f'm{row["movement"]:02d}_s{row["subject"]:02d}_e{row["episode"]:02d}', axis=1))
    df['file'] = df['file'] + df['input_type'].apply(lambda x: '_absolutes' if x == 'abs' else '_features')
    df['file'] = df['file'] + df['correctness'].apply(lambda x: '' if x == 1 else '_inc')

    return df

def custom_dataframe(csv_file_path):
    '''
    If using a custom csv file, read the file and create a dataframe
    '''
    df = pd.read_csv(csv_file_path)

    # shuffle sample order to mix cor/inc test samples
    df = df.sample(frac=1)
    df.sort_values('sample_type', ignore_index=True, inplace=True)

    # generate file names for ease of access
    cor_tag = ''
    df.insert(5, 'file', df.apply(lambda row: f'm{row["movement"]:02d}_s{row["subject"]:02d}_e{row["episode"]:02d}', axis=1))
    df['file'] = df['file'] + df['correctness'].apply(lambda x: cor_tag if x == 1 else '_inc')
    
    return df

def log_experiment(save_path_csv, df):
    save_path_csv = os.path.join(save_path_csv, 'ground_truth.csv')
    df.to_csv(save_path_csv, index=False)

def one_sample_block(sample_num, csv_file_path, save_path_txt, correctness):
    '''
    Append one data sample block (from csv) to txt file
    Correctness: 'correct' or 'incorrect' or 'test'

    '''
    data_start = f"<data {sample_num+1}>"
    append_to_txt(save_path_txt, data_start, newline_after=True)
    
    data = read_csv(csv_file_path)
    for i, row in enumerate(data):
        if i == 0:  # First row contains headers
            append_to_txt(save_path_txt, ','.join(row) + '\n')
        else:
            rounded_row = [f"{float(value):.3f}" for value in row]  # round to 3 decimal places
            append_to_txt(save_path_txt, ','.join(rounded_row) + '\n')

    data_end = f"<\\data>\nLabel {sample_num+1}: {correctness}"
    if correctness == 'test':
        data_end = f"<\\data>\nLabel {sample_num+1}: "
    append_to_txt(save_path_txt, data_end + '\n', newline_after=True)

def get_data_splits(input_type, k):
    """
    Get the number of demos and test samples based on input type and k (k = 0-4).
    """
    if input_type == 'abs':
        base = [5, 5, 5, 4] # k = 0, 19 test samples
        if k < 2:
            base[-1] -= 2 * k # k = 1: [5, 5, 5, 2], 17 test samples
        elif k < 5:
            base = [5, 5, 5] # k = 2, 15 test samples
            base[-1] -= 2 * (k-2)   # k = 3: [5, 5, 3], 13 test samples
                                    # k = 4: [5, 5, 1], 11 test samples
        return base
    # elif input_type in ['feat', 'cot']:
    else:
        base = [9, 10]
        if k < 4:
            base[-1] -= 2 * k
        return base
    # add case for GradCAM

def create_prompts_abs(df, total, k, save_dir_txt, dataname, device, move, movement_map, input_type='absolutes'):
    '''
    Create .txt files for LLM experiment prompts with k demos and remaining test samples
    '''
    os.makedirs(save_dir_txt, exist_ok=True)
    save_path_txt = os.path.join(save_dir_txt, f'0_{2*k}demos1test.txt')
    out_format = "Desired output format: \"Label\""
    instructions = f"Instructions: Identify the label (“correct” or “incorrect”) for the {2*k+1}th data sample below containing sequences of xyz absolute positions of 22 body joints extracted from {device} data of {movement_map[move]} exercise. \
    the joint mapping is: 0: Waist, 1: Spine, 2: Chest, 3: Neck, 4: Head, 5: Head tip, 6: Left collar, 7: Left upper arm, 8: Left forearm, 9: Left hand, 10: Right collar, 11: Right upper arm, 12: Right forearm, 13: Right hand, 14: Left upper leg, 15: Left lower leg, 16: Left foot, 17: Left leg toes, 18: Right upper leg, 19: Right lower leg, 20: Right foot, 21: Right leg toes. \
    Ensure the output adheres to the format provided." 

    write_to_txt(save_path_txt, out_format, newline_after=True)
    append_to_txt(save_path_txt, instructions, newline_after=True)

    sample_num = 0
    while sample_num < 2*k: # for each demo
        file = df.loc[sample_num, 'file']
        correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
        csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
        one_sample_block(sample_num, csv_file_path, save_path_txt, correctness)
        sample_num += 1

    # add one test sample to the end of demos
    file = df.loc[sample_num, 'file']
    correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
    csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
    one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
    sample_num += 1

    # create LLM prompt files to cover remaining test examples
    tests_per_file = get_data_splits('abs', k)

    for file_num in range(len(tests_per_file)):
        num_tests = tests_per_file[file_num]
        save_path_txt = os.path.join(save_dir_txt, f'{file_num+1}_{num_tests}tests.txt')
        instructions = f"Instructions: Here are {num_tests} more samples (unlabelled), please maintain the desired output format, returning each label on a new line either \"correct\" or \"incorrect\"." 
        write_to_txt(save_path_txt, out_format, newline_after=True)
        append_to_txt(save_path_txt, instructions, newline_after=True)

        for i in range(num_tests):
            file = df.loc[sample_num, 'file']
            correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
            csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
            one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
            sample_num += 1

def create_prompts_feat(df, total, k, save_dir_txt, dataname, device, move, movement_map, input_type='features'):
    # create LLM prompt 1 with k demos
    os.makedirs(save_dir_txt, exist_ok=True)
    save_path_txt = os.path.join(save_dir_txt, f'0_{2*k}demos1test.txt')
    out_format = "Desired output format: \"Label\""
    instructions = f"Instructions: Identify the label (“correct” or “incorrect”) for the {2*k+1}th data sample below containing sequences of three features extracted from {device} data of {movement_map[move]} exercise. Ensure the output adheres to the format provided."
    write_to_txt(save_path_txt, out_format, newline_after=True)
    append_to_txt(save_path_txt, instructions, newline_after=True)

    sample_num = 0
    while sample_num < 2*k: # for each demo
        file = df.loc[sample_num, 'file']
        correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
        csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
        one_sample_block(sample_num, csv_file_path, save_path_txt, correctness)
        sample_num += 1

    # add one test sample to the end of demos
    file = df.loc[sample_num, 'file']
    correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
    csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
    one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
    sample_num += 1

    # create LLM prompt files to cover remaining test examples
    tests_per_file = get_data_splits('feat', k)

    for file_num in range(len(tests_per_file)):
        num_tests = tests_per_file[file_num]
        save_path_txt = os.path.join(save_dir_txt, f'{file_num+1}_{num_tests}tests.txt')
        instructions = f"Instructions: Here are {num_tests} more samples (unlabelled), please maintain the desired output format, returning each label on a new line either \"correct\" or \"incorrect\"." 
        write_to_txt(save_path_txt, out_format, newline_after=True)
        append_to_txt(save_path_txt, instructions, newline_after=True)

        for i in range(num_tests):
            file = df.loc[sample_num, 'file']
            correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
            csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
            one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
            sample_num += 1


def create_prompts_cot(df, total, k, save_dir_txt, dataname, device, move, movement_map, input_type='features'):
    # create LLM prompt 1 with k demos
    os.makedirs(save_dir_txt, exist_ok=True)
    save_path_txt = os.path.join(save_dir_txt, f'0_{2*k}demos1test.txt')
    # out_format = "Desired output format: \"Label: \"Adjective\" \"noun\" \"body part and feature\":  "
    out_format = "Desired output format: \"Label\""
    instructions = f"Instructions: Identify the label and provide the top 1 to 3 rationale for the corresponding label “correct” or “incorrect” for the {2*k+1}th data sample below containing sequences of three features extracted from {device} data of {movement_map[move]} exercise. Ensure the output adheres to the format provided. Explain your reasoning."
    write_to_txt(save_path_txt, out_format, newline_after=True)
    append_to_txt(save_path_txt, instructions, newline_after=True)

    sample_num = 0
    while sample_num < 2*k: # for each demo
        file = df.loc[sample_num, 'file']
        correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
        csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
        one_sample_block(sample_num, csv_file_path, save_path_txt, correctness)
        sample_num += 1

    # add one test sample to the end of demos
    file = df.loc[sample_num, 'file']
    correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
    csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
    one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
    sample_num += 1

    # create LLM prompt files to cover remaining test examples
    tests_per_file = get_data_splits('cot', k)

    for file_num in range(len(tests_per_file)):
        num_tests = tests_per_file[file_num]
        save_path_txt = os.path.join(save_dir_txt, f'{file_num+1}_{num_tests}tests.txt')
        instructions = f"Instructions: Here are {num_tests} more unlabelled samples, please maintain the desired output format, returning each label on a new line either \"correct\" or \"incorrect\"." 
        write_to_txt(save_path_txt, out_format, newline_after=True)
        append_to_txt(save_path_txt, instructions, newline_after=True)

        for i in range(num_tests):
            file = df.loc[sample_num, 'file']
            correctness = 'correct' if df.loc[sample_num, 'correctness'] == 1 else 'incorrect'
            csv_file_path = f'dataset/{dataname}/{correctness}/{device}/{input_type}/{file}.csv'
            one_sample_block(sample_num, csv_file_path, save_path_txt, 'test')
            sample_num += 1

if __name__ == '__main__':

    movement_map = {
        'm01': 'Deep squat',
        'm02': 'Hurdle step',
        'm03': 'Inline lunge',
        'm04': 'Side lunge',
        'm05': 'Sit to stand',
        'm06': 'Standing active straight leg raise',
        'm07': 'Standing shoulder abduction',
        'm08': 'Standing shoulder extension',
        'm09': 'Standing shoulder internal–external rotation',
        'm10': 'Standing shoulder scaption',
    }

    # INITIALIZE VARIABLES
    # fixed:
    dataname = 'UI-PRMD'
    device = 'kinect'
    total = 20
    # varies:
    k = 0
    input_type = 'feat'
    m, s, e = 0, 0, 0

    # Count total iterations for progress bar
    total_iterations = 3 * 4 * 10 * 10  # input types * k range * movements * subjects

    with tqdm(total=total_iterations, desc="Processing all experiments") as pbar:
        # loop through all experiment settings
        for input_type in ['abs', 'feat', 'cot']:
            print(f"Creating prompts for {input_type} input type...")
            for k in range(0, 4):  # number of demos per class
                print(f"{k} demos...")
                for m in range(1, 11):  # movements
                    for s in range(1, 11):  # subjects
                        # note: episode is the randomized variable, so no need to loop through
                        exp_name = f"{k}shot-{input_type}_m{m:02d}_s{s:02d}_e{e:02d}"

                        save_dir_txt = os.path.join('dataset', dataname + '_prompts', input_type, str(k) + 'shot', exp_name)

                        if not os.path.exists(save_dir_txt):
                            os.makedirs(save_dir_txt)

                        # Initialize dataframe to store experiment file information
                        df = initialize_dataframe_20(input_type, m, s, e, k)
                        move = f"m{m:02d}"

                        # print(df)

                        if input_type == 'abs':
                            create_prompts_abs(df, total, k, save_dir_txt, dataname, device, move, movement_map, 'absolutes')
                        elif input_type == 'feat':
                            create_prompts_feat(df, total, k, save_dir_txt, dataname, device, move, movement_map, 'features')
                        elif input_type == 'cot':
                            create_prompts_cot(df, total, k, save_dir_txt, dataname, device, move, movement_map, 'features')
                        # Add input_type == 'gradCAM' case #TODO

                        log_experiment(save_dir_txt, df)
                        pbar.update(1)  # Increment the progress bar
    
    print("All experiment prompts saved.")

    ######### CODE FOR UNIT TESTING: DELETE WHEN DONE
    # input_type = 'abs'
    # m = 1
    # s = 1
    # # note: episode is the randomized variable, so no need to loop through
    # exp_name = f"{k}shot-{input_type}_m{m:02d}_s{s:02d}_e{e:02d}"

    # demo_count, test_count = 0, 0

    # save_dir_txt = os.path.join('dataset', dataname + '_prompts', input_type, exp_name)

    # if not os.path.exists(save_dir_txt):
    #     os.makedirs(save_dir_txt)

    # # initialize dataframe to store experiment file information
    # df = initialize_dataframe_20(input_type, m, s, e, k)  
    # move = f"m{m:02d}"  

    # print(df)

    # if input_type == 'abs':
    #     print("save_dir_txt", save_dir_txt)
    #     create_prompts_abs(df, total, k, save_dir_txt, dataname, device, move, movement_map, 'absolutes')
    
    # log_experiment(save_dir_txt + f'/log_{exp_name}.csv', df)
    # print(f"Experiment {exp_name} prompts saved to {save_dir_txt}")
    ######### CODE FOR UNIT TESTING: DELETE WHEN DONE