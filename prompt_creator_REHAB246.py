'''
Creates .txt files for LLM experiment prompts, adapted for the REHAB24-6 dataset.
'''

import os
import pandas as pd
from tqdm import tqdm

def write_to_txt(txt_file_path, text, mode, newline_after=False):
    with open(txt_file_path, mode=mode, encoding='utf-8') as txt_file:
        if newline_after:
            txt_file.write(text + '\n')
        else:
            txt_file.write(text)

def initialize_dataframe(df, exercise_id, person_id, k):
    """
    Initialize a DataFrame with samples: half correct, half incorrect
    """

    correct_samples = df[(df['exercise_id'] == exercise_id) & 
                         (df['person_id'] == person_id) & 
                         (df['correctness'] == 1)].sample(frac=1, replace=False)
    
    incorrect_samples = df[(df['exercise_id'] == exercise_id) & 
                           (df['person_id'] == person_id) & 
                           (df['correctness'] == 0)].sample(frac=1, replace=False)
    
    if len(correct_samples) < k or len(incorrect_samples) < k:
        return pd.DataFrame({'A' : []}) # return empty dataframe

    correct_samples['sample_type'] = ['demo'] * k + ['test'] * (len(correct_samples) - k)
    incorrect_samples['sample_type'] = ['demo'] * k + ['test'] * (len(incorrect_samples) - k)

    combined = pd.concat([correct_samples, incorrect_samples]).sample(frac=1).reset_index(drop=True)
    combined.sort_values('sample_type', ignore_index=True, inplace=True)
    return combined

def log_experiment(save_path_csv, df):
    save_path_csv = os.path.join(save_path_csv, 'ground_truth.csv')
    df.to_csv(save_path_csv, index=False)

def read_csv(csv_path):
    """Read the contents of a CSV file and format them as a string."""
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines)
    except FileNotFoundError:
        return f"Error: File {csv_path} not found."

def create_prompt_files(df, save_dir, exercise_num, exercise_name, k, confidence=False):
    """Generate .txt prompt files with actual data contents."""
    os.makedirs(save_dir, exist_ok=True)
    demo_file = os.path.join(save_dir, f"0_{2 * k}demos1test.txt")
    output_format = "Label, Confidence" if confidence else "Label"

    # General instructions
    instructions = (
        f"Instructions: Identify the label ('correct' or 'incorrect'){' with confidence (0 to 1)' if confidence else ''} "
        f"for the {2 * k + 1}th sample for the {exercise_name} exercise."
    )
    write_to_txt(demo_file, f"Desired output format: {output_format}", mode="w", newline_after=True)
    write_to_txt(demo_file, instructions, mode="a", newline_after=True)

    for i, row in df.iterrows():
        if i <= 2 * k: # Write demos and the 1st test sample
            label = "" if i == 2 * k else ("correct" if row["correctness"] == 1 else "incorrect")
            csv_path = f'dataset/REHAB24-6//2d_joints_segmented/features/Ex{exercise_num}-segmented/{row["file_name"]}.csv'
            file_contents = read_csv(csv_path)

            write_to_txt(demo_file, f"<data {i + 1}>", mode="a", newline_after=True)
            write_to_txt(demo_file, file_contents, mode="a")  # Append file contents
            write_to_txt(demo_file, f"<\\data {i + 1}>\nLabel: {label}\n", mode="a", newline_after=True)

    # Initialize counters
    test_counter = 0
    file_counter = 1

    # Loop through the samples beyond demos (2*k + 1 onward)
    for test_index in range(2 * k + 1, len(df)):
        samples_per_file = 5
        # Start a new test file every `samples_per_file` samples
        if test_counter % samples_per_file == 0:
            # samples_per_file is 5 unless there are fewer than 5 samples left
            samples_per_file = len(df) - test_index if (len(df) - test_index) < 5 else 5
            test_file = os.path.join(save_dir, f"{file_counter}_{samples_per_file}tests.txt")
            write_to_txt(test_file, f"Desired output format: {output_format}", mode="w", newline_after=True)
            write_to_txt(test_file, "Instructions: Classify the following samples as 'correct' or 'incorrect'.", mode="a", newline_after=True)
            file_counter += 1

        # Add the current test sample to the file
        csv_path = f'dataset/REHAB24-6//2d_joints_segmented/features/Ex{exercise_num}-segmented/{df.iloc[test_index]["file_name"]}.csv'
        file_contents = read_csv(csv_path)

        write_to_txt(test_file, f"<data {test_index + 1}>", mode="a", newline_after=True)
        write_to_txt(test_file, file_contents, mode="a")
        write_to_txt(test_file, f"<\\data {test_index + 1}>\nLabel: \n", mode="a", newline_after=True)

        test_counter += 1

if __name__ == '__main__':
    # Load annotated.csv and map exercise IDs to names
    annotated_csv_path = 'dataset/REHAB24-6/2d_joints_segmented/annotations.csv'
    df = pd.read_csv(annotated_csv_path)
    # keep only the columns we need
    df = df[['file_name', 'exercise_id', 'person_id', 'repetition', 'correctness']]
    # modify filename in each row for range [:-4] and add '_features.csv' at the end
    df['file_name'] = df['file_name'].apply(lambda x: x[:-4] + '_features')

    movement_map = {
        1: 'Arm abduction',
        2: 'Arm VW',
        3: 'Inclined push up',
        4: 'Leg Abduction',
        5: 'Leg lunge',
        6: 'Squat',
    }

    save_dir_base = 'dataset/REHAB24-6_prompts'
    os.makedirs(save_dir_base, exist_ok=True)

    # trial with one sample
    exercise_id = 2
    person_id = 2
    k = 0

    ################### FOR A SINGLE EXPERIMENT, DEBUGGING PURPOSES:
    # save_dir_txt = os.path.join(save_dir_base, f'{k}shot', f'{k}shot_m{exercise_id:02}_s{person_id:02}') # m: movement, s: subject
    # sample_df = initialize_dataframe(df, exercise_id, person_id, k=k)
    # print(sample_df)
    # create_prompt_files(sample_df, save_dir_txt, exercise_id, movement_map[exercise_id], k)
    # log_experiment(save_dir_txt, sample_df)
    ###################

    # Generate prompts for each exercise, person, and demo configuration
    with tqdm(total=6 * 10 * 4, desc="Processing experiments") as pbar:
        for exercise_id in range(1, 7):  # 6 exercises
            for person_id in range(1, 11):  # 10 persons
                for k in range(4, 6):  # 0 to 5 demos per class
                    save_dir_txt = os.path.join(save_dir_base, f'{k}shot', f'{k}shot_m{exercise_id:02}_s{person_id:02}') # m: movement, s: subject
                    # print(save_dir_txt)
                    sample_df = initialize_dataframe(df, exercise_id, person_id, k=k)
                    # print(sample_df)
                    if not sample_df.empty:
                        create_prompt_files(sample_df, save_dir_txt, exercise_id, movement_map[exercise_id], k)
                        log_experiment(save_dir_txt, sample_df)
                    else:
                        print(f"skipped {save_dir_txt}, not enough samples for {k} shot.")
                    
                    pbar.update(1)

    print("All prompts generated.")