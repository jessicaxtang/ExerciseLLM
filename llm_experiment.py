'''
Run this script to log LLM responses for web experiments.
for UIPRMD or REHAB24-6.
'''

import pandas as pd
import os

def get_settings():
    """
    Prompt the user for input_type, k-shot, movement, and subject number.
    """
    input_type = input("Enter input type ('abs', 'feat', or 'cot'): ").strip().lower()
    k = int(input("Enter k-shot (0-4): ").strip())
    m = int(input("Enter movement number (1-10): ").strip())
    s = int(input("Enter subject number (1-10): ").strip())
    return input_type, k, m, s

def load_ground_truth(input_type, k, m, s, dataset='UI-PRMD'):
    """
    Load the existing ground truth CSV file for the given parameters.
    """
    file_path = os.path.join("dataset", "UI-PRMD_prompts", input_type, f"{k}shot", f"{k}shot-{input_type}_m{m:02d}_s{s:02d}_e00", "ground_truth.csv")
    if dataset == 'REHAB246':
        file_path = os.path.join("dataset", "REHAB24-6_prompts", f"{k}shot", f"{k}shot_m{m:02d}_s{s:02d}", "ground_truth.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    return pd.read_csv(file_path), file_path

def normalize_response(response):
    """
    Normalize an LLM response by:
    - read responses in the form: "Label 1: correct"
    - Removing extra quotes
    - Stripping leading/trailing whitespace
    - Converting to lowercase
    """
    if ':' in response:
        response = response.split(':', 1)[1]  # Extract the part after ':'
    return response.strip().strip('"').lower()

def get_llm_responses():
    """
    Collect and normalize LLM responses, including multi-line entries.
    Handles various formats such as:
    - "correct"
    - correct
    - incorrect
    - "incorrect"
    """
    print("Paste LLM responses (multi-line allowed). You can paste multiple blocks. Type 'END' on a new line to finish:")
    responses = []
    while True:
        response = input().strip()
        if response.upper() == 'END':
            break
        # Split multi-line inputs into individual lines, normalize, and add to the list
        normalized_responses = [normalize_response(line) for line in response.splitlines()]
        responses.extend(normalized_responses)
    return responses

def update_csv(df, llm_responses, k):
    """
    Update the CSV with normalized LLM responses for correctness,
    starting from the ((k*2)+1)th row.
    """
    # Calculate the starting index
    start_index = (k * 2)

    # Ensure the LLM responses match the appropriate range of rows
    if len(llm_responses) != len(df) - start_index:
        raise ValueError(
            f"Mismatch: {len(llm_responses)} LLM responses for {len(df) - start_index} rows in the CSV "
            f"starting at row {start_index + 1}."
        )

    # Map normalized responses to correctness: "correct" -> 1, "incorrect" -> 0
    llm_correctness = [pd.NA] * start_index  # Initialize with None for the first k*2 rows
    for resp in llm_responses:
        if resp == "correct":
            llm_correctness.append(1)
        elif resp == "incorrect":
            llm_correctness.append(0)
        else:
            raise ValueError(f"Unexpected response format: '{resp}'. Expected 'correct' or 'incorrect'.")

    df['llm_correctness'] = llm_correctness
    return df

def check_combination(tracker, input_type, k, m, s):
    """
    Check if the combination exists and its completion status.
    Returns:
        - 'completed' if the combination is already marked as completed ('y').
        - 'not_completed' if the combination exists but is not completed.
        - 'invalid' if the combination does not exist in the tracker.
    """
    match = tracker[
        (tracker['input_type'] == input_type) &
        (tracker['k'] == k) &
        (tracker['m'] == m) &
        (tracker['s'] == s)
    ]
    
    if match.empty:
        return 'invalid'
    elif match.iloc[0]['completed'] == 'y':
        return 'completed'
    else:
        return 'not_completed'

def update_tracker(tracker, file_path, input_type, k, m, s):
    """
    Mark the combination as completed in the tracker and save the changes.
    """
    tracker.loc[
        (tracker['input_type'] == input_type) &
        (tracker['k'] == k) &
        (tracker['m'] == m) &
        (tracker['s'] == s), 'completed'
    ] = 'y'
    tracker.to_csv(file_path, index=False)
    print("Tracker updated successfully.")

def run_test(dataname):
    # Load the test tracker
    tracker_path = f'test_tracker_{dataname}.csv'
    tracker = pd.read_csv(tracker_path)

    while True:
        try:
            # Get experiment settings from user
            input_type, k, m, s = get_settings()

            # Check if the combination is valid or completed
            status = check_combination(tracker, input_type, k, m, s)

            if status == 'invalid':
                print("Invalid combination. Please try again.")
                continue
            elif status == 'completed':
                proceed = input(
                    f"The combination (input_type: {input_type}, k: {k}, m: {m}, s: {s}) "
                    f"has already been completed. Do you want to proceed anyway? (yes/no): "
                ).strip().lower()
                if proceed != 'yes':
                    print("Please enter a new combination.")
                    continue

            # Load existing CSV
            ground_truth_df, ground_truth_path = load_ground_truth(input_type, k, m, s, dataname)

            print(f"Loaded ground truth file: {ground_truth_path}")

            # Get and update with LLM responses
            llm_responses = get_llm_responses()
            updated_df = update_csv(ground_truth_df, llm_responses, k)

            # Save updated CSV
            updated_df.to_csv(ground_truth_path, index=False)
            print(f"Updated CSV saved to: {ground_truth_path}")

            # Mark the combination as completed in the tracker
            update_tracker(tracker, tracker_path, input_type, k, m, s)

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        # Prompt to continue or exit
        cont = input("Process another file? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting.")
            break

if __name__ == '__main__':
    run_test("REHAB246")
    # run_test("UIPRMD")