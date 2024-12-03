'''
Script to compile the results of the LLM evaluation for a specific input type and k-shot combination.
Run after adding all LLM responses for the given input type and k-shot to csv file.
Run llm_experiment.py to log LLM responses for web experiments.
'''
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Set directory and parameters
input_type = "abs"  # Change as needed
k_shot = 0  # Change as needed
output_file_global = "classification_metrics_global.csv"
output_file_movement = "classification_metrics_per_movement.csv"
print(f"Evaluating {input_type} {k_shot}-shot results...")

# Initialize results storage
global_metrics = pd.DataFrame()
movement_metrics = pd.DataFrame()

# Loop through each combination of input_type and k-shot
input_types = ["abs", "feat", "cot"]  # Modify as needed
k_shots = range(4)  # Modify if more k-shot settings exist

for input_type in input_types:
    for k_shot in k_shots:
        # Initialize combined DataFrame for the current input_type and k_shot
        combined_results = pd.DataFrame()

        # Define the directory path for the specific input_type and k-shot combination
        directory_path = os.path.join("dataset", "UI-PRMD_prompts", input_type, f"{k_shot}shot")
        print(f"data path: {directory_path}")
        
        # Iterate over the files in the directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith("ground_truth.csv"):
                    file_path = os.path.join(root, file)
                    # Read the CSV file
                    data = pd.read_csv(file_path)
                    # Append data to the combined dataframe
                    combined_results = pd.concat([combined_results, data], ignore_index=True)
        
        # If combined_results is empty, skip processing
        if combined_results.empty:
            print(f"No data found for {input_type}_{k_shot}shot.")
            continue

        # Drop rows with missing llm_correctness
        combined_results = combined_results.dropna(subset=['llm_correctness'])

        # Ensure there are still rows left
        if combined_results.empty:
            print(f"No valid rows after filtering for {input_type}_{k_shot}shot.")
            continue

        # Global metrics for the entire dataset
        y_true = combined_results['correctness']
        y_pred = combined_results['llm_correctness']
    
        global_metrics_row = {
            f"{input_type}_{k_shot}shot": [
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred, zero_division=0),
                recall_score(y_true, y_pred, zero_division=0),
                f1_score(y_true, y_pred, zero_division=0)
            ]
        }

        # Append global metrics
        global_metrics = pd.concat([global_metrics, pd.DataFrame(global_metrics_row, index=["accuracy", "precision", "recall", "f1-score"]).T])

        # Per-movement metrics
        movements = combined_results['movement'].unique()
        for movement in movements:
            movement_data = combined_results[combined_results['movement'] == movement]
            y_true_movement = movement_data['correctness']
            y_pred_movement = movement_data['llm_correctness']

            movement_metrics_row = {
                "input_type": input_type,
                "k_shot": k_shot,
                "movement": movement,
                "accuracy": accuracy_score(y_true_movement, y_pred_movement),
                "precision": precision_score(y_true_movement, y_pred_movement, zero_division=0),
                "recall": recall_score(y_true_movement, y_pred_movement, zero_division=0),
                "f1_score": f1_score(y_true_movement, y_pred_movement, zero_division=0)
            }
            # Append movement metrics
            movement_metrics = pd.concat([movement_metrics, pd.DataFrame([movement_metrics_row])], ignore_index=True)

# Save global metrics to CSV
if not global_metrics.empty:
    global_metrics.to_csv(output_file_global, index_label="Metric")
    print(f"Global metrics saved to {output_file_global}.")
else:
    print("No global metrics calculated.")

# Save per-movement metrics to CSV
if not movement_metrics.empty:
    movement_metrics.to_csv(output_file_movement, index=False)
    print(f"Per-movement metrics saved to {output_file_movement}.")
else:
    print("No per-movement metrics calculated.")