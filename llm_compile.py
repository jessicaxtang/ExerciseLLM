'''
Script to compile the results of the LLM evaluation for a specific input type and k-shot combination.
Run after adding all LLM responses for the given input type and k-shot to csv file.
Run llm_experiment.py to log LLM responses for web experiments.
'''

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

OUTPUT_FILE = "classification_results.csv"

def evaluate_all_metrics():
    """
    Evaluate metrics for all input types and k-shot combinations and save results to a single CSV file.
    """

    input_types = ["abs", "feat", "cot"]
    k_shots = range(4) 

    metrics_data = []

    for input_type in input_types:
        for k_shot in k_shots:
            combined_results = compile_results(input_type, k_shot)
            if combined_results.empty:
                print(f"No data found for {input_type}_{k_shot}shot.")
                continue

            # Global metrics
            metrics_data.append(calculate_global_metrics(combined_results, input_type, k_shot))

            # Per-movement metrics
            movements = combined_results['movement'].unique()
            for movement in movements:
                metrics_data.append(calculate_movement_metrics(combined_results, input_type, k_shot, movement))

    save_metrics_to_csv(metrics_data)


def compile_results(input_type, k_shot):
    """
    Compile all results for a specific input type and k-shot combination.
    """
    directory_path = os.path.join("dataset", "UI-PRMD_prompts", input_type, f"{k_shot}shot")
    print(f"Data path: {directory_path}")
    combined_results = pd.DataFrame()

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith("ground_truth.csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                combined_results = pd.concat([combined_results, data], ignore_index=True)

    combined_results = combined_results.dropna(subset=['llm_correctness'])
    return combined_results


def calculate_global_metrics(combined_results, input_type, k_shot):
    """
    Calculate global metrics for the combined results.
    """
    y_true = combined_results['correctness']
    y_pred = combined_results['llm_correctness']
    return {
        "input_type": input_type,
        "k_shot": k_shot,
        "movement": "global",
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


def calculate_movement_metrics(combined_results, input_type, k_shot, movement):
    """
    Calculate metrics for a specific movement.
    """
    movement_data = combined_results[combined_results['movement'] == movement]
    y_true = movement_data['correctness']
    y_pred = movement_data['llm_correctness']
    return {
        "input_type": input_type,
        "k_shot": k_shot,
        "movement": movement,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


def save_metrics_to_csv(metrics_data):
    """
    Save all metrics to the output CSV file.
    """
    df = pd.read_csv(OUTPUT_FILE)
    new_data = pd.DataFrame(metrics_data)

    # Merge new metrics into the existing data
    merged_data = pd.concat([df, new_data]).drop_duplicates(subset=["input_type", "k_shot", "movement"], keep="last")
    merged_data.to_csv(OUTPUT_FILE, index=False)
    print(f"Metrics saved to {OUTPUT_FILE}.")


def evaluate_single_movement(input_type, k_shot, movement):
    """
    Evaluate metrics for a specific movement and update the CSV file with the results.
    """

    if not (1 <= movement <= 10):
        print("Invalid movement. Please enter a number between 1 and 10.")
        return

    combined_results = compile_results(input_type, k_shot)
    if combined_results.empty:
        print(f"No data found for {input_type}_{k_shot}shot.")
        return

    movement_data = combined_results[combined_results['movement'] == movement]
    if movement_data.empty:
        print(f"No data for movement '{movement}' in {input_type}_{k_shot}shot.")
        return

    metrics = calculate_movement_metrics(combined_results, input_type, k_shot, movement)
    print(f"Metrics for movement '{movement}' ({input_type}_{k_shot}shot): {metrics}")

    # Update or append the metrics to the CSV
    df = pd.read_csv(OUTPUT_FILE)

    # Check if this entry already exists
    match_condition = (
        (df["input_type"] == input_type) &
        (df["k_shot"] == k_shot) &
        (df["movement"] == movement)
    )

    if not df[match_condition].empty:
        df.loc[match_condition, ["accuracy", "precision", "recall", "f1_score"]] = [
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"]
        ]
    else:
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Metrics updated in {OUTPUT_FILE}.")


if __name__ == "__main__":

    if not os.path.exists(OUTPUT_FILE):
        columns = ["input_type", "k_shot", "movement", "accuracy", "precision", "recall", "f1_score"]
        pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False)
        print(f"{OUTPUT_FILE} created.")
        
    # User input
    mode = input("Enter 'all' to evaluate all metrics or a movement number (1-10) to evaluate a single movement: ").strip().lower()

    if mode == "all":
        evaluate_all_metrics()
    elif mode.isdigit() and (1 <= int(mode) <= 10):
        input_type = input("Enter input type (e.g., 'abs', 'feat', 'cot'): ").strip()
        k_shot = int(input("Enter k-shot value (e.g., 0, 1, 2, 3): ").strip())
        movement = int(mode)
        evaluate_single_movement(input_type, k_shot, movement)
    else:
        print("Invalid option. Please enter 'all' or a movement number between 1 and 10.")
