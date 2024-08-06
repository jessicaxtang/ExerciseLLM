def remove_empty_lines(input_string):
    # Split the input string into a list of lines
    lines = input_string.split('\n')
    
    # Filter out empty lines
    non_empty_lines = [line for line in lines if line.strip() != '']
    
    # Join the non-empty lines back into a single string
    result_string = '\n'.join(non_empty_lines)
    
    return result_string

def convert_labels_to_binary(input_string):
    # Split the input string into individual label lines
    lines = input_string.split('\n')
    
    # Initialize an empty list to store the binary results
    results = []
    
    # Iterate over each line
    for line in lines:
        # Check if the label is "Correct" or "Incorrect"
        if "Correct" in line:
            results.append('1')
        elif "Incorrect" in line:
            results.append('0')
    
    # Join the binary results with newline characters and add an extra newline at the end
    output_string = '\n'.join(results) + '\n'
    
    return output_string


# Example usage
input_string = """
Label 192: Correct, Consistent shoulder abduction angle, Adequate elbow flexion angle, Stable torso inclination.

Label 193: Correct, Consistent shoulder abduction angle, Adequate elbow flexion angle, Stable torso inclination.

Label 194: Correct, Consistent shoulder abduction angle, Adequate elbow flexion angle, Stable torso inclination.

Label 195: Correct, Consistent shoulder abduction angle, Adequate elbow flexion angle, Stable torso inclination.

Label 196: Correct, Consistent shoulder abduction angle, Adequate elbow flexion angle, Stable torso inclination.

Label 197: Correct, Consistent shoulder abduction angle, Adequate elbow flexion angle, Stable torso inclination.

Label 198: Correct, Consistent shoulder abduction angle, Adequate elbow flexion angle, Stable torso inclination.

Label 199: Correct, Consistent shoulder abduction angle, Adequate elbow flexion angle, Stable torso inclination.

Label 200: Incorrect, Inconsistent shoulder abduction angle, Variable elbow flexion angle, Unstable torso inclination.
"""

new = remove_empty_lines(input_string)
print(new)

print(convert_labels_to_binary(input_string))