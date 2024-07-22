import csv
import os

def read_csv_and_write_to_txt(csv_file_path, txt_file_path, before_data, after_data):
    # Read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_content = [row for row in csv_reader]
    
    # Write to the text file
    with open(txt_file_path, mode='w', encoding='utf-8') as txt_file:
        txt_file.write('\n' + before_data)
        for row in csv_content:
            txt_file.write(','.join(row) + '\n')
        txt_file.write('\n' + after_data)

if __name__ == '__main__':
    # define parameters
    dataname = 'UI-PRMD'
    input_type = 'features'
    downsample_rate = 3
    device = 'kinect'
    correctness = 'correct'
    cor_tag = ''
    subdir = 'positions'
    m = 7
    s = 1
    e = 5

    num_kp = 22 # turn into a parameter
    num_axes = 3 # turn into a parameter

    count = 0
    for correctness in ['correct', 'incorrect']:
        if correctness == 'incorrect':
            cor_tag = '_inc'
        for s in range(1, 11): # m
            # Usage example
            csv_file_path = 'dataset/{}_features/{}/{}/features/m{:02d}_s{:02d}_e{:02d}{}_features.csv'.format(dataname, correctness, device, m, s, e, cor_tag)
            save_dir = 'dataset/{}_prompts/{}/{}/{}'.format(dataname, correctness, device, input_type)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            before_data = 'These are the angles of someone doing one rep of standing shoulder abduction exercise. Movement description: Subject raises one arm to the side by a lateral rotation, keeping the elbow and wrist straight.\n'
            after_data = 'Tell me if the movement is done correctly within 50 words, in the format: [Correctness] [Most important feedback] [How I can improve these movements].'

            save_path = '{}/m{:02d}_s{:02d}_e{:02d}{}_{}.txt'.format(save_dir, m, s, e, cor_tag, input_type)
            print("Save path: ", save_path)

            read_csv_and_write_to_txt(csv_file_path, save_path, before_data, after_data)

            count += 1

    print("Total number of files saved: ", count)
