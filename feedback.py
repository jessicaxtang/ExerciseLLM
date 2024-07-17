import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# base model
model = AutoModelForCausalLM.from_pretrained(
    # "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-llama-3-8B",
    # cache_dir = "/data/base_models",
    device_map = "auto", # takes care of using gpu or cpu automatically
    token = "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"
    )

tokenizer = AutoTokenizer.from_pretrained(
    # "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-llama-3-8B"
    )

instruct_model =  AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    # cache_dir = "/data/base_models",
    device_map = "auto", # takes care of using gpu or cpu automatically
    token = "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",  
    trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage = True,  
    offload_folder=r"transformers\\Llama-2-7b-hf\\offload"
    ).eval()

instruct_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    )

def get_response(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature= 0.00001)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def get_instruct_reponse(prompt, max_new_tokens=50):
    inputs = instruct_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = instruct_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature= 0.00001)
    response = instruct_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# preprocess motion data from T2M-GPT generated motion
"""
motion_data = np.load('motion.npy')
motion_shape = motion_data.shape
motion_data = motion_data.reshape(motion_data.shape[1], 22, 3)
joints = [16, 18, 20] # shoulder, elbow, wrist
sliced_motion = motion_data[:, joints, :]

motion_name = 'waving arms above head'
"""
sliced_motion = np.load('positions_incorrect_m07_s01_e01.npy')
motion_name = 'right shoulder abduction'

# prompt = "Q: is the " + motion_name + " exercise performed correctly given position data of the right upper arm, forearm, and hand?\n" + str(sliced_motion) + " A:"
# prompt = "Q: is the " + motion_name + " exercise performed correctly, can you provide feedback to help me correct it? Here is position data of the right upper arm, forearm, and hand?\n" + str(sliced_motion) + "A:"
prompt = "Q: I am providing you with the x, y, and z coordinates of position data of the right upper arm, forearm, and hand from someone completing the right shoulder abduction exercise. Using just this data, tell me if the exercise was performed correctly or not. (don't ask for additional data)\r\n" + str(sliced_motion) + "\nA:"

# inputs = tokenizer(prompt, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=10)
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)

response = get_response(prompt, max_new_tokens=80)
print(response)