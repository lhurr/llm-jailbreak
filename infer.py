import os
# os.environ["HF_HOME"] = "/tmlscratch/andriush/models/"
import argparse
import random
import string
import torch
import numpy as np
# from loggers import WandBLogger
from judges import load_judge, judge_rule_based
# from language_models import GPT
# from prompts import get_universal_manual_prompt
from conversers import load_target_model
# from utils import insert_adv_string, schedule_n_to_change_fixed, schedule_n_to_change_prob, extract_logprob, early_stopping_condition
# import pandas as pd


parser = argparse.ArgumentParser()
########### Target model parameters ##########
parser.add_argument(
    "--target-model",
    default = "vicuna",
    help = "Name of target model.",
    # choices=["vicuna", "llama2", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview", "claude-instant-1","claude-2", "palm-2"]
)
parser.add_argument(
    "--target-max-n-tokens",
    type = int,
    default = 150,
    help = "Maximum number of generated tokens for the target."
)
##################################################

############ Judge model parameters ##########
parser.add_argument(
    "--judge-model",
    default="no-judge",
    help="Name of judge model.",
    choices=["gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview", "no-judge"]
)
parser.add_argument(
    "--judge-max-n-tokens",
    type = int,
    default = 10,
    help = "Maximum number of tokens for the judge."
)
parser.add_argument(
    "--judge-temperature",
    type=float,
    default=0,
    help="Temperature to use for judge."
)
parser.add_argument(
    "--judge-top-p",
    type=float,
    default=1.0,
    help="Top-p to use for judge."
)
parser.add_argument(
    "--judge-max-n-calls",
    type = int,
    default = 1,
    help = "Maximum number of calls to the judge inside the random search loop."
)
##################################################
args = parser.parse_args()

seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Initialize models and logger 
targetLM = load_target_model(args)


# jb_prompts=df_data["best_msg"]
jb_count=0
import json
# for i in range(len(jb_prompts)):
with open('Behaviors 20240612.json') as data_file:    
    data = json.load(data_file)
    for idx,v in enumerate(data.values()):
        print(v)
        jb_prompt=v
        fp, response=targetLM.get_response([jb_prompt], max_n_tokens=args.target_max_n_tokens, temperature=1.0)
        print(fp[0][''])
        jb_flag=judge_rule_based(fp[0]['text'])
        if jb_flag:
            jb_count=jb_count+1
            # print(f"########################################Batch_{i}########################################")
            print(fp[0]['text'])

print(f"Total Jailbreak: {jb_count}")
# print(f"ASR: {jb_count/len(jb_prompts)}")