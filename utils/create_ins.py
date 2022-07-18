import json
import gym
import torch
from env.case_generator import CaseGenerator

# Generate instances and save to files
def main():
    batch_size = 100
    num_jobs = 20
    num_mas = 5
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)
    with open("../config.json", 'r') as load_f:
        load_dict = json.load(load_f)

    env_paras = load_dict["env_paras"]
    env_paras["batch_size"] = batch_size
    env_paras["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, flag_same_opes=False, flag_doc=True)
    gym.make('fjsp-v0', case=case, env_paras=env_paras)  # Instances are created when the environment is initialized

if __name__ == "__main__":
    main()
    print("Instances are created and stored in the \"./data\"")