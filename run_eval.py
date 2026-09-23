import torch
from stage7_36_train_eval import evaluate_and_test, generate_visual_trace
from stage7_35_final_truth import SeparateActorCriticSymmetric64
policy = SeparateActorCriticSymmetric64(device=torch.device("cpu"), drone_type="WATER")
policy.load_state_dict(torch.load("stage7_36_battery_20updates.pth"))
evaluate_and_test(policy)
generate_visual_trace(policy)
