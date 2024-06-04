import torch

modelA = torch.load('yolov8s.pt', map_location=torch.device('cpu'))
state_dict_A = modelA["model"].state_dict()

modelB = torch.load('yolov8x.pt', map_location=torch.device('cpu'))
state_dict_B = modelB["model"].state_dict()

model_custom = torch.load('yolov8s-custom.pt', map_location=torch.device('cpu'))
state_dict_custom = model_custom["model"].state_dict()

for k in list(state_dict_A.keys()):
    if "22" in k and "cv3" in k:
        continue
    state_dict_custom[k] = state_dict_A[k]

for k in list(state_dict_B.keys()):
    if "22" in k and "cv3" not in k:
        state_dict_custom[k.replace("22", "26")] = state_dict_B[k]
        state_dict_custom[k.replace("22", "30")] = state_dict_B[k]
        state_dict_custom[k.replace("22", "34")] = state_dict_B[k]
        state_dict_custom[k.replace("22", "38")] = state_dict_B[k]

model_custom["model"].load_state_dict(state_dict_custom, strict=False)
torch.save(model_custom, 'yolov8s-custom.pt')