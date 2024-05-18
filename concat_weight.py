import torch

# state_dict = torch.load("pretrained/yolox_s.pth")
# extra_state_dict = torch.load("pretrained/yolox_x.pth")
# for k in list(extra_state_dict["model"].keys()):
#     if "head" in k:
#         state_dict["model"][k.replace("head", "extra_head")] = extra_state_dict["model"][k]
# torch.save(state_dict, "pretrained/yolox_s_with_head_x.pth.tar")
# print(state_dict["model"].keys())

#--------------------------------------------------------------------------

# state_dict = torch.load("pretrained/yolox_s.pth")
# extra_state_dict = torch.load("pretrained/yolox_l.pth")
# extra_state_dict_1 = torch.load("pretrained/yolox_x.pth")
# for k in list(extra_state_dict["model"].keys()):
#     if "head" in k:
#         state_dict["model"][k.replace("head", "extra_head")] = extra_state_dict["model"][k]
# for k in list(extra_state_dict_1["model"].keys()):
#     if "head" in k:
#         state_dict["model"][k.replace("head", "extra_head_1")] = extra_state_dict_1["model"][k]
# torch.save(state_dict, "pretrained/yolox_s_with_head_l_head_x.pth.tar")
# print(state_dict["model"].keys())

#-----------------------------------------------------------------------------
state_dict = torch.load("pretrained/yolox_s.pth")
extra_state_dict = torch.load("pretrained/yolox_x.pth")
extra_state_dict_1 = torch.load("pretrained/yolox_x.pth")
for k in list(extra_state_dict["model"].keys()):
    if "head" in k:
        state_dict["model"][k.replace("head", "extra_head")] = extra_state_dict["model"][k]
for k in list(extra_state_dict_1["model"].keys()):
    if "head" in k:
        state_dict["model"][k.replace("head", "extra_head_1")] = extra_state_dict_1["model"][k]
torch.save(state_dict, "pretrained/yolox_s_with_head_x_head_x.pth.tar")
print(state_dict["model"].keys())