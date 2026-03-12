import torch

def remove_module_prefix(input_pth, output_pth):
    """
    移除模型权重中的 `module.` 前缀，并保存为新的文件。

    Args:
        input_pth (str): 输入的 .pth 文件路径（包含 `module.` 前缀）。
        output_pth (str): 输出的 .pth 文件路径（去除了 `module.` 前缀）。
    """
    all_dict = torch.load(input_pth, map_location='cpu', weights_only=True)  
    state_dict = all_dict['net']

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:] 
        else:
            new_key = key
        new_state_dict[new_key] = value

    all_dict['net'] = new_state_dict
    print(new_state_dict.keys())

    torch.save(all_dict, output_pth)
    print(f"Saved new state dict to {output_pth}")

input_pth = "./ckpt/best.pth" 
output_pth = "./ckpt/best2.pth"  
remove_module_prefix(input_pth, output_pth)