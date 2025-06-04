import torch
print(torch.__version__)  # 应输出类似 2.1.1+cu118
print(torch.cuda.is_available())  # 应输出 True
