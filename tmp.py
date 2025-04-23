import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 场景1：注册为Parameter
        self.register_parameter("pos_embedding_param", nn.Parameter(torch.randn(1, 256, 512)))
        # 场景2：注册为Buffer
        self.register_buffer("pos_embedding_buffer", torch.randn(1, 256, 512))
        x = torch.randn(1, 256, 512)
        x = x + self.pos_embedding_param
model = TestModel()

print("Parameters:")
for name, _ in model.named_parameters():
    print(name)  # 输出: pos_embedding_param

print("\nBuffers:")
for name, _ in model.named_buffers():
    print(name)  # 输出: pos_embedding_buffer