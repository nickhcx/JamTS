import torch
n = 50
M = 100
m = torch.randn(1, n)
M_list = [torch.randn(1, n) for i in range(M)]

output_list = [torch.ger(m.squeeze(), M_i.squeeze()) for M_i in M_list]

output = torch.cat(output_list, dim=1)

output = output.view(1, 1, int(M), int(n*n))

kernel_size = 3
conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size)
output = conv(output)

output = output.view(-1)

fc = torch.nn.Linear(output.shape[0], 1)
weights = fc(output)

