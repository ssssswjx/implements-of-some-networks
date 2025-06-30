import torch
import torch.nn as nn

batch_size = 2
time_steps = 3
embedding_dim = 4
num_group = 2

torch.manual_seed(0)
input = torch.randn(batch_size, time_steps, embedding_dim)

# 实现batch_norm并验证API
batch_norm_op = torch.nn.BatchNorm1d(num_features=embedding_dim, affine=False, momentum=0)
bn_y = batch_norm_op(input.transpose(-1, -2), ).transpose(-1, -2)

# 手写batch_norm
# C -> batch_szie, time_steps, C
bn_mean = input.mean(dim=(0, 1), keepdim=True)
bn_std = input.std(dim=(0, 1), unbiased=False, keepdim=True)
verift_bn_y = (input - bn_mean) / (bn_std + 1e-5)
# print(bn_y)
# print(verift_bn_y)

print(torch.allclose(verift_bn_y, bn_y))

# 实现layer_norm并验证API
ln = nn.LayerNorm(embedding_dim, elementwise_affine=False)
ln_y = ln(input)

#手写layer_nrom
ln_mean = input.mean(dim=(-1), keepdim=True)
ln_std = input.std(dim=(-1), unbiased=False, keepdim=True)
verify_ln_y = (input - ln_mean) / (ln_std + 1e-5)

print(torch.allclose(ln_y, verify_ln_y))

# 实现instance_norm并验证API
it = nn.InstanceNorm1d(embedding_dim, affine=False)
it_y = it(input.transpose(-1, -2)).transpose(-1, -2)

# 手写instance_nrom
it_mean = input.mean(dim=(1), keepdim=True)
it_std = input.std(dim=(1), unbiased=False, keepdim=True)
verify_it_y = (input - it_mean) / (it_std + 1e-5)

print(torch.allclose(it_y, verify_it_y))


# 实现group_norm并验证API
group_norm = nn.GroupNorm(num_groups=num_group, num_channels=embedding_dim, affine=False)
gn_y = group_norm(input.transpose(-1, -2)).transpose(-1, -2)

# 手写 group_nrom
group_input = torch.split(input, embedding_dim // num_group, dim=-1)
result = []
for g_input in group_input:
    gn_mean = g_input.mean(dim=(1, 2), keepdim=True)
    gn_std = g_input.std(dim=(1, 2), unbiased=False, keepdim=True)
    gn_result = (g_input - gn_mean) / (gn_std + 1e-5)
    result.append(gn_result)
verify_gn_y = torch.cat(result, dim=-1)
print(gn_y)
print(verify_gn_y)


# 实现weight_norm 并验证API
linear = nn.Linear(embedding_dim, 3, bias=False)
wn_linear = torch.nn.utils.weight_norm(linear)
wn_linear_output = wn_linear(input)

# 手写weight_norm
weight_direction = linear.weight / linear.weight.norm(dim=1, keepdim=True)
weight_magnitude = wn_linear.weight_g
print(weight_direction.shape)
print(weight_magnitude.shape)

verify_wn_linear_output = input @ (weight_direction.transpose(-1, -2)) * (weight_magnitude)
print(wn_linear_output, verify_wn_linear_output)

















