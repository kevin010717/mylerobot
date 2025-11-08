# %%
# 讲义地址 https://www.cnblogs.com/zhangbo2008/p/17341284.html
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch

# ==== 新增：设备选择（CUDA 优先，自动回退 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

"""
conda activate lerobot-MLPDiffusion
pip uninstall -y numpy scipy scikit-learn matplotlib
conda install -y -c conda-forge \
  "numpy=1.26.*" \
  "scipy>=1.10,<1.14" \
  "scikit-learn>=1.3,<1.6" \
  "matplotlib>=3.7,<3.9" \
  --force-reinstall
"""

s_curve,_ = make_s_curve(10**4,noise=0.1)  # 生成一个三维曲线
s_curve = s_curve[:,[0,2]]/10.0  # 只取 xy 轴
print("shape of s:", np.shape(s_curve))

data = s_curve.T
fig,ax = plt.subplots()
ax.scatter(*data,color='blue',edgecolor='white')
ax.axis('off')

# ==== 改动：把 dataset 放到 device
dataset = torch.tensor(s_curve, dtype=torch.float32, device=device)

# %% [markdown]
# 2、确定超参数的值

# %%
num_steps = 100

# 制定每一步的 beta  （在 device 上创建）
betas = torch.linspace(-6, 6, num_steps, device=device)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

# 计算 alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt 等变量
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1.0], device=device), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape==one_minus_alphas_bar_sqrt.shape
print("all the same shape", betas.shape)

# %% [markdown]
# 3、确定扩散过程任意时刻的采样值

# %%
# 计算任意时刻的 x 采样值，基于 x_0 和重参数化
def q_x(x_0, t):
    """可以基于 x[0] 得到任意时刻 t 的 x[t]"""
    # 保证 t 在正确设备、整型
    if not torch.is_tensor(t):
        t = torch.tensor([t], device=device)
    else:
        t = t.to(device)
    t = t.long()

    noise = torch.randn_like(x_0)  # 与 x_0 同设备
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)

# %% [markdown]
# 4、演示原始数据分布加噪100步后的结果

# %%
num_shows = 20
fig,axs = plt.subplots(2,10,figsize=(28,3))
plt.rc('text',color='black')

for i in range(num_shows):
    j = i//10
    k = i%10
    q_i = q_x(dataset, torch.tensor([i*num_steps//num_shows], device=device))
    # 为了绘图，需要搬回 CPU（.cpu().numpy()）
    qi_cpu = q_i.detach().cpu()
    axs[j,k].scatter(qi_cpu[:,0], qi_cpu[:,1], color='red', edgecolor='white')
    axs[j,k].set_axis_off()
    axs[j,k].set_title('$q(\\mathbf{x}_{'+str(i*num_steps//num_shows)+'})$')

# %% [markdown]
# 5、编写拟合逆扩散过程高斯分布的模型

# %%
import torch.nn as nn

class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )
    def forward(self, x, t):
        t = t.long()
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x)
            x = x + t_embedding
            x = self.linears[2*idx+1](x)
        x = self.linears[-1](x)
        return x

# %% [markdown]
# 6、编写训练的误差函数

# %%
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻 t 进行采样计算 loss"""
    batch_size = x_0.shape[0]

    # 生成随机的时刻 t（在 device 上）
    t = torch.randint(0, n_steps, size=(batch_size//2,), device=device)
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1).long()  # [B,1]

    # 系数
    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]

    # 真实噪声
    e = torch.randn_like(x_0)

    # 构造模型输入
    x = x_0 * a + e * aml

    # 预测噪声
    output = model(x, t.squeeze(-1))

    return (e - output).square().mean()

# %% [markdown]
# 7、编写逆扩散采样函数（inference）

# %%
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """从 x[T] 恢复 x[T-1], x[T-2], ... x[0]"""
    cur_x = torch.randn(shape, device=device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从 x[t] 采样到 x[t-1]"""
    t = torch.tensor([t], device=device).long()

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)

    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample

# %% [markdown]
# 8、开始训练模型，打印loss及中间重构效果

# %%
seed = 1234

class EMA():
    """参数平滑器"""
    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}
    def register(self, name, val):
        self.shadow[name] = val.clone()
    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

print('Training model...')
batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epoch = 4000
plt.rc('text', color='blue')

# ==== 改动：模型放到 device
model = MLPDiffusion(num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(num_epoch):
    for idx, batch_x in enumerate(dataloader):
        # DataLoader 已经提供的是 device 上的张量（因为 dataset 在 device 上）
        loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    if (t % 100 == 0):
        print(loss.item())
        x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)

        fig, axs = plt.subplots(1, 10, figsize=(28, 3))
        for i in range(1, 11):
            cur_x = x_seq[i*10].detach().cpu()  # ==== 为绘图搬回 CPU
            axs[i-1].scatter(cur_x[:,0], cur_x[:,1], color='red', edgecolor='white')
            axs[i-1].set_axis_off()
            axs[i-1].set_title('$q(\\mathbf{x}_{'+str(i*10)+'})$')
# ===== 放在第 9 步之前 =====
# 计算全局正方形坐标范围（基于原始数据，留 10% 边距）
xy = dataset.detach().cpu().numpy()
xmin, xmax = xy[:,0].min(), xy[:,0].max()
ymin, ymax = xy[:,1].min(), xy[:,1].max()
cx, cy = (xmin + xmax)/2.0, (ymin + ymax)/2.0
half = max(xmax - xmin, ymax - ymin) * 0.55   # 0.5 为半径，再加 ~10% 留白
xlim = (cx - half, cx + half)
ylim = (cy - half, cy + half)
# %% [markdown]
# 9、动画演示扩散过程和逆扩散过程（固定范围+等比+固定像素）

import io
from PIL import Image

def render_frame(points_xy):
    """生成一帧：正方形画布 + 等比坐标 + 固定范围，不裁剪"""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)  # 固定像素尺寸，可调
    ax.scatter(points_xy[:,0], points_xy[:,1], s=5, color='red', edgecolor='white')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')         # 关键：等比坐标
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)                  # 填满画布

    buf = io.BytesIO()
    fig.savefig(buf, format='png', pad_inches=0)     # 不要 bbox_inches='tight'
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P")              # GIF 更稳更小

# 正向扩散帧
imgs = []
for i in range(100):
    q_i = q_x(dataset, torch.tensor([i], device=device)).detach().cpu().numpy()
    imgs.append(render_frame(q_i))

# 逆扩散帧（确保 x_seq 存在）
if 'x_seq' not in locals():
    x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)

reverse = []
for i in range(100):
    cur_x = x_seq[i].detach().cpu().numpy()
    reverse.append(render_frame(cur_x))

# 合并与保存
all_frames = imgs + reverse
all_frames[0].save(
    "./demo/diffusion.gif",
    format="GIF",
    save_all=True,
    append_images=all_frames[1:],
    duration=100,
    loop=0,
)
