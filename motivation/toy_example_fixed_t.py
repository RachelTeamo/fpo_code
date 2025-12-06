import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import copy

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 创建绘图结果保存文件夹
PLOT_DIR = Path(__file__).parent / "plots_fixed_t"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Plots will be saved to: {PLOT_DIR}\n")


# --- Flow Matching 模型定义 ---
class FlowMatchingMLP(nn.Module):
    """
    Flow Matching 模型的速度场网络。
    
    参数:
        input_dim (int): 输入维度（位置 + 时间）
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度（速度场维度）
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, t):
        """前向传播"""
        if t.dim() == 1:
            t = t.unsqueeze(1)
        xt = torch.cat([x, t], dim=1)
        return self.layers(xt)


# --- 数据生成（保持原样，但训练时 t 会被重新采样）---
def get_flow_matching_data(n_samples=2048, batch_size=64):
    """
    生成 flow matching 训练数据。
    
    注意：这里生成的 t 只用于评估，训练时会重新采样。
    """
    X_target, _ = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    X_target = torch.tensor(X_target, dtype=torch.float32)
    
    X_source = torch.randn_like(X_target)
    
    # 这些 t 用于评估
    t = torch.rand(n_samples, 1)
    x_t = (1 - t) * X_source + t * X_target
    v_t = X_target - X_source
    
    dataset = TensorDataset(X_source, X_target, v_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 返回评估用的数据
    return loader, (x_t.to(DEVICE), t.squeeze().to(DEVICE), v_t.to(DEVICE))


# --- 模型训练（固定 t）---
def train_model_with_checkpoints_fixed_t(model, loader, epochs=100, checkpoint_epochs=None):
    """
    训练 flow matching 模型，每个 batch 使用相同的 t。
    
    关键修改：每个 batch 内所有样本使用相同的随机 t 值。
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [10, 25, 50, 100]
    
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    checkpoints = {}
    
    print("Training with FIXED t per batch (all samples in a batch use same t)...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_source_batch, X_target_batch, v_t_batch in loader:
            X_source_batch = X_source_batch.to(DEVICE)
            X_target_batch = X_target_batch.to(DEVICE)
            v_t_batch = v_t_batch.to(DEVICE)
            
            batch_size = X_source_batch.size(0)
            
            # 关键：每个 batch 采样一个相同的 t
            t_batch = torch.rand(1).to(DEVICE)  # 单个 t 值
            t_batch = t_batch.repeat(batch_size)  # 复制给整个 batch
            
            # 计算插值位置
            t_expanded = t_batch.unsqueeze(1)
            x_t_batch = (1 - t_expanded) * X_source_batch + t_expanded * X_target_batch
            
            # 前向传播
            optimizer.zero_grad()
            v_pred = model(x_t_batch, t_batch)
            loss = criterion(v_pred, v_t_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 保存检查点
        if epoch in checkpoint_epochs:
            checkpoints[epoch] = copy.deepcopy(model.state_dict())
            avg_loss = epoch_loss / len(loader)
            print(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f} [Checkpoint saved]")
        elif epoch % 20 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"  Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    print(f"Training complete. Saved {len(checkpoints)} checkpoints.\n")
    return checkpoints


# --- 计算单个模型的指标 ---
def get_per_sample_metrics(model, x_t_all, t_all, v_t_all, verbose=True):
    """计算每个样本的 MSE 损失值和梯度范数"""
    if verbose:
        print("Calculating per-sample metrics...")
    
    model.eval()
    n_samples = x_t_all.shape[0]
    losses = []
    grad_norms = []
    
    for i in range(n_samples):
        x_t = x_t_all[i:i+1]
        t = t_all[i:i+1]
        v_t = v_t_all[i:i+1]
        
        v_pred = model(x_t, t)
        loss = nn.functional.mse_loss(v_pred, v_t, reduction='none').sum()
        losses.append(loss.item())
        
        model.zero_grad()
        loss.backward()
        
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.flatten())
        
        all_grads = torch.cat(grad_list)
        grad_norm = torch.norm(all_grads, p=2).item()
        grad_norms.append(grad_norm)
    
    if verbose:
        print("Calculation complete.\n")
    
    return np.array(losses), np.array(grad_norms)


# --- 为所有检查点计算指标 ---
def compute_metrics_for_checkpoints(model_template, checkpoints, x_t_all, t_all, v_t_all):
    """为所有检查点计算损失和梯度范数"""
    results = {}
    
    for epoch in sorted(checkpoints.keys()):
        print(f"Computing metrics for Epoch {epoch}...")
        model = copy.deepcopy(model_template)
        model.load_state_dict(checkpoints[epoch])
        model.to(DEVICE)
        
        losses, grad_norms = get_per_sample_metrics(
            model, x_t_all, t_all, v_t_all, verbose=False
        )
        results[epoch] = (losses, grad_norms)
        print(f"  Epoch {epoch}: Mean loss = {losses.mean():.4f}, Mean grad norm = {grad_norms.mean():.4f}\n")
    
    return results


# --- 可视化 ---
def plot_correlation_multi_epochs(results_dict, n_samples_per_epoch=800):
    """绘制多个 epoch 的损失-梯度相关性图"""
    print("Plotting multi-epoch correlation (Fixed t per batch)...")
    
    epochs = sorted(results_dict.keys())
    n_epochs = len(epochs)
    colors = plt.cm.viridis(np.linspace(0, 1, n_epochs))
    
    fig, ax = plt.subplots(figsize=(12, 9))
    correlations = {}
    
    for i, epoch in enumerate(epochs):
        losses, grad_norms = results_dict[epoch]
        rho, p_val = spearmanr(losses, grad_norms)
        correlations[epoch] = (rho, p_val)
        
        n_total = len(losses)
        if n_total > n_samples_per_epoch:
            indices = np.random.choice(n_total, n_samples_per_epoch, replace=False)
            losses_plot = losses[indices]
            grad_norms_plot = grad_norms[indices]
        else:
            losses_plot = losses
            grad_norms_plot = grad_norms
        
        ax.scatter(
            losses_plot, grad_norms_plot,
            alpha=0.4, s=18,
            c=[colors[i]],
            label=f'Epoch {epoch} (ρ={rho:.3f})'
        )
    
    ax.set_title('Loss vs. Gradient Norm (Fixed t per batch)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Per-Sample MSE Loss (Log Scale)', fontsize=13)
    ax.set_ylabel('Per-Sample Gradient L2 Norm (Log Scale)', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, None)
    ax.set_ylim(1e0, None)
    ax.grid(True, which="both", linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    save_path = PLOT_DIR / "loss_vs_gradient_fixed_t.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}\n")
    plt.close()
    
    return correlations


# --- 分析重合率 ---
def analyze_overlap_multi_epochs(results_dict, percentile=75):
    """分析多个 epoch 的重合率"""
    print(f"{'='*70}")
    print(f"Analyzing Overlap (Fixed t, Top {100-percentile}% samples)")
    print(f"{'='*70}\n")
    
    all_stats = {}
    
    for epoch in sorted(results_dict.keys()):
        losses, grad_norms = results_dict[epoch]
        
        loss_threshold = np.percentile(losses, percentile)
        grad_threshold = np.percentile(grad_norms, percentile)
        
        high_loss_mask = losses >= loss_threshold
        high_grad_mask = grad_norms >= grad_threshold
        overlap_mask = high_loss_mask & high_grad_mask
        
        n_high_loss = high_loss_mask.sum()
        n_high_grad = high_grad_mask.sum()
        n_overlap = overlap_mask.sum()
        
        overlap_rate = n_overlap / n_high_loss if n_high_loss > 0 else 0
        
        all_stats[epoch] = {
            'overlap_rate': overlap_rate,
            'n_overlap': n_overlap,
            'n_high_loss': n_high_loss
        }
        
        print(f"Epoch {epoch}:")
        print(f"  Loss threshold: {loss_threshold:.6f}")
        print(f"  Grad threshold: {grad_threshold:.6f}")
        print(f"  Overlap samples: {n_overlap}/{n_high_loss}")
        print(f"  P(High Grad | High Loss) = {overlap_rate*100:.2f}%\n")
    
    return all_stats


# --- 绘制重合率趋势 ---
def plot_overlap_and_loss_trends(overlap_stats, results_dict):
    """绘制重合率和平均损失随 epoch 变化的趋势图"""
    print("Plotting overlap rate trends (Fixed t)...")
    
    epochs = sorted(overlap_stats.keys())
    overlap_rates = [overlap_stats[e]['overlap_rate'] * 100 for e in epochs]
    mean_losses = [results_dict[e][0].mean() for e in epochs]
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color1 = 'steelblue'
    ax1.set_xlabel('Training Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('P(High Grad | High Loss) [%]', fontsize=13, color=color1, fontweight='bold')
    line1 = ax1.plot(epochs, overlap_rates, 'o-', linewidth=3, markersize=10, 
                     color=color1, label='Overlap Rate')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=11)
    ax1.set_ylim([0, 100])
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    for epoch, rate in zip(epochs, overlap_rates):
        ax1.text(epoch, rate + 3, f'{rate:.1f}%', ha='center', fontsize=10, 
                color=color1, fontweight='bold')
    
    ax2 = ax1.twinx()
    color2 = 'crimson'
    ax2.set_ylabel('Mean Loss (MSE)', fontsize=13, color=color2, fontweight='bold')
    line2 = ax2.plot(epochs, mean_losses, 's--', linewidth=3, markersize=9, 
                     color=color2, label='Mean Loss')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=11)
    
    ax1.set_title('Overlap Rate with Fixed t per Batch\n(percentile=75: Top 25% samples)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    save_path = PLOT_DIR / "overlap_trends_fixed_t.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}\n")
    plt.close()


# --- 主函数 ---
def main():
    """主函数：使用固定 t（每个 batch 相同 t）训练"""
    print(f"{'='*70}")
    print("Flow Matching with FIXED t per Batch")
    print(f"{'='*70}\n")
    
    n_samples = 4096
    batch_size = 128
    train_epochs = 100
    checkpoint_epochs = [10, 25, 50, 100]
    
    print("[1/6] Generating dataset...")
    data_loader, (x_t_all, t_all, v_t_all) = get_flow_matching_data(n_samples, batch_size)
    print(f"Dataset: {n_samples} samples\n")
    
    print("[2/6] Training model with FIXED t per batch...")
    model = FlowMatchingMLP(input_dim=3, hidden_dim=64, output_dim=2)
    checkpoints = train_model_with_checkpoints_fixed_t(
        model, data_loader, 
        epochs=train_epochs,
        checkpoint_epochs=checkpoint_epochs
    )
    
    print("[3/6] Computing metrics for all checkpoints...")
    results = compute_metrics_for_checkpoints(
        model, checkpoints, x_t_all, t_all, v_t_all
    )
    
    print("[4/6] Visualizing multi-epoch correlation...")
    correlations = plot_correlation_multi_epochs(results, n_samples_per_epoch=800)
    
    print("Spearman correlations:")
    for epoch, (rho, p_val) in sorted(correlations.items()):
        print(f"  Epoch {epoch}: ρ = {rho:.4f}, p-value = {p_val:.4e}")
    print()
    
    print("[5/6] Analyzing overlap rates...")
    overlap_stats = analyze_overlap_multi_epochs(results, percentile=75)
    
    print("[6/6] Plotting overlap trends...")
    plot_overlap_and_loss_trends(overlap_stats, results)
    
    print(f"{'='*70}")
    print("Analysis Complete! (Fixed t per batch)")
    print(f"{'='*70}\n")
    print("Key Difference:")
    print("  In this version, all samples within a batch use the SAME t value.")
    print("  Compare with the original version where each sample has different t.\n")
    
    return results, correlations, overlap_stats


if __name__ == "__main__":
    main()

