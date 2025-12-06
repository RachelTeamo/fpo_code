import torch
import numpy as np
import torch.nn.functional as F

"""
Fisher信息加权的扩散模型损失函数

功能说明：
    使用Fisher信息理论对样本进行自适应加权，提高模型对困难样本的关注度。
    
使用示例:
    # 启用Fisher权重
    loss_fn = SILoss(
        use_fisher_weighting=True,
        fisher_ratio=0.3,  # 对30%的困难样本进行额外加权
        fisher_temperature=1.0,  # 权重分布的锐度
        fisher_weight_strategy="softmax"  # 权重计算策略
    )
    
    # 计算损失
    denoising_loss, proj_loss = loss_fn(model, images, model_kwargs, zs)
"""

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            use_fisher_weighting=False,
            fisher_ratio=0.3,
            fisher_temperature=1.0,
            fisher_aug=2.0,
            use_time_conditional_fisher=False,
            fisher_time_range=(0.0, 0.3)
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        
        # Fisher信息权重相关参数
        self.use_fisher_weighting = use_fisher_weighting
        self.fisher_ratio = fisher_ratio  # 选择较差样本的比例
        self.fisher_temperature = fisher_temperature  # 权重分布的锐度
        self.fisher_aug = fisher_aug
        
        # 时间条件Fisher权重参数
        self.use_time_conditional_fisher = use_time_conditional_fisher  # 是否启用时间条件Fisher权重
        self.fisher_time_range = fisher_time_range  # Fisher权重生效的时间范围，例如(0.0, 0.3)表示只对t∈[0, 0.3]的样本应用

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def compute_fisher_weights(self, losses, time_steps=None):
        """
        基于Fisher信息理论计算样本权重
        
        功能说明：
            根据样本损失计算Fisher信息权重。
            【核心修改】：保留了梯度流 (No-Detach)，并统一使用归一化框架。
            
        输入:
            losses: [batch_size] 每个样本的损失值 (注意：这里传进来的必须带梯度!)
            time_steps: [batch_size] 每个样本的时间步
            
        输出:
            weights: [batch_size] 每个样本的权重
        """
        batch_size = losses.shape[0]
        
        # 初始化权重为1.0
        weights = torch.ones_like(losses)
        
        # 如果启用时间条件Fisher权重，筛选时间范围内的样本
        if self.use_time_conditional_fisher and time_steps is not None:
            time_min, time_max = self.fisher_time_range
            
            # 找出在指定时间范围内的样本
            time_mask = (time_steps >= time_min) & (time_steps <= time_max)
            valid_indices = torch.where(time_mask)[0]
            
            if valid_indices.numel() == 0:
                return weights
            
          
            valid_losses = losses[valid_indices]
            valid_batch_size = valid_losses.shape[0]
            
        else:
            valid_indices = torch.arange(batch_size, device=losses.device)
            valid_losses = losses
            valid_batch_size = batch_size
        
        # 根据损失大小排序
        _, sorted_indices = torch.sort(valid_losses, descending=True)
        
        # 计算需要额外加权的样本数量
        num_high_loss = int(valid_batch_size * self.fisher_ratio)
        
        if num_high_loss > 0:
            # 为损失较大的样本分配更高的权重
            high_loss_indices = sorted_indices[:num_high_loss]
            high_losses = valid_losses[high_loss_indices]
            
            # 初始化额外权重容器
            extra_weights = torch.ones_like(high_losses)
            normalized_losses = high_losses / self.fisher_temperature
            exp_losses = torch.exp(normalized_losses - normalized_losses.max())
            softmax_weights = exp_losses / exp_losses.sum() 
            extra_weights = 1.0 + softmax_weights * self.fisher_aug 
            # 将额外权重赋值给对应的样本
            original_high_loss_indices = valid_indices[high_loss_indices]
            weights[original_high_loss_indices] = extra_weights

            # 归一化开启
            if weights.sum() > 0:
                weights = weights * (batch_size / weights.sum())
        
        return weights
    
    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output, zs_tilde  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # projection loss - 保持每个样本的独立损失用于Fisher加权
        proj_loss = 0.    
        bsz = zs[0].shape[0]
        batch_proj_loss = torch.zeros(bsz, device=images.device, dtype=images.dtype)
        
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            z = torch.nn.functional.normalize(z, dim=-1)
            z_tilde = torch.nn.functional.normalize(z_tilde, dim=-1)
            loss_per_sample = mean_flat(-(z * z_tilde).sum(dim=-1))
            batch_proj_loss += loss_per_sample
        batch_proj_loss /= len(zs)
        
        # 应用Fisher信息权重
        if self.use_fisher_weighting:
            # 计算每个样本的总损失（用于排序）
            # proj_loss_per_sample是负数，对齐越好(接近-1)总损失越小，对齐越差(接近1)总损失越大
            
            # 准备时间步信息（如果启用时间条件Fisher权重）
            time_steps_flat = None
            if self.use_time_conditional_fisher:
                # 将time_input展平为1D，范围[0,1]
                time_steps_flat = time_input.view(-1)
            
            # 计算Fisher权重
            fisher_weights_denoise = self.compute_fisher_weights(
                denoising_loss, 
                time_steps=None
            )
            fisher_weights_proj = self.compute_fisher_weights(
                batch_proj_loss, 
                time_steps=time_steps_flat
            )
            # import pdb; pdb.set_trace()
            # 应用权重到损失
            weighted_denoising_loss = denoising_loss * fisher_weights_denoise
            weighted_proj_loss = batch_proj_loss * fisher_weights_proj
            
            # 返回加权后的平均损失
            denoising_loss_mean = weighted_denoising_loss.mean()
            proj_loss_mean = weighted_proj_loss.mean()
        else:
            # 原始的平均损失计算
            denoising_loss_mean = denoising_loss.mean()
            proj_loss_mean = batch_proj_loss.mean()

        return denoising_loss_mean, proj_loss_mean
