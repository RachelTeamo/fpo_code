"""
Loss Function Value and Gradient Magnitude Relationship Analyzer

This module analyzes the relationship between loss function values and corresponding gradient magnitudes
for different samples across various timesteps in diffusion models. By selecting representative timesteps,
it computes loss values and gradient norms for individual samples, helping understand the correlation
between loss and gradients during model training.

Main Features:
1. Calculate loss function values for specified timesteps
2. Compute corresponding gradient L2 norms for individual samples
3. Statistical analysis of loss-gradient relationships across samples
4. Generate visualization charts
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import random
from typing import List, Tuple, Dict
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

# Set seeds for reproducibility
def set_random_seeds(seed: int = 42):
    """
    Set random seeds for all libraries to ensure reproducible results
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Enable deterministic algorithms in PyTorch (if available)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)
    
    print(f"Random seeds set to {seed} for reproducible results")
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.create_model import create_model
from diffusion import create_diffusion
from util.data_util import center_crop_arr
from diffusers.models import AutoencoderKL


class LossGradientAnalyzer:
    """
    Loss Function Value and Gradient Magnitude Relationship Analyzer Class
    
    This class encapsulates all functionality for analyzing the relationship between
    diffusion model losses and gradients, including model initialization, data processing,
    gradient computation, and result analysis for individual samples.
    """
    
    def __init__(self, model_config: str, image_size: int, num_classes: int, data_path: str, data_samples: int, random_seed: int = 42):
        """
        Initialize the analyzer
        
        Args:
            model_config: Path to model configuration file
            image_size: Input image size
            num_classes: Number of classes
            data_path: Dataset path
            data_samples: Number of data samples
            random_seed: Random seed for reproducibility
        """
        self.model_config = model_config
        self.image_size = image_size
        self.num_classes = num_classes
        self.data_path = data_path
        self.scaling_factor = 0.18215
        self.data_samples = data_samples
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        set_random_seeds(self.random_seed)
        
        # Initialize model and related components
        self._initialize_model()
        self._initialize_dataset()
    
    def _initialize_model(self) -> None:
        """
        Initialize diffusion model, VAE encoder, and diffusion process
        
        This method is responsible for loading and configuring all necessary model components,
        ensuring they are on the correct device and in training mode.
        """
        assert torch.cuda.is_available(), "Analysis process requires at least one GPU device"
        
        # Load model configuration
        config = OmegaConf.load(self.model_config)
        latent_size = self.image_size // 8
        config.model.param["latent_size"] = latent_size
        config.model.param["num_classes"] = self.num_classes
        
        # Create and configure model
        self.model = create_model(model_config=config.model)
        self.model.to("cuda")
        self.model.train()
        
        # Initialize diffusion process and VAE encoder
        self.diffusion = create_diffusion(timestep_respacing="")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to("cuda")
    
    def _initialize_dataset(self) -> None:
        """
        Initialize dataset and data loader
        
        Create image preprocessing pipeline and data loader
        for subsequent loss and gradient calculations.
        """
        # Define image preprocessing transforms
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        
        # Create dataset and data loader
        self.dataset = ImageFolder(self.data_path, transform=transform)
        
        # Create deterministic data loader for reproducibility
        def worker_init_fn(worker_id):
            np.random.seed(self.random_seed + worker_id)
            random.seed(self.random_seed + worker_id)
            
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.data_samples,  # Use specified batch size
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )
    
    def _vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space using VAE encoder
        
        Args:
            x: Input image tensor
            
        Returns:
            Encoded latent representation
        """
        with torch.no_grad():
            # Map input images to latent space and normalize
            latent = self.vae.encode(x).latent_dist.sample().mul_(self.scaling_factor)
        return latent
    
    def _calculate_sample_gradient_magnitude(self, loss: torch.Tensor) -> float:
        """
        Calculate L2 norm of model parameter gradients for a single sample loss
        
        Args:
            loss: Loss value for a single sample
            
        Returns:
            L2 norm of all parameter gradients for this sample
        """
        # Clear existing gradients
        self.model.zero_grad()
        
        # Compute gradients for this specific sample
        loss.backward(retain_graph=True)
        
        total_norm = 0.0
        param_count = 0
        
        # Iterate through all parameters with gradients
        for param in self.model.parameters():
            if param.grad is not None:
                # Calculate squared sum of parameter gradients
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        # Return L2 norm
        return total_norm ** 0.5 if param_count > 0 else 0.0
    
    def analyze_sample_loss_gradient_relationship(
        self, 
        weight_path: str, 
        selected_timesteps: List[int] = None,
        num_samples_per_timestep: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Analyze the relationship between loss function values and gradient magnitudes
        for different samples across timesteps
        
        Args:
            weight_path: Path to model weight file
            selected_timesteps: List of selected timesteps, auto-select if None
            num_samples_per_timestep: Number of samples to analyze per timestep
            
        Returns:
            Dictionary containing loss values, gradient magnitudes, timesteps, and sample IDs
        """
        # Load model weights
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location="cuda")
            self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
            print(f"Successfully loaded model weights: {weight_path}")
        else:
            print(f"Warning: Weight file does not exist, using randomly initialized model: {weight_path}")
        
        # Auto-select 10 representative timesteps if not specified
        if selected_timesteps is None:
            total_timesteps = self.diffusion.num_timesteps
            selected_timesteps = [
                int(i * total_timesteps / 10) for i in range(10)
            ]
            selected_timesteps = selected_timesteps[1:]
        
        print(f"Analyzing timesteps: {selected_timesteps}")
        
        # Storage for results
        all_loss_values = []
        all_gradient_magnitudes = []
        all_timesteps = []
        all_sample_ids = []
        
        # Get a fixed set of samples for consistent comparison
        sample_batch = next(iter(self.dataloader))
        x_samples, y_samples = sample_batch
        x_samples, y_samples = x_samples.cuda(), y_samples.cuda()
        
        # Limit to specified number of samples
        if x_samples.shape[0] > num_samples_per_timestep:
            x_samples = x_samples[:num_samples_per_timestep]
            y_samples = y_samples[:num_samples_per_timestep]
        
        print(f"Analyzing {x_samples.shape[0]} samples across {len(selected_timesteps)} timesteps")
        
        # Iterate through selected timesteps
        for timestep in tqdm(selected_timesteps, desc="Analyzing timesteps"):
            # Create timestep tensor
            t = torch.full((x_samples.shape[0],), timestep, dtype=torch.long, device="cuda")
            
            # Encode images to latent space
            latent = self._vae_encode(x_samples)
            
            # Calculate loss for each sample individually
            model_kwargs = dict(y=y_samples)
            loss_dict = self.diffusion.training_losses(self.model, latent, t, model_kwargs)
            sample_losses = loss_dict["loss"]  # Shape: [batch_size]
            
            # Calculate gradient magnitude for each sample
            for sample_idx in range(sample_losses.shape[0]):
                sample_loss = sample_losses[sample_idx]
                
                # Calculate gradient magnitude for this specific sample
                gradient_magnitude = self._calculate_sample_gradient_magnitude(sample_loss)
                
                # Store results
                all_loss_values.append(sample_loss.item())
                all_gradient_magnitudes.append(gradient_magnitude)
                all_timesteps.append(timestep)
                all_sample_ids.append(sample_idx)
            
            print(f"Timestep {timestep}: Processed {sample_losses.shape[0]} samples")
        
        # Convert to numpy arrays
        results = {
            'timesteps': np.array(all_timesteps),
            'loss_values': np.array(all_loss_values),
            'gradient_magnitudes': np.array(all_gradient_magnitudes),
            'sample_ids': np.array(all_sample_ids)
        }
        
        print(f"Analysis complete: {len(all_loss_values)} total sample-timestep pairs")
        return results
    
    def analyze_sample_loss_gradient_correspondence(
        self, 
        results: Dict[str, np.ndarray],
        tracked_samples: List[int] = None
    ) -> Dict[str, Dict]:
        """
        Analyze the correspondence between loss and gradient for each sample at each timestep
        
        Args:
            results: Results from analyze_sample_loss_gradient_relationship
            tracked_samples: List of specific sample IDs to track across timesteps
            
        Returns:
            Dictionary containing detailed correspondence analysis for each timestep and sample tracking
        """
        timesteps = results['timesteps']
        losses = results['loss_values']
        gradients = results['gradient_magnitudes']
        sample_ids = results['sample_ids']
        
        unique_timesteps = np.unique(timesteps)
        unique_samples = np.unique(sample_ids)
        correspondence_analysis = {}
        
        # Initialize sample tracking
        if tracked_samples is None:
            # Auto-select first 5 samples for tracking
            tracked_samples = sorted([int(s) for s in unique_samples])[:5]
        
        # Ensure tracked samples exist in the data and convert to int
        tracked_samples = [int(s) for s in tracked_samples if s in unique_samples]
        
        print(f"Analyzing loss-gradient correspondence for each timestep...")
        print(f"Tracking samples: {tracked_samples}")
        
        # Initialize sample tracking data
        sample_tracking = {
            sample_id: {
                'timesteps': [],
                'loss_values': [],
                'gradient_values': [],
                'loss_ranks': [],
                'gradient_ranks': [],
                'rank_differences': []
            } for sample_id in tracked_samples
        }
        
        for timestep in unique_timesteps:
            # Get data for this timestep
            timestep_mask = timesteps == timestep
            timestep_losses = losses[timestep_mask]
            timestep_gradients = gradients[timestep_mask]
            timestep_sample_ids = sample_ids[timestep_mask]
            
            # Create sample data list
            sample_data = []
            for i, sample_id in enumerate(timestep_sample_ids):
                sample_data.append({
                    'sample_id': int(sample_id),
                    'loss': float(timestep_losses[i]),
                    'gradient': float(timestep_gradients[i])
                })
            
            # Sort samples by loss value (descending)
            loss_ranked_samples = sorted(sample_data, key=lambda x: x['loss'], reverse=True)
            
            # Sort samples by gradient value (descending)
            gradient_ranked_samples = sorted(sample_data, key=lambda x: x['gradient'], reverse=True)
            
            # Calculate rank correlation (Spearman)
            loss_ranks = np.array([sample['loss'] for sample in sample_data])
            gradient_ranks = np.array([sample['gradient'] for sample in sample_data])
            
            # Handle case where there's insufficient data for correlation
            if len(loss_ranks) < 2:
                spearman_corr, spearman_p = 0.0, 1.0
                pearson_corr, pearson_p = 0.0, 1.0
            else:
                try:
                    spearman_corr, spearman_p = spearmanr(loss_ranks, gradient_ranks)
                    pearson_corr, pearson_p = pearsonr(loss_ranks, gradient_ranks)
                    
                    # Handle NaN results
                    if np.isnan(spearman_corr):
                        spearman_corr = 0.0
                    if np.isnan(pearson_corr):
                        pearson_corr = 0.0
                    if np.isnan(spearman_p):
                        spearman_p = 1.0
                    if np.isnan(pearson_p):
                        pearson_p = 1.0
                        
                except Exception as e:
                    print(f"Warning: Correlation calculation failed for timestep {timestep}: {e}")
                    spearman_corr, spearman_p = 0.0, 1.0
                    pearson_corr, pearson_p = 0.0, 1.0
            
            # Find samples that are high in both loss and gradient (top 50%)
            n_samples = len(sample_data)
            top_50_percent = max(1, n_samples // 2)
            
            high_loss_samples = set(s['sample_id'] for s in loss_ranked_samples[:top_50_percent])
            high_gradient_samples = set(s['sample_id'] for s in gradient_ranked_samples[:top_50_percent])
            both_high_samples = high_loss_samples.intersection(high_gradient_samples)
            
            # Calculate overlap percentage
            overlap_percentage = len(both_high_samples) / top_50_percent * 100 if top_50_percent > 0 else 0
            
            # Analyze rank differences
            rank_differences = []
            for sample in sample_data:
                loss_rank = next(i for i, s in enumerate(loss_ranked_samples) if s['sample_id'] == sample['sample_id'])
                gradient_rank = next(i for i, s in enumerate(gradient_ranked_samples) if s['sample_id'] == sample['sample_id'])
                rank_differences.append(abs(loss_rank - gradient_rank))
            
            avg_rank_difference = np.mean(rank_differences)
            
            # Track specific samples for this timestep
            for sample_id in tracked_samples:
                # Find sample data for this timestep
                sample_idx = next((i for i, s in enumerate(sample_data) if s['sample_id'] == sample_id), None)
                if sample_idx is not None:
                    sample_loss = sample_data[sample_idx]['loss']
                    sample_gradient = sample_data[sample_idx]['gradient']
                    
                    # Find ranks
                    loss_rank = next(i for i, s in enumerate(loss_ranked_samples) if s['sample_id'] == sample_id)
                    gradient_rank = next(i for i, s in enumerate(gradient_ranked_samples) if s['sample_id'] == sample_id)
                    
                    # Store tracking data
                    sample_tracking[sample_id]['timesteps'].append(int(timestep))
                    sample_tracking[sample_id]['loss_values'].append(sample_loss)
                    sample_tracking[sample_id]['gradient_values'].append(sample_gradient)
                    sample_tracking[sample_id]['loss_ranks'].append(loss_rank + 1)  # 1-based ranking
                    sample_tracking[sample_id]['gradient_ranks'].append(gradient_rank + 1)  # 1-based ranking
                    sample_tracking[sample_id]['rank_differences'].append(abs(loss_rank - gradient_rank))
            
            correspondence_analysis[int(timestep)] = {
                'sample_data': sample_data,
                'loss_ranked_samples': loss_ranked_samples,
                'gradient_ranked_samples': gradient_ranked_samples,
                'correlations': {
                    'pearson': {'value': pearson_corr, 'p_value': pearson_p},
                    'spearman': {'value': spearman_corr, 'p_value': spearman_p}
                },
                'high_overlap_analysis': {
                    'high_loss_samples': list(high_loss_samples),
                    'high_gradient_samples': list(high_gradient_samples),
                    'both_high_samples': list(both_high_samples),
                    'overlap_percentage': overlap_percentage,
                    'top_percentage': 50
                },
                'rank_analysis': {
                    'average_rank_difference': avg_rank_difference,
                    'rank_differences': rank_differences
                },
                'statistics': {
                    'n_samples': n_samples,
                    'loss_range': [float(np.min(timestep_losses)), float(np.max(timestep_losses))],
                    'gradient_range': [float(np.min(timestep_gradients)), float(np.max(timestep_gradients))]
                }
            }
            
            print(f"Timestep {timestep}: Pearson={pearson_corr:.4f}, "
                  f"High-value overlap={overlap_percentage:.1f}%")
        
        # Add sample tracking data to the result
        correspondence_analysis['sample_tracking'] = sample_tracking
        correspondence_analysis['tracked_sample_ids'] = tracked_samples
        
        return correspondence_analysis
    
    def visualize_sample_results(self, results: Dict[str, np.ndarray], save_path: str = None) -> None:
        """
        Visualize the relationship between loss function values and gradient magnitudes across samples
        
        Args:
            results: Analysis results dictionary
            save_path: Chart save path, only display if None
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        timesteps = results['timesteps']
        losses = results['loss_values']
        gradients = results['gradient_magnitudes']
        sample_ids = results['sample_ids']
        
        unique_timesteps = np.unique(timesteps)
        unique_samples = np.unique(sample_ids)
        
        # Subplot 1: Loss values across timesteps for different samples
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_samples), 10)))
        for i, sample_id in enumerate(unique_samples[:10]):  # Show first 10 samples
            sample_mask = sample_ids == sample_id
            sample_timesteps = timesteps[sample_mask]
            sample_losses = losses[sample_mask]
            ax1.plot(sample_timesteps, sample_losses, 'o-', color=colors[i], 
                    label=f'Sample {sample_id}', alpha=0.7, markersize=4)
        
        ax1.set_xlabel('Timestep (t)')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Loss Values Across Timesteps for Different Samples')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Subplot 2: Gradient magnitudes across timesteps for different samples
        for i, sample_id in enumerate(unique_samples[:10]):  # Show first 10 samples
            sample_mask = sample_ids == sample_id
            sample_timesteps = timesteps[sample_mask]
            sample_gradients = gradients[sample_mask]
            ax2.plot(sample_timesteps, sample_gradients, 's-', color=colors[i], 
                    label=f'Sample {sample_id}', alpha=0.7, markersize=4)
        
        ax2.set_xlabel('Timestep (t)')
        ax2.set_ylabel('Gradient L2 Norm')
        ax2.set_title('Gradient Magnitudes Across Timesteps for Different Samples')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Subplot 3: Loss vs Gradient scatter plot colored by timestep
        scatter = ax3.scatter(losses, gradients, c=timesteps, cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Gradient L2 Norm')
        ax3.set_title('Loss Value vs Gradient Magnitude (Colored by Timestep)')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Timestep')
        
        # Subplot 4: Statistical analysis
        correlation = np.corrcoef(losses, gradients)[0, 1]
        
        # Calculate per-timestep statistics
        timestep_correlations = []
        for t in unique_timesteps:
            t_mask = timesteps == t
            if np.sum(t_mask) > 1:
                t_corr = np.corrcoef(losses[t_mask], gradients[t_mask])[0, 1]
                timestep_correlations.append(t_corr)
        
        avg_timestep_corr = np.mean(timestep_correlations) if timestep_correlations else 0.0
        
        ax4.text(0.5, 0.8, f'Overall Pearson Correlation: {correlation:.4f}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12, weight='bold')
        ax4.text(0.5, 0.7, f'Avg Per-Timestep Correlation: {avg_timestep_corr:.4f}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.text(0.5, 0.6, f'Total Samples: {len(unique_samples)}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.text(0.5, 0.5, f'Total Sample-Timestep Pairs: {len(losses)}', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.text(0.5, 0.4, f'Loss Range: [{losses.min():.4f}, {losses.max():.4f}]', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.text(0.5, 0.3, f'Gradient Range: [{gradients.min():.4f}, {gradients.max():.4f}]', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.text(0.5, 0.2, f'Timestep Range: [{timesteps.min()}, {timesteps.max()}]', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title('Statistical Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save or display chart
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
            
            # Save individual subplots
            self._save_individual_subplots_basic(
                (ax1, ax2, ax3, ax4), 
                save_path, 
                ['loss_across_timesteps', 'gradient_across_timesteps', 'loss_vs_gradient_scatter', 'statistical_summary']
            )
        
        plt.show()
    
    def _save_individual_subplots_basic(
        self, 
        axes: tuple, 
        original_save_path: str, 
        subplot_names: List[str]
    ) -> None:
        """
        Save individual subplots separately for basic sample analysis
        
        Args:
            axes: Tuple of matplotlib axes
            original_save_path: Original save path
            subplot_names: List of names for each subplot
        """
        base_name = os.path.splitext(os.path.basename(original_save_path))[0]
        base_dir = os.path.dirname(original_save_path)
        subfolder = os.path.join(base_dir, f"{base_name}_individual_plots")
        
        os.makedirs(subfolder, exist_ok=True)
        
        for i, (ax, name) in enumerate(zip(axes, subplot_names)):
            # Create a new figure for this subplot
            fig_individual = plt.figure(figsize=(10, 8))
            ax_new = fig_individual.add_subplot(111)
            
            # Copy the content from original axis to new axis
            # Copy lines
            for line in ax.get_lines():
                ax_new.plot(line.get_xdata(), line.get_ydata(), 
                           color=line.get_color(), linestyle=line.get_linestyle(),
                           marker=line.get_marker(), markersize=line.get_markersize(),
                           linewidth=line.get_linewidth(), label=line.get_label(),
                           alpha=line.get_alpha())
            
            # Copy scatter plots
            for collection in ax.collections:
                if hasattr(collection, 'get_offsets'):
                    offsets = collection.get_offsets()
                    if len(offsets) > 0:
                        colors = collection.get_facecolors()
                        sizes = collection.get_sizes()
                        scatter = ax_new.scatter(offsets[:, 0], offsets[:, 1], c=colors, s=sizes, alpha=0.7)
                        
                        # Add colorbar for scatter plots
                        if name == 'loss_vs_gradient_scatter':
                            # Create colorbar with proper mapping
                            # Get the original colormap data
                            if hasattr(collection, 'get_array') and collection.get_array() is not None:
                                array_data = collection.get_array()
                                norm = mcolors.Normalize(vmin=array_data.min(), vmax=array_data.max())
                                mappable = cm.ScalarMappable(norm=norm, cmap=collection.get_cmap())
                                cbar = plt.colorbar(mappable, ax=ax_new)
                                cbar.set_label('Timestep', fontsize=12)
                            else:
                                # Fallback: create a generic colorbar
                                cbar = plt.colorbar(scatter, ax=ax_new)
                                cbar.set_label('Timestep', fontsize=12)
            
            # Copy axis properties
            ax_new.set_xlabel(ax.get_xlabel(), fontsize=14)
            ax_new.set_ylabel(ax.get_ylabel(), fontsize=14)
            ax_new.set_title(ax.get_title(), fontsize=16, fontweight='bold')
            ax_new.grid(True, alpha=0.3)
            
            # Copy legend if exists
            if ax.get_legend():
                ax_new.legend(fontsize=12)
            
            # Copy text content for statistical summary
            if name == 'statistical_summary':
                for text in ax.texts:
                    ax_new.text(text.get_position()[0], text.get_position()[1], 
                               text.get_text(), transform=ax_new.transAxes,
                               fontsize=text.get_fontsize(), 
                               verticalalignment=text.get_verticalalignment(),
                               fontfamily=text.get_fontfamily())
                ax_new.axis('off')
            
            # Handle inverted y-axis
            if ax.yaxis_inverted():
                ax_new.invert_yaxis()
            
            plt.tight_layout()
            
            # Save individual subplot
            individual_path = os.path.join(subfolder, f"{name}.png")
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            plt.close(fig_individual)
            
        print(f"Individual subplots saved to: {subfolder}")
    
    def visualize_correspondence_analysis(
        self, 
        correspondence_analysis: Dict[str, Dict], 
        save_path: str = None
    ) -> None:
        """
        Visualize the loss-gradient correspondence analysis for each timestep
        
        Args:
            correspondence_analysis: Results from analyze_sample_loss_gradient_correspondence
            save_path: Path to save the visualization
        """
        # Filter out non-timestep keys and sort
        timesteps = sorted([k for k in correspondence_analysis.keys() 
                           if isinstance(k, int) or (isinstance(k, str) and k.isdigit())])
        timesteps = [int(t) for t in timesteps]  # Ensure they're integers
        n_timesteps = len(timesteps)
        
        # Create a large figure with subplots for each timestep
        fig = plt.figure(figsize=(20, 4 * ((n_timesteps + 2) // 3)))
        
        # Calculate grid layout
        n_cols = 3
        n_rows = (n_timesteps + 2) // 3
        
        for idx, timestep in enumerate(timesteps):
            data = correspondence_analysis[timestep]
            sample_data = data['sample_data']
            
            # Extract loss and gradient values
            losses = [s['loss'] for s in sample_data]
            gradients = [s['gradient'] for s in sample_data]
            sample_ids = [s['sample_id'] for s in sample_data]
            
            # Create subplot
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Scatter plot with sample IDs as annotations
            scatter = ax.scatter(losses, gradients, c=range(len(sample_ids)), 
                               cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add sample ID annotations for a few points
            for i, (loss, grad, sid) in enumerate(zip(losses, gradients, sample_ids)):
                if i % max(1, len(sample_ids) // 10) == 0:  # Annotate every 10th sample
                    ax.annotate(f'S{sid}', (loss, grad), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.8)
            
            # Set labels and title
            ax.set_xlabel('Loss Value')
            ax.set_ylabel('Gradient Magnitude')
            
            pearson_corr = data['correlations']['pearson']['value']
            overlap_pct = data['high_overlap_analysis']['overlap_percentage']
            
            ax.set_title(f'Timestep {timestep}\nPearson: {pearson_corr:.3f}, '
                        f'Top-50% Overlap: {overlap_pct:.1f}%', fontsize=10)
            
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(losses, gradients, 1)
            p = np.poly1d(z)
            ax.plot(sorted(losses), p(sorted(losses)), "r--", alpha=0.8, linewidth=1)
        
        plt.suptitle('Loss-Gradient Correspondence Analysis by Timestep', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correspondence analysis chart saved to: {save_path}")
            
            # Save individual timestep plots
            self._save_individual_timestep_plots(correspondence_analysis, save_path)
        
        plt.show()
        
        # Create a summary plot
        self._plot_correspondence_summary(correspondence_analysis, save_path)
    
    def _save_individual_timestep_plots(
        self, 
        correspondence_analysis: Dict[str, Dict], 
        original_save_path: str
    ) -> None:
        """
        Save individual timestep correspondence plots
        
        Args:
            correspondence_analysis: Results from analyze_sample_loss_gradient_correspondence
            original_save_path: Original save path
        """
        base_name = os.path.splitext(os.path.basename(original_save_path))[0]
        base_dir = os.path.dirname(original_save_path)
        subfolder = os.path.join(base_dir, f"{base_name}_individual_plots")
        
        os.makedirs(subfolder, exist_ok=True)
        
        # Filter out non-timestep keys and sort
        timesteps = sorted([k for k in correspondence_analysis.keys() 
                           if isinstance(k, int) or (isinstance(k, str) and k.isdigit())])
        timesteps = [int(t) for t in timesteps]
        
        for timestep in timesteps:
            data = correspondence_analysis[timestep]
            sample_data = data['sample_data']
            
            # Extract loss and gradient values
            losses = [s['loss'] for s in sample_data]
            gradients = [s['gradient'] for s in sample_data]
            sample_ids = [s['sample_id'] for s in sample_data]
            
            # Create individual figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Scatter plot with sample IDs as annotations
            scatter = ax.scatter(losses, gradients, c=range(len(sample_ids)), 
                               cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add sample ID annotations for a few points
            for i, (loss, grad, sid) in enumerate(zip(losses, gradients, sample_ids)):
                if i % max(1, len(sample_ids) // 10) == 0:  # Annotate every 10th sample
                    ax.annotate(f'S{sid}', (loss, grad), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.8)
            
            # Set labels and title
            ax.set_xlabel('Loss Value', fontsize=14)
            ax.set_ylabel('Gradient Magnitude', fontsize=14)
            
            pearson_corr = data['correlations']['pearson']['value']
            overlap_pct = data['high_overlap_analysis']['overlap_percentage']
            
            ax.set_title(f'Timestep {timestep} Loss-Gradient Correspondence\n'
                        f'Pearson Correlation: {pearson_corr:.3f}, Top-50% Overlap: {overlap_pct:.1f}%', 
                        fontsize=16, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(losses, gradients, 1)
            p = np.poly1d(z)
            ax.plot(sorted(losses), p(sorted(losses)), "r--", alpha=0.8, linewidth=2)
            
            # Add colorbar  
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Sample Index', fontsize=12)
            
            plt.tight_layout()
            
            # Save individual timestep plot
            individual_path = os.path.join(subfolder, f"timestep_{timestep}.png")
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Individual timestep plots saved to: {subfolder}")
    
    def visualize_sample_tracking(
        self, 
        correspondence_analysis: Dict[str, Dict], 
        save_path: str = None
    ) -> None:
        """
        Visualize the ranking trajectory of tracked samples across timesteps
        
        Args:
            correspondence_analysis: Results from analyze_sample_loss_gradient_correspondence
            save_path: Path to save the visualization
        """
        if 'sample_tracking' not in correspondence_analysis:
            print("No sample tracking data found in correspondence analysis.")
            return
            
        sample_tracking = correspondence_analysis['sample_tracking']
        tracked_sample_ids = correspondence_analysis['tracked_sample_ids']
        
        if not tracked_sample_ids:
            print("No tracked samples found.")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Define colors for different samples
        colors = plt.cm.tab10(np.linspace(0, 1, len(tracked_sample_ids)))
        
        # Plot 1: Loss ranking over timesteps
        for i, sample_id in enumerate(tracked_sample_ids):
            data = sample_tracking[sample_id]
            if data['timesteps']:
                ax1.plot(data['timesteps'], data['loss_ranks'], 'o-', 
                        color=colors[i], label=f'Sample {sample_id}', 
                        linewidth=2, markersize=6)
        
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Loss Rank (1=highest loss)')
        ax1.set_title('Loss Ranking Trajectory Across Timesteps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Lower rank (better) should be higher on the plot
        
        # Plot 2: Gradient ranking over timesteps
        for i, sample_id in enumerate(tracked_sample_ids):
            data = sample_tracking[sample_id]
            if data['timesteps']:
                ax2.plot(data['timesteps'], data['gradient_ranks'], 's-', 
                        color=colors[i], label=f'Sample {sample_id}', 
                        linewidth=2, markersize=6)
        
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Gradient Rank (1=highest gradient)')
        ax2.set_title('Gradient Ranking Trajectory Across Timesteps')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Lower rank (better) should be higher on the plot
        
        # Plot 3: Rank differences over timesteps
        for i, sample_id in enumerate(tracked_sample_ids):
            data = sample_tracking[sample_id]
            if data['timesteps']:
                ax3.plot(data['timesteps'], data['rank_differences'], 'D-', 
                        color=colors[i], label=f'Sample {sample_id}', 
                        linewidth=2, markersize=5)
        
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('|Loss Rank - Gradient Rank|')
        ax3.set_title('Ranking Difference Between Loss and Gradient')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Loss vs Gradient ranking correlation for each sample
        sample_correlations = []
        sample_labels = []
        
        for sample_id in tracked_sample_ids:
            data = sample_tracking[sample_id]
            if len(data['loss_ranks']) > 1:
                try:
                    corr_matrix = np.corrcoef(data['loss_ranks'], data['gradient_ranks'])
                    corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    sample_correlations.append(corr)
                    sample_labels.append(f'S{sample_id}')
                except Exception:
                    # Skip this sample if correlation calculation fails
                    continue
        
        if sample_correlations:
            bars = ax4.bar(range(len(sample_correlations)), sample_correlations, 
                          color=colors[:len(sample_correlations)], alpha=0.7)
            ax4.set_xlabel('Sample ID')
            ax4.set_ylabel('Loss-Gradient Rank Correlation')
            ax4.set_title('Individual Sample Rank Correlations')
            ax4.set_xticks(range(len(sample_labels)))
            ax4.set_xticklabels(sample_labels)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels on bars
            for bar, corr in zip(bars, sample_correlations):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.05 if height >= 0 else -0.1),
                        f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.suptitle('Sample Ranking Tracking Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            tracking_path = save_path.replace('.png', '_tracking.png')
            plt.savefig(tracking_path, dpi=300, bbox_inches='tight')
            print(f"Sample tracking chart saved to: {tracking_path}")
            
            # Save individual tracking subplots
            self._save_individual_subplots_tracking(
                (ax1, ax2, ax3, ax4), 
                tracking_path, 
                ['loss_ranking_trajectory', 'gradient_ranking_trajectory', 'ranking_difference', 'individual_correlations']
            )
        
        plt.show()
    
    def _save_individual_subplots_tracking(
        self, 
        axes: tuple, 
        original_save_path: str, 
        subplot_names: List[str]
    ) -> None:
        """
        Save individual subplots separately for sample tracking analysis
        
        Args:
            axes: Tuple of matplotlib axes
            original_save_path: Original save path
            subplot_names: List of names for each subplot
        """
        base_name = os.path.splitext(os.path.basename(original_save_path))[0]
        base_dir = os.path.dirname(original_save_path)
        subfolder = os.path.join(base_dir, f"{base_name}_individual_plots")
        
        os.makedirs(subfolder, exist_ok=True)
        
        for i, (ax, name) in enumerate(zip(axes, subplot_names)):
            # Create a new figure for this subplot
            fig_individual = plt.figure(figsize=(12, 8))
            ax_new = fig_individual.add_subplot(111)
            
            # Copy the content from original axis to new axis
            # Copy lines
            for line in ax.get_lines():
                ax_new.plot(line.get_xdata(), line.get_ydata(), 
                           color=line.get_color(), linestyle=line.get_linestyle(),
                           marker=line.get_marker(), markersize=line.get_markersize(),
                           linewidth=line.get_linewidth(), label=line.get_label(),
                           alpha=line.get_alpha())
            
            # Copy bar plots
            for patch in ax.patches:
                if hasattr(patch, 'get_height'):
                    ax_new.bar(patch.get_x() + patch.get_width()/2, patch.get_height(),
                              width=patch.get_width(), color=patch.get_facecolor(),
                              alpha=patch.get_alpha(), edgecolor=patch.get_edgecolor())
            
            # Copy axis properties
            ax_new.set_xlabel(ax.get_xlabel(), fontsize=14)
            ax_new.set_ylabel(ax.get_ylabel(), fontsize=14)
            ax_new.set_title(ax.get_title(), fontsize=16, fontweight='bold')
            ax_new.grid(True, alpha=0.3)
            
            # Copy legend if exists
            if ax.get_legend():
                ax_new.legend(fontsize=12)
            
            # Copy x-tick labels for bar plots
            if len(ax.get_xticklabels()) > 0:
                ax_new.set_xticks(ax.get_xticks())
                ax_new.set_xticklabels([label.get_text() for label in ax.get_xticklabels()])
            
            # Copy text annotations
            for text in ax.texts:
                ax_new.text(text.get_position()[0], text.get_position()[1], 
                           text.get_text(), transform=ax_new.transData,
                           fontsize=text.get_fontsize(), 
                           ha=text.get_horizontalalignment(),
                           va=text.get_verticalalignment())
            
            # Handle inverted y-axis
            if ax.yaxis_inverted():
                ax_new.invert_yaxis()
            
            # Add horizontal line at y=0 for correlation plot
            if name == 'individual_correlations':
                ax_new.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save individual subplot
            individual_path = os.path.join(subfolder, f"{name}.png")
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            plt.close(fig_individual)
            
        print(f"Individual tracking subplots saved to: {subfolder}")
    
    def _plot_correspondence_summary(
        self, 
        correspondence_analysis: Dict[str, Dict], 
        save_path: str = None
    ) -> None:
        """
        Plot summary statistics of correspondence analysis across timesteps
        """
        # Filter out non-timestep keys and sort
        timesteps = sorted([k for k in correspondence_analysis.keys() 
                           if isinstance(k, int) or (isinstance(k, str) and k.isdigit())])
        timesteps = [int(t) for t in timesteps]  # Ensure they're integers
        
        # Extract summary statistics
        pearson_corrs = []
        spearman_corrs = []
        overlap_percentages = []
        avg_rank_diffs = []
        
        for timestep in timesteps:
            data = correspondence_analysis[timestep]
            pearson_corrs.append(data['correlations']['pearson']['value'])
            spearman_corrs.append(data['correlations']['spearman']['value'])
            overlap_percentages.append(data['high_overlap_analysis']['overlap_percentage'])
            avg_rank_diffs.append(data['rank_analysis']['average_rank_difference'])
        
        # Create summary plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Correlation coefficients across timesteps
        ax1.plot(timesteps, pearson_corrs, 'b-o', label='Pearson', linewidth=2, markersize=6)
        ax1.plot(timesteps, spearman_corrs, 'r-s', label='Spearman', linewidth=2, markersize=6)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.set_title('Loss-Gradient Correlations Across Timesteps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 2: High-value overlap percentage
        bars = ax2.bar(timesteps, overlap_percentages, alpha=0.7, color='green')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Overlap Percentage (%)')
        ax2.set_title('Top-50% Loss-Gradient Overlap Across Timesteps')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, pct in zip(bars, overlap_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Average rank differences
        ax3.plot(timesteps, avg_rank_diffs, 'purple', marker='D', linewidth=2, markersize=6)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Average Rank Difference')
        ax3.set_title('Average Rank Difference Between Loss and Gradient Rankings')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics text
        overall_pearson = np.mean(pearson_corrs)
        overall_overlap = np.mean(overlap_percentages)
        overall_rank_diff = np.mean(avg_rank_diffs)
        
        # Count timesteps with strong correlation (>0.5)
        strong_corr_count = sum(1 for corr in pearson_corrs if corr > 0.5)
        strong_corr_pct = strong_corr_count / len(timesteps) * 100
        
        # Count timesteps with high overlap (>70% for top-50% analysis)
        high_overlap_count = sum(1 for overlap in overlap_percentages if overlap > 70)
        high_overlap_pct = high_overlap_count / len(timesteps) * 100
        
        summary_text = f"""Summary Statistics:

Overall Analysis:
• Average Pearson Correlation: {overall_pearson:.4f}
• Average Top-50% Overlap: {overall_overlap:.1f}%
• Average Rank Difference: {overall_rank_diff:.2f}

Validation Results:
• Timesteps with strong correlation (>0.5): {strong_corr_count}/{len(timesteps)} ({strong_corr_pct:.1f}%)
• Timesteps with high overlap (>70%): {high_overlap_count}/{len(timesteps)} ({high_overlap_pct:.1f}%)

Interpretation:
• Higher correlation → Loss and gradient are more aligned
• Higher overlap → Samples with high loss tend to have high gradients
• Lower rank difference → Consistent ranking between loss and gradient"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Overall Summary')
        ax4.axis('off')
        
        plt.suptitle('Loss-Gradient Correspondence Summary', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            summary_path = save_path.replace('.png', '_summary.png')
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            print(f"Summary chart saved to: {summary_path}")
            
            # Save individual summary subplots
            self._save_individual_subplots_summary(
                (ax1, ax2, ax3, ax4), 
                summary_path, 
                ['correlations_across_timesteps', 'overlap_across_timesteps', 'rank_differences', 'summary_statistics']
            )
        
        plt.show()
    
    def _save_individual_subplots_summary(
        self, 
        axes: tuple, 
        original_save_path: str, 
        subplot_names: List[str]
    ) -> None:
        """
        Save individual subplots separately for correspondence summary
        
        Args:
            axes: Tuple of matplotlib axes
            original_save_path: Original save path
            subplot_names: List of names for each subplot
        """
        base_name = os.path.splitext(os.path.basename(original_save_path))[0]
        base_dir = os.path.dirname(original_save_path)
        subfolder = os.path.join(base_dir, f"{base_name}_individual_plots")
        
        os.makedirs(subfolder, exist_ok=True)
        
        for i, (ax, name) in enumerate(zip(axes, subplot_names)):
            # Create a new figure for this subplot
            fig_individual = plt.figure(figsize=(12, 8))
            ax_new = fig_individual.add_subplot(111)
            
            # Copy the content from original axis to new axis
            # Copy lines
            for line in ax.get_lines():
                ax_new.plot(line.get_xdata(), line.get_ydata(), 
                           color=line.get_color(), linestyle=line.get_linestyle(),
                           marker=line.get_marker(), markersize=line.get_markersize(),
                           linewidth=line.get_linewidth(), label=line.get_label(),
                           alpha=line.get_alpha())
            
            # Copy bar plots
            for patch in ax.patches:
                if hasattr(patch, 'get_height'):
                    x_pos = patch.get_x() + patch.get_width()/2
                    height = patch.get_height()
                    ax_new.bar(x_pos, height, width=patch.get_width(), 
                              color=patch.get_facecolor(), alpha=patch.get_alpha(),
                              edgecolor=patch.get_edgecolor())
                    
                    # Copy text labels on bars
                    if name == 'overlap_across_timesteps':
                        ax_new.text(x_pos, height + height*0.01, f'{height:.1f}%',
                                   ha='center', va='bottom', fontsize=10)
            
            # Copy axis properties
            ax_new.set_xlabel(ax.get_xlabel(), fontsize=14)
            ax_new.set_ylabel(ax.get_ylabel(), fontsize=14)
            ax_new.set_title(ax.get_title(), fontsize=16, fontweight='bold')
            ax_new.grid(True, alpha=0.3)
            
            # Copy legend if exists
            if ax.get_legend():
                ax_new.legend(fontsize=12)
            
            # Copy text content for summary statistics
            if name == 'summary_statistics':
                for text in ax.texts:
                    ax_new.text(text.get_position()[0], text.get_position()[1], 
                               text.get_text(), transform=ax_new.transAxes,
                               fontsize=text.get_fontsize(), 
                               verticalalignment=text.get_verticalalignment(),
                               fontfamily=text.get_fontfamily())
                ax_new.axis('off')
            
            # Add horizontal reference lines
            if name == 'correlations_across_timesteps':
                ax_new.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            
            # Save individual subplot
            individual_path = os.path.join(subfolder, f"{name}.png")
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            plt.close(fig_individual)
            
        print(f"Individual summary subplots saved to: {subfolder}")
    
    def generate_sample_report(self, results: Dict[str, np.ndarray]) -> str:
        """
        Generate analysis report for sample-based loss-gradient relationship
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Formatted analysis report string
        """
        timesteps = results['timesteps']
        losses = results['loss_values']
        gradients = results['gradient_magnitudes']
        sample_ids = results['sample_ids']
        
        unique_timesteps = np.unique(timesteps)
        unique_samples = np.unique(sample_ids)
        
        # Calculate statistical metrics
        overall_correlation = np.corrcoef(losses, gradients)[0, 1]
        loss_std = np.std(losses)
        gradient_std = np.std(gradients)
        
        # Calculate per-timestep correlations
        timestep_correlations = []
        for t in unique_timesteps:
            t_mask = timesteps == t
            if np.sum(t_mask) > 1:
                t_corr = np.corrcoef(losses[t_mask], gradients[t_mask])[0, 1]
                timestep_correlations.append((t, t_corr))
        
        avg_timestep_corr = np.mean([corr for _, corr in timestep_correlations]) if timestep_correlations else 0.0
        
        # Calculate per-sample statistics
        sample_stats = []
        for sample_id in unique_samples:
            s_mask = sample_ids == sample_id
            s_losses = losses[s_mask]
            s_gradients = gradients[s_mask]
            if len(s_losses) > 1:
                s_corr = np.corrcoef(s_losses, s_gradients)[0, 1]
                sample_stats.append({
                    'id': sample_id,
                    'correlation': s_corr,
                    'avg_loss': np.mean(s_losses),
                    'avg_gradient': np.mean(s_gradients)
                })
        
        # Generate report
        correlation_strength = (
            'Strong Positive' if overall_correlation > 0.7 else
            'Moderate Positive' if overall_correlation > 0.3 else
            'Weak' if abs(overall_correlation) <= 0.3 else
            'Negative'
        )
        
        report = f"""
=== Sample-based Loss Function Value and Gradient Magnitude Relationship Analysis Report ===

1. Basic Statistical Information:
   - Total samples analyzed: {len(unique_samples)}
   - Total timesteps analyzed: {len(unique_timesteps)}
   - Total sample-timestep pairs: {len(losses)}
   - Timestep range: {timesteps.min()} - {timesteps.max()}
   
2. Loss Function Value Statistics:
   - Mean: {losses.mean():.6f}
   - Standard deviation: {loss_std:.6f}
   - Minimum: {losses.min():.6f}
   - Maximum: {losses.max():.6f}
   
3. Gradient Magnitude Statistics:
   - Mean: {gradients.mean():.6f}
   - Standard deviation: {gradient_std:.6f}
   - Minimum: {gradients.min():.6f}
   - Maximum: {gradients.max():.6f}
   
4. Correlation Analysis:
   - Overall Pearson correlation: {overall_correlation:.6f}
   - Correlation strength: {correlation_strength}
   - Average per-timestep correlation: {avg_timestep_corr:.6f}
   
5. Per-Timestep Correlation Details:
"""
        
        # Add per-timestep correlation details
        for t, corr in timestep_correlations:
            report += f"   Timestep {t:4d}: Correlation = {corr:.6f}\n"
        
        report += f"\n6. Sample Statistics Summary:\n"
        report += f"   - Samples with valid correlations: {len(sample_stats)}\n"
        
        if sample_stats:
            avg_sample_corr = np.mean([s['correlation'] for s in sample_stats])
            report += f"   - Average per-sample correlation: {avg_sample_corr:.6f}\n"
            
            # Show top 5 samples with highest correlations
            sorted_samples = sorted(sample_stats, key=lambda x: x['correlation'], reverse=True)
            report += f"\n   Top 5 samples with highest loss-gradient correlations:\n"
            for i, sample in enumerate(sorted_samples[:5]):
                report += f"   {i+1}. Sample {sample['id']}: Corr={sample['correlation']:.4f}, "
                report += f"Avg Loss={sample['avg_loss']:.4f}, Avg Gradient={sample['avg_gradient']:.4f}\n"
        
        return report
    
    def generate_correspondence_report(
        self, 
        correspondence_analysis: Dict[str, Dict]
    ) -> str:
        """
        Generate detailed correspondence analysis report
        
        Args:
            correspondence_analysis: Results from analyze_sample_loss_gradient_correspondence
            
        Returns:
            Formatted correspondence analysis report
        """
        # Filter out non-timestep keys and sort
        timesteps = sorted([k for k in correspondence_analysis.keys() 
                           if isinstance(k, int) or (isinstance(k, str) and k.isdigit())])
        timesteps = [int(t) for t in timesteps]  # Ensure they're integers
        
        # Calculate overall statistics
        all_pearson_corrs = []
        all_overlap_percentages = []
        all_rank_diffs = []
        
        for timestep in timesteps:
            data = correspondence_analysis[timestep]
            all_pearson_corrs.append(data['correlations']['pearson']['value'])
            all_overlap_percentages.append(data['high_overlap_analysis']['overlap_percentage'])
            all_rank_diffs.append(data['rank_analysis']['average_rank_difference'])
        
        overall_pearson = np.mean(all_pearson_corrs)
        overall_overlap = np.mean(all_overlap_percentages)
        overall_rank_diff = np.mean(all_rank_diffs)
        
        # Validation statistics
        strong_corr_count = sum(1 for corr in all_pearson_corrs if corr > 0.5)
        moderate_corr_count = sum(1 for corr in all_pearson_corrs if 0.3 <= corr <= 0.5)
        weak_corr_count = sum(1 for corr in all_pearson_corrs if corr < 0.3)
        
        high_overlap_count = sum(1 for overlap in all_overlap_percentages if overlap > 70)
        moderate_overlap_count = sum(1 for overlap in all_overlap_percentages if 50 <= overlap <= 70)
        low_overlap_count = sum(1 for overlap in all_overlap_percentages if overlap < 50)
        
        # Generate report
        report = f"""
=== Loss-Gradient Correspondence Analysis Report ===

EXECUTIVE SUMMARY:
The analysis examines whether samples with higher loss values consistently exhibit higher gradient magnitudes
across different timesteps. This helps validate the hypothesis that "loss magnitude correlates with gradient magnitude."

OVERALL VALIDATION RESULTS:
• Average Pearson Correlation: {overall_pearson:.4f}
• Average High-Value Overlap (Top 50%): {overall_overlap:.1f}%
• Average Rank Difference: {overall_rank_diff:.2f}

HYPOTHESIS VALIDATION:
• Strong evidence (correlation > 0.5): {strong_corr_count}/{len(timesteps)} timesteps ({strong_corr_count/len(timesteps)*100:.1f}%)
• Moderate evidence (correlation 0.3-0.5): {moderate_corr_count}/{len(timesteps)} timesteps ({moderate_corr_count/len(timesteps)*100:.1f}%)
• Weak evidence (correlation < 0.3): {weak_corr_count}/{len(timesteps)} timesteps ({weak_corr_count/len(timesteps)*100:.1f}%)

HIGH-VALUE OVERLAP ANALYSIS:
• High overlap (>70%): {high_overlap_count}/{len(timesteps)} timesteps ({high_overlap_count/len(timesteps)*100:.1f}%)
• Moderate overlap (50-70%): {moderate_overlap_count}/{len(timesteps)} timesteps ({moderate_overlap_count/len(timesteps)*100:.1f}%)
• Low overlap (<50%): {low_overlap_count}/{len(timesteps)} timesteps ({low_overlap_count/len(timesteps)*100:.1f}%)

DETAILED TIMESTEP ANALYSIS:
"""
        
        # Add detailed analysis for each timestep
        for timestep in timesteps:
            data = correspondence_analysis[timestep]
            
            pearson_corr = data['correlations']['pearson']['value']
            pearson_p = data['correlations']['pearson']['p_value']
            spearman_corr = data['correlations']['spearman']['value']
            overlap_pct = data['high_overlap_analysis']['overlap_percentage']
            rank_diff = data['rank_analysis']['average_rank_difference']
            n_samples = data['statistics']['n_samples']
            
            # Determine correlation strength
            if pearson_corr > 0.7:
                strength = "Very Strong"
            elif pearson_corr > 0.5:
                strength = "Strong"
            elif pearson_corr > 0.3:
                strength = "Moderate"
            elif pearson_corr > 0.1:
                strength = "Weak"
            else:
                strength = "Very Weak"
            
            # Statistical significance
            significance = "Significant" if pearson_p < 0.05 else "Not Significant"
            
            # Top high-loss and high-gradient samples
            high_loss_samples = data['high_overlap_analysis']['high_loss_samples'][:3]
            high_gradient_samples = data['high_overlap_analysis']['high_gradient_samples'][:3]
            both_high_samples = data['high_overlap_analysis']['both_high_samples']
            
            report += f"""
Timestep {timestep}:
  • Samples analyzed: {n_samples}
  • Pearson correlation: {pearson_corr:.4f} ({strength}, {significance}, p={pearson_p:.4f})
  • Spearman correlation: {spearman_corr:.4f}
  • Top-50% overlap: {overlap_pct:.1f}% ({len(both_high_samples)} samples in both high-loss and high-gradient groups)
  • Average rank difference: {rank_diff:.2f}
  • Top high-loss samples: {high_loss_samples}
  • Top high-gradient samples: {high_gradient_samples}
  • Samples high in both: {both_high_samples}
"""
        
        # Add interpretation
        report += f"""

INTERPRETATION AND CONCLUSIONS:

1. Overall Relationship Strength:
   {'Strong' if overall_pearson > 0.5 else 'Moderate' if overall_pearson > 0.3 else 'Weak'} overall correlation ({overall_pearson:.4f}) suggests that 
   {"the hypothesis is well-supported" if overall_pearson > 0.5 else "there is moderate support for the hypothesis" if overall_pearson > 0.3 else "the hypothesis has limited support"}.

2. Consistency Across Timesteps:
   {strong_corr_count}/{len(timesteps)} timesteps show strong correlation, indicating 
   {"high consistency" if strong_corr_count/len(timesteps) > 0.7 else "moderate consistency" if strong_corr_count/len(timesteps) > 0.4 else "low consistency"}.

3. High-Value Sample Overlap:
   Average {overall_overlap:.1f}% overlap means that samples with high loss 
   {"frequently" if overall_overlap > 80 else "sometimes" if overall_overlap > 60 else "rarely"} also have high gradients.

4. Validation Result:
   {"✓ HYPOTHESIS CONFIRMED" if overall_pearson > 0.5 and overall_overlap > 70 else "△ HYPOTHESIS PARTIALLY SUPPORTED" if overall_pearson > 0.3 or overall_overlap > 60 else "✗ HYPOTHESIS NOT WELL SUPPORTED"}
   
   The analysis {"strongly supports" if overall_pearson > 0.5 and overall_overlap > 70 else "moderately supports" if overall_pearson > 0.3 or overall_overlap > 60 else "does not strongly support"} 
   the hypothesis that samples with higher loss values tend to have higher gradient magnitudes.

RECOMMENDATIONS:
• {'Focus training on high-loss samples as they contribute most to gradients' if overall_pearson > 0.5 else 'Consider sample-specific training strategies' if overall_pearson > 0.3 else 'Investigate other factors affecting gradient magnitudes'}
• {'The relationship is consistent across timesteps' if np.std(all_pearson_corrs) < 0.2 else 'The relationship varies significantly across timesteps - investigate timestep-specific patterns'}
"""
        
        # Add sample tracking analysis if available
        if 'sample_tracking' in correspondence_analysis:
            report += self._generate_sample_tracking_section(correspondence_analysis)
        
        return report
    
    def _generate_sample_tracking_section(self, correspondence_analysis: Dict[str, Dict]) -> str:
        """
        Generate the sample tracking section of the report
        
        Args:
            correspondence_analysis: Results containing sample tracking data
            
        Returns:
            Formatted sample tracking section
        """
        sample_tracking = correspondence_analysis['sample_tracking']
        tracked_sample_ids = correspondence_analysis['tracked_sample_ids']
        
        if not tracked_sample_ids:
            return "\n\nSAMPLE TRACKING ANALYSIS:\nNo samples were tracked.\n"
        
        tracking_section = f"""

=== INDIVIDUAL SAMPLE TRACKING ANALYSIS ===

TRACKED SAMPLES: {tracked_sample_ids}

This section analyzes how specific samples' loss and gradient rankings change across timesteps,
providing insights into individual sample behavior patterns.

SAMPLE-SPECIFIC ANALYSIS:
"""
        
        for sample_id in tracked_sample_ids:
            data = sample_tracking[sample_id]
            
            if not data['timesteps']:
                tracking_section += f"\nSample {sample_id}: No data available\n"
                continue
            
            # Calculate statistics for this sample
            avg_loss_rank = np.mean(data['loss_ranks'])
            avg_gradient_rank = np.mean(data['gradient_ranks'])
            avg_rank_diff = np.mean(data['rank_differences'])
            
            loss_rank_std = np.std(data['loss_ranks'])
            gradient_rank_std = np.std(data['gradient_ranks'])
            
            # Calculate rank correlation for this sample
            if len(data['loss_ranks']) > 1:
                try:
                    corr_matrix = np.corrcoef(data['loss_ranks'], data['gradient_ranks'])
                    rank_corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                except Exception:
                    rank_corr = 0.0
            else:
                rank_corr = 0.0
            
            # Determine consistency
            loss_consistency = "High" if loss_rank_std < 2 else "Medium" if loss_rank_std < 5 else "Low"
            gradient_consistency = "High" if gradient_rank_std < 2 else "Medium" if gradient_rank_std < 5 else "Low"
            
            # Determine overall ranking level
            timestep_keys = [t for t in correspondence_analysis.keys() if isinstance(t, int)]
            if timestep_keys:
                first_timestep = timestep_keys[0]
                total_samples = len(correspondence_analysis[first_timestep]['sample_data'])
            else:
                total_samples = 10  # fallback
            
            if avg_loss_rank <= total_samples * 0.2:
                loss_level = "Consistently High Loss"
            elif avg_loss_rank <= total_samples * 0.5:
                loss_level = "Medium Loss"
            else:
                loss_level = "Consistently Low Loss"
                
            if avg_gradient_rank <= total_samples * 0.2:
                gradient_level = "Consistently High Gradient"
            elif avg_gradient_rank <= total_samples * 0.5:
                gradient_level = "Medium Gradient"
            else:
                gradient_level = "Consistently Low Gradient"
            
            tracking_section += f"""
Sample {sample_id} Analysis:
  • Timesteps analyzed: {len(data['timesteps'])}
  • Average loss rank: {avg_loss_rank:.2f} ({loss_level})
  • Average gradient rank: {avg_gradient_rank:.2f} ({gradient_level})
  • Average rank difference: {avg_rank_diff:.2f}
  • Loss rank consistency: {loss_consistency} (std: {loss_rank_std:.2f})
  • Gradient rank consistency: {gradient_consistency} (std: {gradient_rank_std:.2f})
  • Loss-gradient rank correlation: {rank_corr:.4f}
  
  Detailed trajectory:"""
            
            for i, timestep in enumerate(data['timesteps']):
                tracking_section += f"""
    Timestep {timestep}: Loss rank {data['loss_ranks'][i]}, Gradient rank {data['gradient_ranks'][i]}, Difference {data['rank_differences'][i]}"""
        
        # Add cross-sample comparison
        tracking_section += f"""

CROSS-SAMPLE COMPARISON:

Sample Ranking Correlations:"""
        
        for sample_id in tracked_sample_ids:
            data = sample_tracking[sample_id]
            if len(data['loss_ranks']) > 1:
                try:
                    corr_matrix = np.corrcoef(data['loss_ranks'], data['gradient_ranks'])
                    rank_corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                except Exception:
                    rank_corr = 0.0
                corr_strength = "Strong" if abs(rank_corr) > 0.7 else "Moderate" if abs(rank_corr) > 0.4 else "Weak"
                tracking_section += f"""
  • Sample {sample_id}: {rank_corr:.4f} ({corr_strength} {'positive' if rank_corr > 0 else 'negative'} correlation)"""
        
        # Calculate average metrics across tracked samples
        all_rank_corrs = []
        all_avg_rank_diffs = []
        
        for sample_id in tracked_sample_ids:
            data = sample_tracking[sample_id]
            if len(data['loss_ranks']) > 1:
                try:
                    corr_matrix = np.corrcoef(data['loss_ranks'], data['gradient_ranks'])
                    rank_corr = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
                    all_rank_corrs.append(rank_corr)
                    all_avg_rank_diffs.append(np.mean(data['rank_differences']))
                except Exception:
                    # Skip this sample if correlation calculation fails
                    continue
        
        if all_rank_corrs:
            overall_avg_corr = np.mean(all_rank_corrs)
            overall_avg_diff = np.mean(all_avg_rank_diffs)
            
            tracking_section += f"""

TRACKING SUMMARY:
• Average individual sample correlation: {overall_avg_corr:.4f}
• Average individual rank difference: {overall_avg_diff:.2f}
• Samples with strong positive correlation (>0.5): {sum(1 for c in all_rank_corrs if c > 0.5)}/{len(all_rank_corrs)}
• Samples with consistent ranking (avg diff <3): {sum(1 for d in all_avg_rank_diffs if d < 3)}/{len(all_avg_rank_diffs)}

INDIVIDUAL SAMPLE INSIGHTS:
• Samples showing consistent loss-gradient alignment across timesteps demonstrate stable predictive patterns
• Large ranking differences suggest sample-specific factors affecting the loss-gradient relationship
• Strong individual correlations support the overall hypothesis at the sample level
"""
        
        return tracking_section


def main():
    """
    Main function: Parse command line arguments and execute analysis
    """
    parser = argparse.ArgumentParser(description="Sample-based Loss Function Value and Gradient Magnitude Relationship Analyzer")
    
    # Required parameters
    parser.add_argument(
        "--weight_path", 
        type=str, 
        required=True,
        help="Path to model weight file"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="/root/dataset/imagenet/train",
        help="Path to training dataset"
    )
    
    # Optional parameters
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="config/DiT-S.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        choices=[256, 512], 
        default=256,
        help="Input image size"
    )
    parser.add_argument(
        "--num_classes", 
        type=int, 
        default=1,
        help="Number of classification classes"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=50,
        help="Number of samples to analyze per timestep"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./analysis_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set global random seed at the beginning
    set_random_seeds(args.random_seed)
    
    # Initialize analyzer
    print("Initializing sample-based loss-gradient relationship analyzer...")
    analyzer = LossGradientAnalyzer(
        model_config=args.model_config,
        image_size=args.image_size,
        num_classes=args.num_classes,
        data_path=args.data_path,
        data_samples=args.num_samples,
        random_seed=args.random_seed
    )
    
    # Execute analysis
    print("Starting analysis of loss function values and gradient magnitudes across samples...")
    results = analyzer.analyze_sample_loss_gradient_relationship(
        weight_path=args.weight_path,
        num_samples_per_timestep=args.num_samples
    )
    
    # Execute correspondence analysis
    print("Analyzing loss-gradient correspondence for each timestep...")
    correspondence_analysis = analyzer.analyze_sample_loss_gradient_correspondence(results, tracked_samples=None)
    
    # Generate visualization charts
    print("Generating visualization charts...")
    plot_path = os.path.join(args.output_dir, "sample_loss_gradient_analysis.png")
    analyzer.visualize_sample_results(results, save_path=plot_path)
    
    # Generate correspondence visualization
    print("Generating correspondence analysis visualization...")
    correspondence_plot_path = os.path.join(args.output_dir, "correspondence_analysis.png")
    analyzer.visualize_correspondence_analysis(correspondence_analysis, save_path=correspondence_plot_path)
    
    # Generate sample tracking visualization
    print("Generating sample tracking visualization...")
    tracking_plot_path = os.path.join(args.output_dir, "sample_tracking.png")
    analyzer.visualize_sample_tracking(correspondence_analysis, save_path=tracking_plot_path)
    
    # Generate analysis reports
    print("Generating analysis reports...")
    
    # Basic sample analysis report
    basic_report = analyzer.generate_sample_report(results)
    basic_report_path = os.path.join(args.output_dir, "sample_analysis_report.txt")
    with open(basic_report_path, 'w', encoding='utf-8') as f:
        f.write(basic_report)
    
    # Detailed correspondence analysis report
    correspondence_report = analyzer.generate_correspondence_report(correspondence_analysis)
    correspondence_report_path = os.path.join(args.output_dir, "correspondence_analysis_report.txt")
    with open(correspondence_report_path, 'w', encoding='utf-8') as f:
        f.write(correspondence_report)
    
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("\n=== BASIC ANALYSIS SUMMARY ===")
    print(basic_report[:1000] + "..." if len(basic_report) > 1000 else basic_report)
    print("\n=== CORRESPONDENCE ANALYSIS SUMMARY ===")
    print(correspondence_report[:1500] + "..." if len(correspondence_report) > 1500 else correspondence_report)


if __name__ == "__main__":
    main()

'''
Example usage:
python loss_gradient_analyzer.py \
    --weight_path /path/to/weights.pth \
    --data_path /home/user/code/burn/BiDVL/data/celeba_hq/celeba-hq/celeba_hq_256 \
    --num_samples 50 \
    --output_dir ./sample_analysis_results \
    --random_seed 42

This script will generate:
1. sample_loss_gradient_analysis.png - Basic sample analysis visualization
2. correspondence_analysis.png - Detailed timestep-wise correspondence analysis
3. correspondence_analysis_summary.png - Summary statistics across timesteps
4. sample_tracking.png - Individual sample ranking trajectory tracking
5. sample_analysis_report.txt - Basic analysis report
6. correspondence_analysis_report.txt - Detailed correspondence validation report with sample tracking

The analysis includes:
- Overall loss-gradient correlation validation across all samples
- Individual sample tracking showing how specific samples' loss and gradient rankings change across timesteps
- Cross-sample comparison to identify consistent vs inconsistent patterns
- Comprehensive validation of whether samples with higher loss values consistently exhibit higher gradient magnitudes
'''