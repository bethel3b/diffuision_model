from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np   

class ForwardProcess:
    """Forward diffusion process for adding noise to images in a diffusion model."""
    def __init__(self, image, timesteps, betas):
        """ Initialize the forward diffusion process.
        Args:
            image (torch.Tensor): Input image tensor to apply diffusion process to 
            timesteps (int): Total number of diffusion timesteps
            betas (torch.Tensor): Noise schedule - variance values for each timestep
        """
        self.image = image 
        self.timesteps = timesteps
        self.betas = betas 

    def get_epislon(self):
        """Generate random noise tensor with same shape as input image.

        Returns:
            torch.Tensor: Random noise sampled from standard normal distribution
        """
        epislon = torch.randn_like(self.image)
        return epislon 

    def direct_sampling(self, t):
        """ Directly sample x_t at any timestep using closed-form solution:
        x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        
        This avoids the need to iterate through all previous steps by using
        the cumulative product of alphas (ᾱ_t) for efficient sampling.
        
        Args:
            t (int): Target timestep to sample at
            
        Returns:
            torch.Tensor: Noisy image at timestep t
        """

        # Base case: return original image at t=0
        if t == 0:
            x0 = self.image
            return x0
 
        # Step 1: Compute α_t = 1 - β_t (signal retention factor)
        alphas = 1 - self.betas 

        # Step 2: Compute ᾱ_t = ∏_{s=1}^t α_s (cumulative signal retention)
        alpha_cum_prod = torch.cumprod(alphas, dim=0)

        # Step 3: Scale original image by √ᾱ_t - remaining signal strength
        sqrt_alpha_cum_product_t = torch.sqrt(alpha_cum_prod[t-1]).view(-1, 1, 1)    
        remaining_image = sqrt_alpha_cum_product_t * self.image

        # Step 4: Add noise scaled by √(1-ᾱ_t) - cumulative noise level
        one_minus_alpha_cum = torch.sqrt(1-alpha_cum_prod[t-1]).view(-1, 1, 1)
        noise = one_minus_alpha_cum * self.get_epislon()

        # Step 5: Combine signal and noise components
        xt = remaining_image + noise
        return xt
    
    def get_schedule(self):
        """ Get the noise schedule parameters used in the diffusion process.
        
        Returns:
            dict: Dictionary containing beta and alpha schedules
                - "betas": noise variance schedule β_t
                - "alphas": signal retention schedule α_t = 1 - β_t
        """
        schedules = {
            "betas": self.betas,           # Noise variance at each step
            "alphas": 1 - self.betas       # Signal retention at each step
        }
        return schedules
