import jax
import jax.numpy as jnp

class CrystalDiffusion:
    def __init__(self, schedule="cosine", min_signal_ratio=1e-4):
        self.schedule = schedule
        self.min_signal_ratio = min_signal_ratio

    def _get_alpha_bar(self, t):
        if self.schedule == "linear":
            return 1.0 - t * (1.0 - self.min_signal_ratio)
        elif self.schedule == "cosine":
            s = 0.008
            f_t = jnp.cos((t + s) / (1.0 + s) * jnp.pi / 2.0) ** 2
            f_0 = jnp.cos(s / (1.0 + s) * jnp.pi / 2.0) ** 2
            alpha_bar = f_t / f_0
            return jnp.clip(alpha_bar, self.min_signal_ratio, 1.0)

    def add_noise(self, clean_crystal, key, t):
        # CORRECCIÓN 1: quitamos el 'key' de aquí
        alpha_bar = self._get_alpha_bar(t) 
        
        noise = jax.random.normal(key, shape=clean_crystal.shape)
        guided_noise = noise
        
        noisy_crystal = jnp.sqrt(alpha_bar) * clean_crystal + jnp.sqrt(1.0 - alpha_bar) * guided_noise
        return noisy_crystal, guided_noise

    def predict_crystal(self, noisy_crystal, predicted_noise, t):
        alpha_bar = self._get_alpha_bar(t)
        pred_clean_crystal = (noisy_crystal - jnp.sqrt(1.0 - alpha_bar) * predicted_noise) / jnp.sqrt(alpha_bar)
        return jnp.clip(pred_clean_crystal, 0.0, 1.0)