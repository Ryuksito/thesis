import os
import jax
import jax.numpy as jnp
from tensorneat.common import State
from .models.diffusion import CrystalDiffusion

class CrystalGenerator:
    def __init__(self, model_network, weights_path: str, seed: int = 42):
        """
        Carga el cerebro entrenado y prepara el motor de inferencia.
        """
        self.model = model_network
        self.diffusion = CrystalDiffusion()
        self.key = jax.random.PRNGKey(seed)
        
        # 1. Cargar los pesos evolutivos (Nodos y Conexiones)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No se encontraron pesos en {weights_path}")
            
        data = jnp.load(weights_path)
        self.best_nodes = data['nodes']
        self.best_conns = data['conns']
        
        # 2. Inicializar el estado de TensorNEAT
        self.state = State(randkey=self.key)
        self.state = self.model.neat.setup(self.state)
        
    def generate(self, elements, embeddings, props, num_steps=50):
        """
        Genera un cristal a partir de ruido puro condicionándolo a un contexto.
        """
        # 1. EL LIENZO EN BLANCO: Ruido Gaussiano Puro (t = 1.0)
        self.key, key_lat, key_atom = jax.random.split(self.key, 3)
        
        # OJO: Asumimos que quieres generar 4 átomos por ahora
        current_lattice = jax.random.normal(key_lat, shape=(6,))
        current_atoms = jax.random.normal(key_atom, shape=(4, 4))
        
        # 2. LA CUENTA REGRESIVA (De t=1.0 bajando hasta t=0.0)
        # Usamos pasos espaciados uniformemente
        times = jnp.linspace(1.0, 0.0, num_steps)
        
        print("🌌 Iniciando creación de materia desde el espacio latente...")
        
        for i in range(num_steps - 1):
            t_now = times[i]
            t_next = times[i+1]
            
            # A) El Cerebro predice el ruido presente en este instante
            pred_noise_lat, pred_noise_atoms = self.model.forward_crystal(
                self.state, self.best_nodes, self.best_conns, 
                jnp.array([t_now]), elements, embeddings, props, 
                current_lattice, current_atoms
            )
            
            # B) Calculamos cómo se vería el cristal 100% limpio (x_0)
            x0_lat = self.diffusion.predict_crystal(current_lattice, pred_noise_lat, t_now)
            x0_atoms = self.diffusion.predict_crystal(current_atoms, pred_noise_atoms, t_now)
            
            # C) PASO DDIM (Matemática SOTA):
            # En lugar de saltar a x_0, retrocedemos al instante t_next inyectando
            # la cantidad de ruido exacta que corresponde a ese paso previo.
            alpha_bar_next = self.diffusion._get_alpha_bar(t_next)
            
            current_lattice = jnp.sqrt(alpha_bar_next) * x0_lat + jnp.sqrt(1 - alpha_bar_next) * pred_noise_lat
            current_atoms = jnp.sqrt(alpha_bar_next) * x0_atoms + jnp.sqrt(1 - alpha_bar_next) * pred_noise_atoms

        # 3. El paso final (t=0), devolvemos el cristal limpio (x_0)
        final_lattice = self.diffusion.predict_crystal(current_lattice, pred_noise_lat, times[-2])
        final_atoms = self.diffusion.predict_crystal(current_atoms, pred_noise_atoms, times[-2])
        
        # Cortamos posibles impurezas numéricas
        final_lattice = jnp.clip(final_lattice, 0.0, 1.0)
        final_atoms = jnp.clip(final_atoms, 0.0, 1.0)
        
        return final_lattice, final_atoms