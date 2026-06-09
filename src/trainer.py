import os
import time
import csv
import jax
import numpy as np
from tqdm.auto import tqdm
import jax.numpy as jnp
from tensorneat.common import State
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt

from .loss import crystal_loss_fn, noise_loss_fn
from .data import JAXBatchLoader
from .models.diffusion import CrystalDiffusion
from .models.diffusion_network import CrystalDiffusionNetwork

@dataclass
class BaseLogs:
    gen: int                
    learning_rate: float
    init_noise_loss: float  # Error de ruido al INICIO del grad_step
    final_noise_loss: float # Error de ruido al FINAL del grad_step
    best_eval_loss: float   # Mejor individuo (crystal_loss_fn global)
    best_acsf: float        # Mejor ACSF (Geometría)
    best_z: float           # Mejor Z (Química)
    best_lat: float         # Mejor Lattice (Caja)
    best_var: float         # Mejor Varianza (Exploración)
    best_rep: float         # Mejor Repulsión (Anti-colapso)
    valid_pop: int          # Sobrevivientes sin NaN
    vram_peak_gb: float     # Memoria de GPU
    time_sec: float         # Tiempo por generación


class Trainer():
    def __init__(self, model: CrystalDiffusionNetwork, logs_path: str, n_generations:int=100, grad_steps_per_gen:int=10, seed:int=42, lr_min:float=1e-5, lr_max:float=0.005):
        self.n_generations = n_generations
        self.grad_steps_per_gen = grad_steps_per_gen
        self.key = jax.random.PRNGKey(seed)
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.logs_path = logs_path
        
        self.model = model
        self.diffusion = CrystalDiffusion()
        
        # Crear la carpeta de logs si no existe
        os.makedirs(self.logs_path, exist_ok=True)

    def save_logs(self, file_name: str, logs: BaseLogs):
        """Guarda automáticamente la dataclass en un CSV sin hardcodear columnas"""
        file_path = os.path.join(self.logs_path, file_name)
        file_exists = os.path.exists(file_path)
        
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Si el archivo es nuevo, escribe las cabeceras basadas en la dataclass
            if not file_exists:
                writer.writerow(logs.__dict__.keys())
            # Escribe los valores
            writer.writerow(logs.__dict__.values())
        
    @staticmethod
    @jax.jit
    def get_lr(step, total_grad_steps, lr_min, lr_max):
        """Decaimiento cosenoidal dinámico"""
        decay_ratio = step / total_grad_steps
        coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
        return lr_min + coeff * (lr_max - lr_min)

    def fit(self, train_loader: JAXBatchLoader, log_file_name="training_logs.csv"):
        loader_iter = iter(train_loader)
        state = State(randkey=self.key)
        state = self.model.neat.setup(state)

        # ====================================================================
        # 1. LAMARCKISMO (Estudiar el Ruido)
        # ====================================================================
        def single_grad_step(nodes, conns, state, batch_x, batch_y, current_lr, step_key):
            batch_size = batch_y['lattice'].shape[0]
            
            key_t, key_lat, key_atom = jax.random.split(step_key, 3)
            t_batch = jax.random.uniform(key_t, shape=(batch_size,))
            keys_lat = jax.random.split(key_lat, batch_size)
            keys_atom = jax.random.split(key_atom, batch_size)
            
            vmap_noise = jax.vmap(self.diffusion.add_noise, in_axes=(0, 0, 0))
            noisy_lat, noise_lat = vmap_noise(batch_y['lattice'], keys_lat, t_batch)
            noisy_atoms, noise_atoms = vmap_noise(batch_y['atoms'], keys_atom, t_batch)
            
            def loss_fn(n, c):
                # CORRECCIÓN 1: in_axes con 9 elementos (None, None, None, 0, 0, 0, 0, 0, 0)
                batch_forward = jax.vmap(self.model.forward_crystal, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
                pred_noise_lat, pred_noise_atoms = batch_forward(
                    state, n, c, t_batch, 
                    batch_x['elements'], batch_x['embeddings'], batch_x['props'], 
                    noisy_lat, noisy_atoms
                )
                mask = (batch_y['atoms'][:, :, 0] > 0).astype(jnp.float32)
                return noise_loss_fn(pred_noise_lat, pred_noise_atoms, noise_lat, noise_atoms, mask)

            loss, (grads_n, grads_c) = jax.value_and_grad(loss_fn, argnums=(0, 1))(nodes, conns)
            
            # CORRECCIÓN 2: SANITIZACIÓN DE GRADIENTES (El truco TensorNEAT)
            # Evita que el "Envenenamiento por NaN" destruya el aprendizaje Lamarckiano
            grads_n = jnp.where(jnp.isnan(grads_n), 0.0, grads_n)
            grads_c = jnp.where(jnp.isnan(grads_c), 0.0, grads_c)
            
            grads_n = jnp.clip(grads_n, -1.0, 1.0)
            grads_c = jnp.clip(grads_c, -1.0, 1.0)
            
            return nodes - current_lr * grads_n, conns - current_lr * grads_c, loss

        batch_grad_step = jax.jit(
            jax.vmap(single_grad_step, in_axes=(0, 0, None, None, None, None, None))
        )

        # ====================================================================
        # 2. DARWINISMO (Evaluación Física)
        # ====================================================================
        def single_eval_step(nodes, conns, state, batch_x, batch_y, eval_key):
            batch_size = batch_y['lattice'].shape[0]
            
            # CORRECCIÓN 3: Evaluación cercana a T=0
            # Evaluamos la arquitectura física desde un nivel de ruido mínimo (1%)
            # Esto permite que la predicción de 1 solo paso sea físicamente coherente.
            t_batch = jnp.full((batch_size,), 0.01)
            
            key_lat, key_atom = jax.random.split(eval_key, 2)
            keys_lat = jax.random.split(key_lat, batch_size)
            keys_atom = jax.random.split(key_atom, batch_size)
            
            vmap_noise = jax.vmap(self.diffusion.add_noise, in_axes=(0, 0, 0))
            noisy_lat, noise_lat = vmap_noise(batch_y['lattice'], keys_lat, t_batch)
            noisy_atoms, noise_atoms = vmap_noise(batch_y['atoms'], keys_atom, t_batch)
            
            batch_forward = jax.vmap(self.model.forward_crystal, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
            pred_noise_lat, pred_noise_atoms = batch_forward(
                state, nodes, conns, t_batch, 
                batch_x['elements'], batch_x['embeddings'], batch_x['props'], 
                noisy_lat, noisy_atoms
            )
            
            vmap_denoise = jax.vmap(self.diffusion.predict_crystal, in_axes=(0, 0, 0))
            pred_clean_lat = vmap_denoise(noisy_lat, pred_noise_lat, t_batch)
            pred_clean_atoms = vmap_denoise(noisy_atoms, pred_noise_atoms, t_batch)
            
            flat_pred_atoms = pred_clean_atoms.reshape(batch_size, -1)
            preds_flat = jnp.concatenate([pred_clean_lat, flat_pred_atoms], axis=-1)
            
            flat_target_atoms = batch_y['atoms'].reshape(batch_size, -1)
            targets_flat = jnp.concatenate([batch_y['lattice'], flat_target_atoms], axis=-1)
            
            return crystal_loss_fn(preds_flat, targets_flat, separate_results=True)

        batch_evaluate = jax.jit(
            jax.vmap(single_eval_step, in_axes=(0, 0, None, None, None, None))
        )
        
        # ====================================================================
        # 3. EL BUCLE PRINCIPAL (FIT)
        # ====================================================================
        total_grad_steps = self.n_generations * self.grad_steps_per_gen
        rng_key = self.key
        self.global_best_loss = float('inf')
        
        with tqdm(total=self.n_generations, desc="🧬 Escultores Lamarckianos", mininterval=1.0) as pbar:
            for generation in range(self.n_generations):
                start_time = time.time()
                rng_key, ask_key, eval_key = jax.random.split(rng_key, 3)

                pop_nodes, pop_conns = self.model.neat.ask(state)
                batch_x, batch_y, batch_ids = next(loader_iter)

                # --- TRACKEO DE LAMARCKISMO ---
                init_noise_loss = 0.0
                final_noise_loss = 0.0
                current_lr = 0.0

                for step in range(self.grad_steps_per_gen):
                    global_step = generation * self.grad_steps_per_gen + step
                    current_lr = self.get_lr(global_step, total_grad_steps, self.lr_min, self.lr_max)
                    rng_key, step_key = jax.random.split(rng_key)
                    
                    pop_nodes, pop_conns, batch_losses = batch_grad_step(
                        pop_nodes, pop_conns, state, batch_x, batch_y, current_lr, step_key
                    )
                    
                    # Capturamos el error inicial (paso 0) y el final (último paso)
                    mean_loss = np.nanmean(jax.device_get(batch_losses))
                    if step == 0:
                        init_noise_loss = mean_loss
                    if step == self.grad_steps_per_gen - 1:
                        final_noise_loss = mean_loss

                # --- TRACKEO DE DARWINISMO ---
                eval_results = batch_evaluate(pop_nodes, pop_conns, state, batch_x, batch_y, eval_key)
                
                cpu_totals = jax.device_get(eval_results[0])
                valid = np.isfinite(cpu_totals)
                cpu_losses_safe = np.where(valid, cpu_totals, 1e6)
                
                # Supervivencia
                fitnesses = -cpu_losses_safe
                state = self.model.neat.tell(state, fitnesses)

                # --- EXTRACCIÓN DE MÉTRICAS DEL MEJOR INDIVIDUO ---
                best_idx = np.argmin(cpu_losses_safe)
                best_loss = cpu_losses_safe[best_idx]

                # --- EXTRACCIÓN DE MÉTRICAS DEL MEJOR INDIVIDUO ---
                best_idx = np.argmin(cpu_losses_safe)
                best_loss = cpu_losses_safe[best_idx]
                
                # NUEVO: GUARDAR LOS PESOS DEL MEJOR INDIVIDUO HISTÓRICO
                if not hasattr(self, 'global_best_loss'):
                    self.global_best_loss = float('inf')
                
                if best_loss < self.global_best_loss:
                    self.global_best_loss = best_loss
                    best_nodes = pop_nodes[best_idx]
                    best_conns = pop_conns[best_idx]
                    
                    # Guardamos el genoma en la carpeta de logs
                    save_path = os.path.join(self.logs_path, "best_genome.npz")
                    jnp.savez(save_path, nodes=best_nodes, conns=best_conns)
                    # print(f" 💾 ¡Nuevo campeón guardado! Loss: {best_loss:.4f}")
                
                # Extraemos el desglose de la física del mejor individuo
                cpu_lats = jax.device_get(eval_results[1])
                cpu_acsfs = jax.device_get(eval_results[2])
                cpu_zs = jax.device_get(eval_results[3])
                cpu_vars = jax.device_get(eval_results[4])
                cpu_reps = jax.device_get(eval_results[5])

                # --- LECTURA SEGURA DE GPU (MAC M4 / CUDA) ---
                try:
                    # En Apple Silicon (Metal) o CUDA, extraemos la memoria
                    peak_bytes = jax.local_devices()[0].memory_stats().get('peak_bytes_in_use', 0)
                    vram_gb = peak_bytes / 1e9
                except:
                    vram_gb = 0.0

                gen_time = time.time() - start_time

                # --- GUARDAR EN EL CSV ---
                log_entry = BaseLogs(
                    gen=generation,
                    learning_rate=float(current_lr),
                    init_noise_loss=float(init_noise_loss),
                    final_noise_loss=float(final_noise_loss),
                    best_eval_loss=float(best_loss),
                    best_acsf=float(cpu_acsfs[best_idx]),
                    best_z=float(cpu_zs[best_idx]),
                    best_lat=float(cpu_lats[best_idx]),
                    best_var=float(cpu_vars[best_idx]),
                    best_rep=float(cpu_reps[best_idx]),
                    valid_pop=int(np.sum(valid)),
                    vram_peak_gb=float(vram_gb),
                    time_sec=float(gen_time)
                )
                self.save_logs(log_file_name, log_entry)

                # --- ACTUALIZAR LA CONSOLA ---
                pbar.set_postfix({
                    "Geo_Err": f"{best_loss:.2f}", 
                    "Noise(I->F)": f"{init_noise_loss:.3f}->{final_noise_loss:.3f}",
                    "VRAM": f"{vram_gb:.2f}GB"
                })
                pbar.update(1)

        # ====================================================================
        # 4. FIN DEL ENTRENAMIENTO: GENERACIÓN DE GRÁFICOS AUTOMÁTICOS
        # ====================================================================
        self.generate_plots(log_file_name)
    
    def generate_plots(self, log_file_name: str):
        """
        Lee el archivo CSV de logs y genera gráficos comparativos.
        Se ejecuta automáticamente al terminar el entrenamiento.
        """
        # 1. Definir las rutas
        csv_path = os.path.join(self.logs_path, log_file_name)
        if not os.path.exists(csv_path):
            print(f"⚠️ No se pudo graficar: No se encontró el archivo {csv_path}")
            return
            
        # 2. Crear carpeta para los gráficos (ej. 'fase1a_prueba_01_plots')
        plot_folder_name = log_file_name.replace('.csv', '_plots')
        plots_dir = os.path.join(self.logs_path, plot_folder_name)
        os.makedirs(plots_dir, exist_ok=True)
        
        # 3. Leer los datos
        df = pd.read_csv(csv_path)
        gens = df['gen']
        lr = df['learning_rate']
        
        # --- FUNCIÓN AUXILIAR: Eje Y doble (Izquierda: Métrica, Derecha: LR) ---
        def plot_with_lr(y_cols, title, ylabel, filename, log_scale_y=False):
            fig, ax1 = plt.subplots(figsize=(12, 7))
            
            # Eje Y Izquierdo (Métricas)
            colors = plt.cm.tab10.colors
            for i, col in enumerate(y_cols):
                if col in df.columns:
                    ax1.plot(gens, df[col], label=col, color=colors[i % len(colors)], linewidth=2)
            
            ax1.set_xlabel('Generaciones', fontsize=12, fontweight='bold')
            ax1.set_ylabel(ylabel, color='black', fontsize=12, fontweight='bold')
            if log_scale_y:
                ax1.set_yscale('log') # Log scale es vital para ver grandes diferencias de errores
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3)
            
            # Eje Y Derecho (Learning Rate)
            ax2 = ax1.twinx()
            ax2.plot(gens, lr, label='Learning Rate', color='red', linestyle='--', alpha=0.6, linewidth=1.5)
            ax2.set_ylabel('Learning Rate (Cosenoidal)', color='red', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Juntar las leyendas de ambos ejes
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, filename), dpi=300) # Alta resolución para el paper
            plt.close()

        # =========================================================
        # GENERACIÓN DE LOS GRÁFICOS SOLICITADOS
        # =========================================================
        
        # Gráfico 1: TODOS los errores comparados
        all_errors = ['init_noise_loss', 'final_noise_loss', 'best_eval_loss', 'best_acsf', 'best_z', 'best_lat', 'best_var', 'best_rep']
        plot_with_lr(all_errors, 'Comparativa Global de Todos los Errores', 'Loss (Log Scale)', '01_todos_los_errores.png', log_scale_y=True)
        
        # Gráfico 2: Solo Lamarckismo (Ruido Inicial vs Final)
        plot_with_lr(['init_noise_loss', 'final_noise_loss'], 'Aprendizaje Lamarckiano: Adivinación de Ruido', 'MSE Noise Loss', '02_ruido_lamarckiano.png')
        
        # Gráfico 3: Solo Darwinismo (Supervivencia y Geometría)
        darwin_errors = ['best_eval_loss', 'best_acsf', 'best_z', 'best_lat', 'best_var', 'best_rep']
        plot_with_lr(darwin_errors, 'Evaluación Darwiniana: Física y Geometría', 'Crystal Loss', '03_evaluacion_fisica.png', log_scale_y=True)
        
        # Gráfico 4: Tiempos por generación
        plot_with_lr(['time_sec'], 'Tiempo de Cómputo por Generación', 'Segundos', '04_tiempo_por_gen.png')
        
        # Gráfico 5: MÉTRICAS DEL SISTEMA Y POBLACIÓN (VRAM y Sobrevivientes)
        # Este lo hacemos especial sin el LR
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(gens, df['vram_peak_gb'], label='VRAM Usada (GB)', color='purple', linewidth=2)
        ax1.set_xlabel('Generaciones', fontsize=12, fontweight='bold')
        ax1.set_ylabel('VRAM (Gigabytes)', color='purple', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax1.set_ylim(0, df['vram_peak_gb'].max() + 1)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.plot(gens, df['valid_pop'], label='Población Válida (Sin NaNs)', color='green', linestyle='-.', linewidth=2)
        ax2.set_ylabel('Individuos', color='green', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, df['valid_pop'].max() + 5)
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
        
        plt.title('Estabilidad del Sistema: Memoria y Supervivencia', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, '05_sistema_vram.png'), dpi=300)
        plt.close()
        
        print(f"\n📊 ¡Terminado! 5 Gráficos generados exitosamente en:\n📁 {plots_dir}")