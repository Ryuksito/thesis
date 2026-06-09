import os
import sys
import time
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.85'
if os.getcwd().endswith("scripts"):
    os.chdir("../")
sys.path.append(os.getcwd())    

print(f"Directorio de trabajo actual: {os.getcwd()}")
print(f"Rutas de importación de Python: {sys.path[-1]}")

import jax, jax.numpy as jnp
from tensorneat import algorithm, genome
from tensorneat.pipeline import Pipeline
from tensorneat.genome import DefaultGenome, BiasNode, DefaultMutation
from tensorneat.problem.func_fit import CustomFuncFit
from tensorneat.common import ACT, AGG, State
from tensorneat.common.functions import act_jnp
from tensorneat.pipeline import Pipeline
from tensorneat.algorithm.neat import NEAT
from tensorneat.algorithm.hyperneat import HyperNEAT, FullSubstrate
import h5py
import numpy as np
from tqdm.auto import tqdm
import csv

from src.data import JAXBatchLoader
from src.loss import crystal_loss_fn

DATA_PATH = "crystals/"
CIF_PATH = DATA_PATH + "cif/"
METADATA_PATH = DATA_PATH + "_metadata.json"
EMBEDDINGS_PATH = DATA_PATH + "_embeddings.json"
FINAL_DATA_PATH = DATA_PATH + "_data.json"
DATASET_PATH = DATA_PATH + "dataset_fase1a.h5" 
RUN_DIR = "runs/v1/"
CHKPT_DIR = os.path.join(RUN_DIR, "chkpt")
LOGS_DIR = os.path.join(RUN_DIR, "logs")
os.makedirs(CHKPT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

log_file_path = os.path.join(LOGS_DIR, "history.csv")
with open(log_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Cabeceras ampliadas con todo lo útil
    writer.writerow([
        'gen', 
        'learning_rate', 
        'init_grad_loss',   # Error al inicio de la relajación
        'final_grad_loss',  # Error al final de la relajación
        'best_eval_loss',   # Mejor individuo en el dataset de validación
        'best_acsf',        # Mejor ACSF (Forma)
        'best_z',           # Mejor Z (Química)
        'best_lat',         # Mejor Lattice (Caja)
        'best_var',         # Mejor Varianza (Exploración)
        'best_rep',         # Mejor Repulsión (Anti-colapso)
        'valid_pop',        # Cuántos sobrevivieron sin dar NaN
        'vram_peak_gb', 
        'time_sec'
    ])

# Neat
POPSIZE = 96
SPECIES_SIZE = 20
SURVIVAL_THRESHOLD = 0.3
MAX_NODES = 200
MAX_CONNS = 2200
CONN_ADD_PROB = 0.6
CONN_DELETE_PROB = 0.3
NODE_ADD_PROB = 0.3
NODE_DELETE_PROB = 0.05

# Structure
MAX_ELEMENTS = 2       # Tipos únicos de elemento por estructura
ELEM_FEATURES = 9      # Features por elemento
MAX_ATOMS = 4         # Límite de átomos en output
CRYSTAL_EMBED = 64     # CHGNet embedding
CRYSTAL_PROPS = 3      # bandgap, e_hull, density
LATTICE_PARAMS = 6     # a, b, c, alpha, beta, gamma
STRUCTURE_DIM = MAX_ATOMS * 4 # (Z, x, y, z)

# Training Loop
BATCH_SIZE = 8
INPUT_DIM  = (MAX_ELEMENTS * ELEM_FEATURES) + CRYSTAL_EMBED + CRYSTAL_PROPS
OUTPUT_DIM = LATTICE_PARAMS + STRUCTURE_DIM
N_GENERATIONS = 500
GRAD_STEPS_PER_GEN = 10
GRAD_LR = 0.005
LR_MAX = 0.005
LR_MIN = 1e-5
TOTAL_GRAD_STEPS = N_GENERATIONS * GRAD_STEPS_PER_GEN
LOG_EVERY = 1

SEED = 42

@jax.jit
def get_lr(step):
    """Calcula el LR usando decaimiento cosenoidal puro en JAX"""
    decay_ratio = step / TOTAL_GRAD_STEPS
    coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
    return LR_MIN + coeff * (LR_MAX - LR_MIN)

def load_crystal_dataset(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        np_inputs  = hf['inputs'][:]
        np_targets = hf['targets'][:]
        np_ids     = hf['material_ids'][:].astype(str)
    
    mb = (np_inputs.nbytes + np_targets.nbytes) / 1e6
    print(f"Dataset cargado en RAM: {mb:.1f} MB")
    print(f"Muestras: {len(np_inputs)}")
    return np_inputs, np_targets, np_ids


X_train, Y_train, material_ids = load_crystal_dataset(DATASET_PATH)
train_loader = JAXBatchLoader(X_train, Y_train, batch_size=BATCH_SIZE)    

INPUT_DIM  = X_train.shape[1]
OUTPUT_DIM = Y_train.shape[1]

print(f"INPUT_DIM:  {INPUT_DIM}")
print(f"OUTPUT_DIM: {OUTPUT_DIM}")
print(f"Cristales:  {len(material_ids)}")


neat = algorithm.NEAT(
    pop_size=POPSIZE,
    species_size=SPECIES_SIZE,
    survival_threshold=SURVIVAL_THRESHOLD,
    genome=genome.DefaultGenome(
        num_inputs=INPUT_DIM,
        num_outputs=OUTPUT_DIM,
        max_nodes=MAX_NODES,
        max_conns=MAX_CONNS,      
        init_hidden_layers=(),
        output_transform=act_jnp.sigmoid_,
        mutation=DefaultMutation(
            conn_add=CONN_ADD_PROB,         # correlation explorer
            conn_delete=CONN_DELETE_PROB,   # synaptic pruning
            node_add=NODE_ADD_PROB,         # creator of depth
            node_delete=NODE_DELETE_PROB,   # destructor of connections
        ),
    ),
)


state = State(randkey=jax.random.key(SEED))
state = neat.setup(state)
g = neat.genome

def grad_step(nodes, conns, state, x_batch, y_batch, current_lr):
    def loss_fn(preds):
        return crystal_loss_fn(preds, y_batch)

    loss, (grads_n, grads_c) = g.grad(state, nodes, conns, x_batch, loss_fn)
    grads_n = jnp.clip(grads_n, -1.0, 1.0)
    grads_c = jnp.clip(grads_c, -1.0, 1.0)
    
    # Usamos el LR dinámico en lugar de la constante
    return nodes - current_lr * grads_n, conns - current_lr * grads_c, loss

# Vectorizar sobre población y compilar UNA SOLA VEZ
batch_grad_step = jax.jit(
    jax.vmap(grad_step, in_axes=(0, 0, None, None, None, None))
)

def evaluate_step(nodes, conns, state, x_all, y_all):
    transformed_network = g.transform(state, nodes, conns)
    preds = jax.vmap(g.forward, in_axes=(None, None, 0))(state, transformed_network, x_all)
    
    # Activamos la separación de resultados SOLO aquí
    return crystal_loss_fn(preds, y_all, separate_results=True)

batch_evaluate = jax.jit(
    jax.vmap(evaluate_step, in_axes=(0, 0, None, None, None))
)

print("batch_grad_step compilado y listo.")
print(f"Población: {POPSIZE} | Batch: {BATCH_SIZE} | Grad steps/gen: {GRAD_STEPS_PER_GEN}")


pbar = tqdm(range(N_GENERATIONS), desc="🧬 Evolución NEAT", mininterval=60.0)
history = []
WINDOW_PCT = 1.0
steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
window_size = max(1, int(steps_per_epoch * WINDOW_PCT))

for generation in pbar:
    start_time = time.time()
    # 1. NEAT genera la población de esta generación
    pop_nodes, pop_conns = neat.ask(state)

    x_batch, y_batch, batch_idx = next(train_loader)

    current_mids = [material_ids[i] for i in batch_idx]
    mids_str = ",".join(current_mids)

    # Capturamos el LR inicial de esta generación para el log
    gen_lr = float(get_lr(generation * GRAD_STEPS_PER_GEN))

    # 3. Gradient descent sobre toda la población con este batch
    for step in range(GRAD_STEPS_PER_GEN):
        global_step = generation * GRAD_STEPS_PER_GEN + step
        current_lr = get_lr(global_step)
        
        pop_nodes, pop_conns, batch_losses = batch_grad_step(
            pop_nodes, pop_conns, state, x_batch, y_batch, current_lr
        )
        # Capturamos cómo empezó la población en el primer step
        if step == 0:
            loss_estudio_inicio = float(np.nanmin(jax.device_get(batch_losses)))
        # Capturamos cómo terminó la población en el último step
        if step == GRAD_STEPS_PER_GEN - 1:
            loss_estudio_final = float(np.nanmin(jax.device_get(batch_losses)))


    eval_results = batch_evaluate(pop_nodes, pop_conns, state, X_train, Y_train)
    cpu_totals = jax.device_get(eval_results[0])
    cpu_lats   = jax.device_get(eval_results[1])
    cpu_acsfs  = jax.device_get(eval_results[2])
    cpu_zs     = jax.device_get(eval_results[3])
    cpu_vars   = jax.device_get(eval_results[4])
    cpu_reps   = jax.device_get(eval_results[5])

    # 4. Fitness = negativo del loss (NEAT maximiza)
    valid = np.isfinite(cpu_totals)
    cpu_losses_safe = np.where(valid, cpu_totals, 1e6)
    fitnesses = -cpu_losses_safe

    # 5. NEAT selecciona y genera siguiente generación
    state = neat.tell(state, fitnesses)

    # 6. Telemetría y Logging Dinámico
    valid_count = int(valid.sum())
    valid_losses = cpu_losses_safe[valid]
    
    # Cálculos estadísticos seguros
    bl = float(np.min(valid_losses)) if valid_count > 0 else float('nan')
    best_idx = int(np.nanargmin(cpu_losses_safe))
    best_lat  = float(cpu_lats[best_idx])
    best_acsf = float(cpu_acsfs[best_idx])
    best_z    = float(cpu_zs[best_idx])
    best_var  = float(cpu_vars[best_idx])
    best_rep  = float(cpu_reps[best_idx])
    
    peak_gb = jax.devices("gpu")[0].memory_stats()['peak_bytes_in_use'] / 1e9
    gen_time = time.time() - start_time
    
    history.append({
        'gen': generation,
        'init_loss': loss_estudio_inicio,
        'best_loss': bl,
        'valid': valid_count
    })

    # Guardar todo en el CSV con un buen formato de decimales
    with open(log_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            generation, 
            f"{gen_lr:.6e}",           # LR en notación científica
            f"{loss_estudio_inicio:.6f}", 
            f"{loss_estudio_final:.6f}", 
            f"{bl:.6f}", 
            f"{best_acsf:.4f}",   # Forma
            f"{best_z:.4f}",      # Química
            f"{best_lat:.4f}",    # Caja
            f"{best_var:.4f}",    # Varianza
            f"{best_rep:.4f}",    # Repulsión
            valid_count, 
            f"{peak_gb:.2f}", 
            f"{gen_time:.2f}"
        ])

    if generation % 10 == 0 or generation == N_GENERATIONS - 1:
        best_idx = int(np.nanargmin(cpu_losses_safe))
        best_nodes = jax.device_get(pop_nodes[best_idx])
        best_conns = jax.device_get(pop_conns[best_idx])
        
        chkpt_file = os.path.join(CHKPT_DIR, f"best_gen_{generation:04d}.npz")
        np.savez(chkpt_file, nodes=best_nodes, conns=best_conns, loss=bl)


    ventana = history[-window_size:]
    init_mean = np.mean([h['init_loss'] for h in ventana])
    best_mean = np.mean([h['best_loss'] for h in ventana])
    
    pbar.set_postfix({
        'IDs': mids_str[:15],
        f'InitMean': f"{init_mean:.2f}",
        f'BestMean': f"{best_mean:.2f}",
        f'BestIndividual': f"{bl:.2f}",
        'VRAM': f"{peak_gb:.1f}G"
    })

print("\n✅ Entrenamiento finalizado. Checkpoints guardados en", CHKPT_DIR)

# Usamos cpu_losses_safe que tiene la evaluación global de la última generación
best_idx = int(np.nanargmin(cpu_losses_safe))
best_nodes = pop_nodes[best_idx]
best_conns = pop_conns[best_idx]

print(f"Mejor individuo (Global Eval): loss = {cpu_losses_safe[best_idx]:.6f}")

best_transformed = jax.jit(g.transform)(state, best_nodes, best_conns)
forward_fn = jax.jit(g.forward, static_argnums=(0,))

print("\nPredicciones vs targets:")
for i in range(len(material_ids)):
    x = jnp.array(X_train[i:i+1])
    pred = forward_fn(state, best_transformed, x[0])
    target = Y_train[i]

    pred_lat = jax.device_get(pred[:LATTICE_PARAMS])
    tgt_lat  = target[:LATTICE_PARAMS]

    print(f"\n{material_ids[i]}:")
    print(f"  Lattice real: {jnp.round(jnp.array(tgt_lat), 4)}")
    print(f"  Lattice pred: {jnp.round(jnp.array(pred_lat), 4)}")



