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

import jax.numpy as jnp
import jax
import h5py
from tqdm.auto import tqdm
import csv
from tensorneat.common import State
import numpy as np

import importlib
import src.models.diffusion_network
import src.models.diffusion
import src.data
import src.loss
import src.config
import src.trainer
importlib.reload(src.models.diffusion_network)
importlib.reload(src.models.diffusion)
importlib.reload(src.config)
importlib.reload(src.models)
importlib.reload(src.data)
importlib.reload(src.loss)
importlib.reload(src.trainer)

from src.models.diffusion_network import CrystalDiffusionNetwork, BaseGenome, BaseMutation, BaseNEAT
from src.data import JAXBatchLoader
from src.loss import crystal_loss_fn
from src.trainer import Trainer

from src.config import *
KEY = jax.random.PRNGKey(SEED)

def load_crystal_dataset(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        # 1. Cargar el Contexto Segmentado (Inputs inmutables)
        context_elements = hf['context_elements'][:]
        context_embeddings = hf['context_embeddings'][:]
        context_props = hf['context_props'][:]
        
        # 2. Cargar el Lienzo Cristalino Segmentado (Targets para Difusión)
        target_lattice = hf['target_lattice'][:]
        target_atoms = hf['target_atoms'][:] # Mantiene shape (N, MAX_ATOMS, 4)
        
        # 3. Cargar IDs de los materiales
        np_ids = hf['material_ids'][:]
        
    # Cálculo estadístico de consumo en RAM
    print(f"📦 Muestras Totales: {len(np_ids)}")
    print(f"📐 Shapes segmentados disponibles internamente:")
    print(f"   └─ Matrices de Átomos con Padding : {target_atoms.shape}")
    print(f"   └─ Parámetros del Lattice Global  : {target_lattice.shape}")
    print(f"   └─ Embeddings Cuánticos (CHGNet)  : {context_embeddings.shape}")
    
    # Devolvemos tanto el formato unificado plano clásico como las piezas difusivas segmentadas
    return {
        "ids": np_ids,
        "context_elements": context_elements,
        "context_embeddings": context_embeddings,
        "context_props": context_props,
        "target_lattice": target_lattice,
        "target_atoms": target_atoms
    }

dataset_dict = load_crystal_dataset(DATASET_PATH)

# Separamos cada componente manteniendo sus formas nativas
inputs_dict = {
    "elements": dataset_dict['context_elements'],       # (N, 18)
    "embeddings": dataset_dict['context_embeddings'],   # (N, 64)
    "props": dataset_dict['context_props']              # (N, 3)
}

targets_dict = {
    "lattice": dataset_dict['target_lattice'],  # (N, 6)
    "atoms": dataset_dict['target_atoms']       # (N, MAX_ATOMS, 4) Matrix!
}
material_ids = dataset_dict["ids"]                     # (N,)

# Cálculo de las dimensiones base para la red
CONTEXT_DIM = inputs_dict['elements'].shape[1] + inputs_dict['embeddings'].shape[1] + inputs_dict['props'].shape[1]
MAX_ATOMS = targets_dict['atoms'].shape[1] + targets_dict['lattice'].shape[1]

print(f"⚙️ CONTEXT_DIM Global (Suma de Inputs): {CONTEXT_DIM}")
print(f"⚙️ MAX_ATOMS con Padding en Matriz    : {MAX_ATOMS}")

train_loader = JAXBatchLoader(inputs_dict, targets_dict, material_ids, batch_size=BATCH_SIZE)

# Verificamos un lote de prueba para estar 100% seguros
batch_x, batch_y, batch_idx = next(train_loader)

print("✅ Pipeline de datos verificado con éxito:")
print(f"   Matrix de átomos en batch (Lienzo): {batch_y['atoms'].shape}") # Debería ser (8, 4, 4)
print(f"   Lattice en batch (Lienzo)         : {batch_y['lattice'].shape}") # Debería ser (8, 6)
print(f"   Embeddings en batch (Contexto)    : {batch_x['embeddings'].shape}") # Debería ser (8, 64)

print("INPUT DIM: ", INPUT_DIM)
print("OUTPUT DIM: ", OUTPUT_DIM)

# ======================================================================
# CELDA 4: INICIALIZACIÓN Y BUCLE DE ENTRENAMIENTO
# ======================================================================

# 1. Configurar las estructuras de TensorNEAT usando src.config
neat_config = BaseNEAT(
    pop_size=POPSIZE,
    species_size=SPECIES_SIZE,
    survival_threshold=SURVIVAL_THRESHOLD
)

genome_config = BaseGenome(
    max_nodes=MAX_NODES,
    max_conns=MAX_CONNS
)

mutation_config = BaseMutation(
    conn_add_prob=CONN_ADD_PROB,
    conn_delete_prob=CONN_DELETE_PROB,
    node_add_prob=NODE_ADD_PROB,
    node_delete_prob=NODE_DELETE_PROB
)

# 2. Inicializar el Cerebro (Point-Cloud Network Condicionada)
model = CrystalDiffusionNetwork(
    elements_dim=MAX_ELEMENTS * ELEM_FEATURES,  # 18
    embeddings_dim=CRYSTAL_EMBED,               # 64
    props_dim=CRYSTAL_PROPS,                    # 3
    lattice_dim=LATTICE_PARAMS,                 # 6
    atom_dim=4,                                 # Z, x, y, z
    neat_config=neat_config,
    genome_config=genome_config,
    mutation_config=mutation_config
)

# 3. Inicializar el Entrenador (Lamarckismo + Darwinismo)
# Creamos una subcarpeta 'logs' dentro de tu carpeta principal de datos
LOGS_DIR = "runs/v2/"

trainer = Trainer(
    model=model,
    logs_path=LOGS_DIR,
    n_generations=N_GENERATIONS,
    grad_steps_per_gen=GRAD_STEPS_PER_GEN,
    seed=SEED,
    lr_min=LR_MIN,
    lr_max=LR_MAX
)


# 4. ¡Arrancar la Evolución Difusiva!
print(f"🚀 Iniciando entrenamiento difusivo Lamarckiano...")
print(f"   └─ Generaciones: {N_GENERATIONS}")
print(f"   └─ Grad Steps por Gen: {GRAD_STEPS_PER_GEN}")
print(f"   └─ Población NEAT: {POPSIZE}")
print(f"   └─ Batch Size: {BATCH_SIZE}\n")

# NOTA MENTAL: La primera generación tardará un poco más porque JAX
# debe compilar el grafo computacional (JIT) para tu Mac M4.
# A partir de la generación 2, volará.
trainer.fit(train_loader)

