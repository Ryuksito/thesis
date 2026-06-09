import os
from dotenv import load_dotenv

load_dotenv()

# Rutas cargadas del .env
DATA_PATH = os.getenv("DATA_PATH")
CIF_PATH = os.path.join(DATA_PATH, "cif/")
METADATA_PATH = os.path.join(DATA_PATH, "_metadata.json")
EMBEDDINGS_PATH = os.path.join(DATA_PATH, "_embeddings.json")
FINAL_DATA_PATH = os.path.join(DATA_PATH, "_data.json")
DATASET_PATH = os.path.join(DATA_PATH, "dataset_fase1a.h5")

#===============================Dataset===============================
# --- Dimensiones Confirmadas ---
MAX_ELEMENTS = 2       # Tipos únicos de elemento por estructura
ELEM_FEATURES = 9      # Features por elemento
MAX_ATOMS = 4         # Límite de átomos en output
CRYSTAL_EMBED = 64     # CHGNet embedding
CRYSTAL_PROPS = 3      # bandgap, e_hull, density
# --- Constantes de Normalización ---
LATTICE_PARAMS = 6
MAX_ATOMIC_NUMBER = 118.0  
MAX_LATTICE_LENGTH = 5.0 # IMPORTANTE: Debe ser igual al del dataset grande
MAX_LATTICE_ANGLE = 180.0  
# --- Constantes de Entrada y Salida
INPUT_DIM = (MAX_ELEMENTS * ELEM_FEATURES) + CRYSTAL_EMBED + CRYSTAL_PROPS  # 112
OUTPUT_DIM = 6 + (MAX_ATOMS * 4)  # Lattice(6) + 64 * (Z, x, y, z) = 262

#===============================Neat===============================
POPSIZE = 64
SPECIES_SIZE = 20
SURVIVAL_THRESHOLD = 0.3
MAX_NODES = 200
MAX_CONNS = 2200
CONN_ADD_PROB = 0.6
CONN_DELETE_PROB = 0.3
NODE_ADD_PROB = 0.3
NODE_DELETE_PROB = 0.05

#===============================Structure===============================
STRUCTURE_DIM = MAX_ATOMS * 4

#===============================Training Loop===============================
BATCH_SIZE = 8
N_GENERATIONS = 50
GRAD_STEPS_PER_GEN = 10
GRAD_LR = 0.005
LR_MAX = 0.005
LR_MIN = 1e-5
TOTAL_GRAD_STEPS = N_GENERATIONS * GRAD_STEPS_PER_GEN
LOG_EVERY = 1

SEED = 42