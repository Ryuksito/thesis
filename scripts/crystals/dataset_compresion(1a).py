import h5py
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore", module="pymatgen")
import pandas
from pymatgen.core import Structure, Element
from src.config import *

CSV_PATH = DATA_PATH + "dataset_fase1a.csv"

# --- LISTA DE ÉLITE (FASE 1A y 1B) ---
# Solo compilaremos el HDF5 con estos 16 semiconductores base
csv = pandas.read_csv(CSV_PATH)
TARGET_IDS = csv['material_id'].tolist()

INPUT_DIM  = (MAX_ELEMENTS * ELEM_FEATURES) + CRYSTAL_EMBED + CRYSTAL_PROPS  # 112
OUTPUT_DIM = 6 + (MAX_ATOMS * 4)  # Lattice(6) + 64 * (Z, x, y, z) = 262

# --- Constantes de Normalización ---
LATTICE_PARAMS = 6
MAX_ATOMIC_NUMBER = 118.0  
MAX_LATTICE_LENGTH = 5.0 # IMPORTANTE: Debe ser igual al del dataset grande
MAX_LATTICE_ANGLE = 180.0  

print(f"⚙️ Input dim:  {INPUT_DIM}")
print(f"⚙️ Output dim: {OUTPUT_DIM}")

# ── Propiedades atómicas desde pymatgen ──
def get_element_vector(symbol):
    try:
        el = Element(symbol)
        Z    = el.Z / MAX_ATOMIC_NUMBER
        EN   = float(el.X or 0) / 4.0
        r    = float(el.atomic_radius or 0) / 2.5
        mass = float(el.atomic_mass) / 300.0
        IE   = float(el.ionization_energies[0]) / 25.0 if el.ionization_energies else 0.0
        group  = float(el.group or 0) / 18.0
        period = float(el.row) / 7.0
        block_map = {'s': 0.25, 'p': 0.5, 'd': 0.75, 'f': 1.0}
        block = block_map.get(el.block, 0.0)
        type_map = {
            'metal': 0.0, 'alkali metal': 0.0, 'alkaline earth metal': 0.0,
            'transition metal': 0.0, 'post-transition metal': 0.0,
            'lanthanoid': 0.0, 'actinoid': 0.0,
            'metalloid': 0.5,
            'nonmetal': 1.0, 'halogen': 1.0, 'noble gas': 1.0,
        }
        el_type = type_map.get(str(el.category).lower(), 0.5)
        return [Z, EN, r, mass, IE, group, period, block, el_type]
    except Exception:
        return [0.0] * ELEM_FEATURES

# ── Normalización de propiedades del cristal ──
def normalize_props(bandgap, e_hull, density):
    bg   = float(bandgap   or 0.0) / 10.0          # 0-10 eV
    eh   = float(e_hull    or 0.0) / 2.0           # 0-2 eV/atom
    dens = float(density   or 0.0) / 20.0          # 0-20 g/cm3
    return [bg, eh, dens]

# ── Cargar y Filtrar Metadata ──
print("📂 Cargando metadata...")
metadata = json.load(open(METADATA_PATH))

def get_unique_elements(formula_str):
    try:
        from pymatgen.core import Composition
        comp = Composition(formula_str)
        return [str(el) for el in comp.elements]
    except:
        return []

metadata_valido = []
for d in metadata:
    # FILTRO ESTRICTO: Solo aceptamos si está en nuestra lista de Élite
    if d['material_id'] not in TARGET_IDS:
        continue
        
    cif_path = os.path.join(DATA_PATH, d['cif_filepath'])
    if not os.path.exists(cif_path):
        continue
    if not d.get('embeddings') or len(d['embeddings']) != CRYSTAL_EMBED:
        continue
    unique_els = get_unique_elements(d['formula'])
    if len(unique_els) == 0 or len(unique_els) > MAX_ELEMENTS:
        continue
    metadata_valido.append(d)

print(f"✅ Cristales Élite válidos para el dataset: {len(metadata_valido)}")

# ── Compilar HDF5 ──
output_path = DATA_PATH + "dataset_fase1a.h5"  # <--- Nombre de archivo cambiado
skipped = 0

print("🔨 Construyendo tensores...")
with h5py.File(output_path, "w") as hf:
    dset_elem_features = hf.create_dataset("context_elements", shape=(len(metadata_valido), MAX_ELEMENTS * ELEM_FEATURES), dtype=np.float32)
    dset_embeddings    = hf.create_dataset("context_embeddings", shape=(len(metadata_valido), CRYSTAL_EMBED), dtype=np.float32)
    dset_phys_props    = hf.create_dataset("context_props", shape=(len(metadata_valido), CRYSTAL_PROPS), dtype=np.float32)

    dset_lattice       = hf.create_dataset("target_lattice", shape=(len(metadata_valido), LATTICE_PARAMS), dtype=np.float32)
    dset_atoms         = hf.create_dataset("target_atoms", shape=(len(metadata_valido), MAX_ATOMS, 4), dtype=np.float32)

    dset_ids = hf.create_dataset("material_ids", shape=(len(metadata_valido),), dtype=np.float32)

    current_index = 0
    for i, d in enumerate(metadata_valido):
        try:
            struct = Structure.from_file(os.path.join(DATA_PATH, d['cif_filepath']))
        except Exception as e:
            skipped += 1
            continue

        if len(struct) > MAX_ATOMS:
            skipped += 1
            continue

        # ── INPUT 1: Elementos únicos con padding (45) ──
        unique_syms = list(dict.fromkeys([str(s.specie.symbol) for s in struct]))
        elem_matrix = np.zeros((MAX_ELEMENTS, ELEM_FEATURES), dtype=np.float32)
        for j, sym in enumerate(unique_syms[:MAX_ELEMENTS]):
            elem_matrix[j] = get_element_vector(sym)
        input_elements = elem_matrix.flatten()

        # ── INPUT 2: CHGNet embedding (64) ──
        input_embeddings = np.array(d['embeddings'], dtype=np.float32)

        # ── INPUT 3: Propiedades físicas (3) ──
        input_props = np.array(normalize_props(
            d.get('band_gap_eV'),
            d.get('energy_above_hull_eV_atom'),
            d.get('density_g_cm3')
        ), dtype=np.float32)


        # ── OUTPUT 1: Lattice Parameters (6) ──
        output_lattice = np.zeros(LATTICE_PARAMS, dtype=np.float32)
        output_lattice[0] = struct.lattice.a / MAX_LATTICE_LENGTH
        output_lattice[1] = struct.lattice.b / MAX_LATTICE_LENGTH
        output_lattice[2] = struct.lattice.c / MAX_LATTICE_LENGTH
        output_lattice[3] = struct.lattice.alpha / MAX_LATTICE_ANGLE
        output_lattice[4] = struct.lattice.beta / MAX_LATTICE_ANGLE
        output_lattice[5] = struct.lattice.gamma / MAX_LATTICE_ANGLE

        # ── OUTPUT 2: Posiciones atómicas ORDENADAS (256) ──
        output_atoms = np.zeros((MAX_ATOMS, 4), dtype=np.float32) # Matrix limpia con padding integrado
        sorted_sites = sorted(struct.sites, key=lambda site: (
            site.specie.Z, 
            site.frac_coords[0], 
            site.frac_coords[1], 
            site.frac_coords[2]
        ))

        for j, site in enumerate(sorted_sites[:MAX_ATOMS]):
            output_atoms[j, 0] = site.specie.Z / MAX_ATOMIC_NUMBER  
            output_atoms[j, 1] = float(site.frac_coords[0])         
            output_atoms[j, 2] = float(site.frac_coords[1])         
            output_atoms[j, 3] = float(site.frac_coords[2])  

        # Guardar
        dset_elem_features[current_index] = input_elements
        dset_embeddings[current_index]    = input_embeddings
        dset_phys_props[current_index]    = input_props
        dset_lattice[current_index]       = output_lattice
        dset_atoms[current_index]         = output_atoms
        dset_ids[current_index]           = int(d['material_id'].split('-')[1])
        
        current_index += 1

print(f"\n🚀 Dataset de Anclas compilado exitosamente:")
print(f"   Cristales guardados: {len(metadata_valido) - skipped}")
print(f"   Skipped por errores: {skipped}")
print(f"   Input shape:  ({len(metadata_valido) - skipped}, {INPUT_DIM})")
print(f"   Target shape: ({len(metadata_valido) - skipped}, {OUTPUT_DIM})")
print(f"   Archivo: {output_path}")