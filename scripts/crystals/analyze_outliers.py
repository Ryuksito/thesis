import os
import json
import warnings
from pymatgen.core import Structure
from tqdm import tqdm

warnings.filterwarnings("ignore", module="pymatgen")

DATA_PATH = "crystals/"
METADATA_PATH = DATA_PATH + "_data.json"

def get_top_n(data_list, n=10, reverse=True):
    """Ordena una lista de tuplas y devuelve los top N."""
    return sorted(data_list, key=lambda x: x[0], reverse=reverse)[:n]

def main():
    print("🕵️‍♂️ Iniciando Análisis Forense de Outliers...")
    
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"❌ Error cargando metadatos: {e}")
        return

    # Listas para guardar (Valor, ID, Fórmula, Eje/Detalle)
    lattice_lengths = []
    lattice_angles_max = []
    lattice_angles_min = []
    densities = []
    
    errores = 0

    pbar = tqdm(metadata, desc="Escaneando Estructuras")
    for d in pbar:
        mp_id = d['material_id']
        formula = d['formula']
        cif_path = os.path.join(DATA_PATH, d['cif_filepath'])
        
        if not os.path.exists(cif_path):
            errores += 1
            continue

        try:
            struct = Structure.from_file(cif_path)
            lat = struct.lattice
            
            # 1. Longitudes de Red (a, b, c)
            lattice_lengths.append((lat.a, mp_id, formula, "eje 'a'"))
            lattice_lengths.append((lat.b, mp_id, formula, "eje 'b'"))
            lattice_lengths.append((lat.c, mp_id, formula, "eje 'c'"))
            
            # 2. Ángulos Máximos y Mínimos
            angulos = [("alpha", lat.alpha), ("beta", lat.beta), ("gamma", lat.gamma)]
            for nombre, valor in angulos:
                lattice_angles_max.append((valor, mp_id, formula, nombre))
                lattice_angles_min.append((valor, mp_id, formula, nombre))

            # 3. Densidad (ya la tenemos en el metadata, pero confirmamos)
            densidad = d.get('density_g_cm3', 0)
            if densidad > 0:
                densities.append((densidad, mp_id, formula, "g/cm³"))

        except Exception as e:
            errores += 1
            continue

    # --- IMPRIMIR REPORTE ---
    print("\n" + "="*60)
    print("🚨 REPORTE DE OUTLIERS EXTREMOS 🚨")
    print("="*60)

    print("\n📏 TOP 10 LONGITUDES DE RED MÁS LARGAS (Lattice Length):")
    for val, mp_id, form, detalle in get_top_n(lattice_lengths):
        print(f"  - {val:>8.4f} Å | {mp_id:<12} | {form:<10} | ({detalle})")

    print("\n📐 TOP 5 ÁNGULOS MÁS OBTUSOS (Casi 180°):")
    for val, mp_id, form, detalle in get_top_n(lattice_angles_max, n=5):
        print(f"  - {val:>8.4f}° | {mp_id:<12} | {form:<10} | ({detalle})")

    print("\n🔪 TOP 5 ÁNGULOS MÁS AGUDOS (Casi 0°):")
    for val, mp_id, form, detalle in get_top_n(lattice_angles_min, n=5, reverse=False):
        print(f"  - {val:>8.4f}° | {mp_id:<12} | {form:<10} | ({detalle})")

    print("\n🧱 TOP 5 MATERIALES MÁS DENSOS:")
    for val, mp_id, form, detalle in get_top_n(densities, n=5):
        print(f"  - {val:>8.4f} {detalle} | {mp_id:<12} | {form:<10}")
        
    print("\n🎈 TOP 5 MATERIALES MENOS DENSOS (Casi vacío):")
    for val, mp_id, form, detalle in get_top_n(densities, n=5, reverse=False):
        print(f"  - {val:>8.4f} {detalle} | {mp_id:<12} | {form:<10}")

    print("="*60)

if __name__ == "__main__":
    main()