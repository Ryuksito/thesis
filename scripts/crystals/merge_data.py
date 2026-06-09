import json
import os
import numpy as np
import warnings
from umap import UMAP

# --- CONFIGURACIÓN DE RUTAS ---
DATA_PATH = "crystals/"
METADATA_PATH = DATA_PATH + "_metadata.json"
EMBEDDINGS_PATH = DATA_PATH + "_embeddings.json"
FINAL_DATA_PATH = DATA_PATH + "_data.json"

def main():
    print("🔄 Iniciando consolidación y verificación del dataset...")

    # 1. Cargar las fuentes (Ya no necesitamos ids_catalogo)
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata_list = json.load(f) 
        with open(EMBEDDINGS_PATH, 'r') as f:
            embeddings = json.load(f)
    except Exception as e:
        print(f"❌ Error cargando archivos base: {e}")
        return

    print(f"📊 Metadatos limpios disponibles: {len(metadata_list)}")
    print(f"🧠 Embeddings extraídos: {len(embeddings)}")

    # 2. Unión y Verificación
    final_dataset = []
    raw_embeddings_matrix = []
    errores = []

    print("\n🧐 Verificando integridad de cada estructura...")
    
    # Iteramos directamente sobre la fuente de la verdad: los metadatos
    for m_data in metadata_list:
        mp_id = m_data["material_id"]
        
        # Verificamos si existe el embedding
        if mp_id not in embeddings:
            errores.append(f"{mp_id}: Falta embedding")
            continue

        # Verificamos si existe el archivo CIF físico
        cif_relative_path = m_data["cif_filepath"]
        if not os.path.exists(os.path.join(DATA_PATH, cif_relative_path)):
            errores.append(f"{mp_id}: Archivo CIF no encontrado en disco")
            continue

        # Si pasó las pruebas, extraemos los datos (incluyendo num_sites)
        entry = {
            "material_id": mp_id,
            "formula": m_data["formula"],
            "cif_filepath": cif_relative_path,
            "band_gap_eV": m_data.get("band_gap_eV"),
            "energy_above_hull_eV_atom": m_data.get("energy_above_hull_eV_atom"),
            "density_g_cm3": m_data.get("density_g_cm3"),
            "formation_energy_eV_atom": m_data.get("formation_energy_eV_atom"),
            "num_sites": m_data.get("num_sites"),
            "num_elements": m_data.get("num_elements"),
            "embeddings": embeddings[mp_id],
            "embeddings_3D": None 
        }
        
        final_dataset.append(entry)
        raw_embeddings_matrix.append(embeddings[mp_id])

    print(f"✅ Estructuras íntegras (Single Source of Truth): {len(final_dataset)}")
    if errores:
        print(f"⚠️ Estructuras descartadas por inconsistencia: {len(errores)}")

    # 3. Reducción de Dimensionalidad (UMAP 3D)
    if final_dataset:
        print(f"\n🔮 Calculando UMAP 3D para {len(final_dataset)} puntos...")
        matrix = np.array(raw_embeddings_matrix)
        
        # Silenciamos el warning inofensivo de UMAP sobre n_jobs y random_state
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=-1)
            embeddings_3d = reducer.fit_transform(matrix)

        for i, coord in enumerate(embeddings_3d):
            final_dataset[i]["embeddings_3D"] = [round(float(x), 6) for x in coord]

    # 4. Guardado Final
    print(f"\n💾 Guardando dataset consolidado en {FINAL_DATA_PATH}...")
    with open(FINAL_DATA_PATH, 'w') as f:
        json.dump(final_dataset, f, indent=4)

    if errores:
        with open(DATA_PATH + "_merge_errors.log", "w") as f:
            f.write("\n".join(errores))
        print(f"📝 Log de errores guardado en _merge_errors.log")

    print("\n✨ ¡Listo! Archivo _data.json generado con éxito.")

if __name__ == "__main__":
    main()