import json
import pandas as pd
import plotly.express as px

def main():
    PATH = "crystals/_data.json"
    print(f"📂 Cargando {PATH}...")
    
    with open(PATH, 'r') as f:
        data = json.load(f)
    
    # Convertimos a DataFrame para manejo fácil
    print("📊 Procesando datos para visualización...")
    df = pd.DataFrame(data)
    
    # Extraemos las coordenadas 3D en columnas separadas
    df[['x', 'y', 'z']] = pd.DataFrame(df['embeddings_3D'].tolist(), index=df.index)
    
    # Creamos la figura 3D
    print("🎨 Generando mapa interactivo...")
    fig = px.scatter_3d(
        df, 
        x='x', y='y', z='z',
        color='energy_above_hull_eV_atom',  # Los puntos brillarán según su estabilidad
        hover_name='formula',
        hover_data=['material_id', 'band_gap_eV', 'num_sites', 'num_elements'],
        title='Mapa del Universo Químico (CHGNet + UMAP)',
        labels={'energy_above_hull_eV_atom': 'E_hull (eV/atom)'},
        color_continuous_scale='Viridis',
        opacity=0.7
    )

    # Ajustar tamaño de los puntos
    fig.update_traces(marker=dict(size=2))
    
    print("🚀 Abriendo visualización en el navegador...")
    fig.show()

if __name__ == "__main__":
    main()