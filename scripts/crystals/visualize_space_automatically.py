import os
import json
import pandas as pd
import numpy as np
import matplotlib
# Obligamos a matplotlib a trabajar en modo "Headless" (Sin interfaz gráfica)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def main():
    PATH = "crystals/_data.json"
    print(f"📂 Cargando {PATH}...")
    
    with open(PATH, 'r') as f:
        data = json.load(f)
    
    print("📊 Procesando embeddings 3D...")
    df = pd.DataFrame(data)
    df[['x', 'y', 'z']] = pd.DataFrame(df['embeddings_3D'].tolist(), index=df.index)
    
    # 🎨 ESTÉTICA DE INGENIERÍA (Dark Mode)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), dpi=150) # Alta resolución (1080p aprox)
    ax = fig.add_subplot(111, projection='3d')
    
    # Colorear por energía y ajustar tamaño
    scatter = ax.scatter(
        df['x'], df['y'], df['z'],
        c=df['energy_above_hull_eV_atom'],
        cmap='viridis',
        s=8,        # Tamaño del punto
        alpha=0.8,  # Ligera transparencia para ver densidad
        edgecolors='none'
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Inestabilidad (eV/atom)', fontsize=12, labelpad=15, color='white')
    
    # Limpiamos absolutamente toda la "basura visual" (ejes, fondos, números)
    ax.set_axis_off()
    ax.set_box_aspect(None, zoom=3)
    
    # Ajustamos el ángulo de inclinación vertical (Elevación)
    ax.view_init(elev=20, azim=0)

    # 🎥 MOTOR DE ANIMACIÓN
    print("🎥 Compilando video MP4 (Esto usará tu CPU al 100% un momento)...")
    
    # Definimos cómo cambia el gráfico en cada fotograma
    def update(frame):
        # Rotamos el ángulo azimutal
        ax.view_init(elev=20, azim=frame)
        
        # Imprimir progreso en la misma línea
        if frame % 10 == 0:
            print(f"  -> Renderizando ángulo {frame}° / 360°", end="\r")
        return scatter,

    # Creamos la animación: 360 grados, saltando de 1 en 1 (360 frames)
    anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), blit=False)
    
    # Configuramos el escritor de video (FFmpeg a 30 FPS, buena calidad)
    writer = FFMpegWriter(fps=30, bitrate=2500)
    output_file = "universo_quimico_rotacion.mp4"
    anim.save(output_file, writer=writer)

    writer = PillowWriter(fps=30)
    output_file = "universo_quimico_rotacion.gif"
    anim.save(output_file, writer=writer)
    
    print(f"\n✅ ¡Éxito! Video guardado como '{output_file}'")

if __name__ == "__main__":
    main()