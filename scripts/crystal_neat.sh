#!/bin/bash

# Definir variables
SCRIPT_NAME="crystal_neat.py"
OUT_FILE="../runs/v1/logs/terminal.out"
ERR_FILE="../runs/v1/logs/terminal.err"

echo "🚀 Iniciando trabajo en segundo plano..."
echo "Los logs estándar se guardarán en: $OUT_FILE"
echo "Los errores críticos se guardarán en: $ERR_FILE"

# La magia de Conda Run: Ejecuta Python dentro del entorno 'thesis' sin necesidad de activarlo
conda run -n thesis python3 $SCRIPT_NAME > $OUT_FILE 2> $ERR_FILE

echo "✅ Proceso terminado o colapsado. Revisa los archivos de log."