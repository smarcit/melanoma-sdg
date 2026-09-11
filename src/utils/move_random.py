import os
import random
import shutil

# Rutas de origen y destino
src_dir = "/home/marsi/TFM/data/imagefolder/synth/train/melanoma"
dst_dir = "/home/marsi/TFM/data/imagefolder/train/melanoma"

# Número de imágenes a mover
num_to_move = 1293

# Crear destino si no existe
os.makedirs(dst_dir, exist_ok=True)

# Listar imágenes disponibles en el origen
images = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Verificar que hay suficientes imágenes
if len(images) < num_to_move:
    raise ValueError(f"No hay suficientes imágenes en {src_dir}. Solo hay {len(images)}.")

# Seleccionar aleatoriamente
selected = random.sample(images, num_to_move)

# Mover archivos
for img in selected:
    shutil.move(os.path.join(src_dir, img), os.path.join(dst_dir, img))

print(f"Se han movido {num_to_move} imágenes de {src_dir} a {dst_dir}.")
