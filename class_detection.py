import os
import glob

# Caminho onde estão seus arquivos de rótulo YOLO
labels_dir = "dataset/folder_structure/yolov5/labels"
txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))

classes_encontradas = set()
total_linhas = 0

for file_path in txt_files:
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])  # O primeiro número da linha é o ID da classe
                classes_encontradas.add(class_id)
                total_linhas += 1

print(f"Total de caixas/anotações analisadas: {total_linhas}")
print(f"IDs das classes encontradas nos arquivos .txt: {sorted(list(classes_encontradas))}")