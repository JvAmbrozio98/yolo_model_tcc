import os
import glob

# Direcionar para a pasta do dataset divido (train, val, test)
yolo_dataset_dir = "dataset/folder_structure/yolo_dataset/labels"

# O ID correspondente à fratura nos seus arquivos de origem
FRACTURE_CLASS_ID = 3

def filter_fracture_labels(labels_dir):
    txt_files = glob.glob(os.path.join(labels_dir, "**/*.txt"), recursive=True)
    filtered_files_count = 0
    total_fracture_boxes = 0

    for file_path in txt_files:
        new_lines = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                
                # Se for a classe de Fratura (ID 3), mantemos e reindexamos para 0
                if class_id == FRACTURE_CLASS_ID:
                    parts[0] = '0'  # Reindexa para classe 0 no YOLO de classe única
                    new_lines.append(" ".join(parts))
                    total_fracture_boxes += 1
        
        # Sobrescreve o arquivo apenas com as anotações de fratura
        with open(file_path, 'w') as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        
        if new_lines:
            filtered_files_count += 1

    print("--- Filtragem das Fraturas Concluída ---")
    print(f"Total de arquivos com pelo menos uma fratura: {filtered_files_count}")
    print(f"Total de caixas de fraturas mantidas: {total_fracture_boxes}")

if __name__ == "__main__":
    filter_fracture_labels(yolo_dataset_dir)