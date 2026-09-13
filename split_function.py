import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def split_grazpedwri_dataset(
    csv_path="dataset/dataset.csv",
    images_dir="dataset/folder_structure/images_conversion",
    labels_dir="dataset/folder_structure/yolov5/labels",
    output_dir="dataset/folder_structure/yolo_dataset",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    img_extension=".png", # Altere para .jpg se suas imagens convertidas forem JPG
    random_seed=42
):
    """
    Realiza o split do dataset GRAZPEDWRI-DX a partir de 'images_conversion' e 'yolov5',
    organizando por patient_id para evitar Data Leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "A soma das proporções deve ser igual a 1.0"

    print(f"Lendo metadados de: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Obter pacientes únicos para garantir que não haja vazamento entre treinos
    unique_patients = df['patient_id'].unique()
    
    # 2. Divisão Treino vs (Val + Test)
    train_patients, temp_patients = train_test_split(
        unique_patients, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_seed
    )
    
    # 3. Divisão Validação vs Teste
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients, 
        test_size=(1.0 - relative_val_ratio), 
        random_state=random_seed
    )
    
    # Mapear cada paciente para sua respectiva partição
    split_map = {}
    for p in train_patients: split_map[p] = 'train'
    for p in val_patients: split_map[p] = 'val'
    for p in test_patients: split_map[p] = 'test'
    
    df['split'] = df['patient_id'].map(split_map)

    # 4. Criar estrutura de pastas do YOLO
    for split_type in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split_type), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split_type), exist_ok=True)

    # Obter lista única de nomes base dos arquivos (filestem)
    unique_files = df[['filestem', 'split']].drop_duplicates()
    
    copied_images = {'train': 0, 'val': 0, 'test': 0}
    copied_labels = {'train': 0, 'val': 0, 'test': 0}
    
    # 5. Copiar imagens de images_conversion e rótulos de yolov5
    for _, row in unique_files.iterrows():
        stem = str(row['filestem'])
        split = row['split']
        
        # Nomes dos arquivos de origem
        img_filename = f"{stem}{img_extension}"
        txt_filename = f"{stem}.txt"
        
        src_img = os.path.join(images_dir, img_filename)
        src_txt = os.path.join(labels_dir, txt_filename)
        
        dest_img = os.path.join(output_dir, 'images', split, img_filename)
        dest_txt = os.path.join(output_dir, 'labels', split, txt_filename)
        
        # Copiar Imagem
        if os.path.exists(src_img):
            shutil.copy(src_img, dest_img)
            copied_images[split] += 1
        else:
            print(f"Aviso: Imagem não encontrada -> {src_img}")

        # Copiar Label (se a imagem possuir fraturas/anotações em txt)
        if os.path.exists(src_txt):
            shutil.copy(src_txt, dest_txt)
            copied_labels[split] += 1

    print("\n--- Processo concluído ---")
    print(f"Treino   -> {copied_images['train']} imagens | {copied_labels['train']} rótulos")
    print(f"Validação -> {copied_images['val']} imagens | {copied_labels['val']} rótulos")
    print(f"Teste     -> {copied_images['test']} imagens | {copied_labels['test']} rótulos")

# Executar a função com os caminhos da sua árvore de diretórios
if __name__ == "__main__":
    split_grazpedwri_dataset()