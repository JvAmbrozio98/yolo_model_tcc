from ultralytics import YOLO

def start_training():
    # 1. Carregar o modelo base pré-treinado
    # yolo11s.pt ou yolov8s.pt: ideal para 6GB VRAM (equilíbrio entre velocidade e precisão)
    # yolo11m.pt ou yolov8m.pt: maior capacidade de extração (se o batch for reduzido para 4)
    model = YOLO("yolo11s.pt")  # Pode alternar para 'yolov8s.pt' se preferir YOLOv8

    # 2. Iniciar o treinamento
    results = model.train(
        data="dataset.yaml",          # Caminho para o seu dataset.yaml
        epochs=30,                  # Número total de épocas
        imgsz=1024,                  # Resolução da imagem (1024 preserva detalhes de microfraturas)
        batch=8,                     # Tamanho do lote (se der Out Of Memory, altere para 4)
        device=0,                    # 0 para usar a GPU dedicada
        workers=4,                   # Número de threads para carregar imagens
        project="runs/detect",       # Diretório onde os resultados serão salvos
        name="grazpedwri_fracture_exp1", # Nome do experimento
        save=True,                   # Salva checkpoints (.pt)
        save_period=10,              # Salvar pesos a cada 10 épocas
        patience=15,                 # Early Stopping: para se não houver melhora em 15 épocas
        amp=True,                    # Automatic Mixed Precision (acelera o treino na GPU e economiza VRAM)
        
        # Augmentations ajustadas para radiografias
        degrees=10.0,                # Rotação leve (+/- 10 graus)
        translate=0.05,              # Translação leve
        scale=0.1,                   # Escala leve
        fliplr=0.5,                  # Espelhamento horizontal
        mosaic=1.0,                  # Mosaic augmentation ativado
    )

    print("\n--- Treinamento Concluído ---")
    print(f"Melhor modelo salvo em: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    start_training()