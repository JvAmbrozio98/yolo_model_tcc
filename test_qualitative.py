from ultralytics import YOLO
import os

def run_predictions():
    # Carregar o melhor modelo
    best_weights_path = "./runs/detect/runs/detect/grazpedwri_fracture_exp1/weights/best.pt"
    model = YOLO(best_weights_path)

    # Rodar a inferência sobre as imagens da pasta de TESTE
    results = model.predict(
        source="dataset/folder_structure/yolo_dataset/images/test", # Pasta de imagens do teste
        imgsz=1024,
        conf=0.25,        # Threshold de confiança (desenha caixas com > 25% de certeza)
        iou=0.45,         # Threshold de NMS (evita caixas duplicadas na mesma fratura)
        save=True,        # Salvar as imagens com as caixas desenhadas
        save_txt=True,    # Salvar as coordenadas das caixas preditas em txt
        save_conf=True,   # Salvar a probabilidade de cada predição
        project="runs/predict",
        name="test_predictions",
        device=0
    )

    print(f"\nPredições concluídas! As imagens com os retângulos desenhados foram salvas em:")
    print("runs/predict/test_predictions")

if __name__ == "__main__":
    run_predictions()