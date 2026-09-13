from ultralytics import YOLO

def evaluate_model():
    # 1. Carregar o melhor modelo salvo do treinamento
    best_weights_path = "./runs/detect/runs/detect/grazpedwri_fracture_exp1/weights/best.pt"  # Caminho para o melhor modelo salvo
    model = YOLO(best_weights_path)

    # 2. Executar a validação forçando o split='test'
    metrics = model.val(
        data="dataset.yaml",        # Arquivo de configuração
        split="test",               # Força a avaliação no conjunto de TESTE
        imgsz=1024,                 # Mesma resolução usada no treino
        batch=8,
        device=0,                   # GPU
        project="runs/val",
        name="test_results",
        save_json=True              # Salva métricas em formato JSON se precisar exportar para tabelas do TCC
    )

    print("\n================ RESULTADOS DO TESTE ================")
    print(f"mAP@50    (IoU 0.50):    {metrics.box.map50:.4f}")
    print(f"mAP@50-95 (IoU 0.50:0.95): {metrics.box.map:.4f}")
    print(f"Precision (Precisão):      {metrics.box.mp:.4f}")
    print(f"Recall    (Sensibilidade): {metrics.box.mr:.4f}")
    print("======================================================")

if __name__ == "__main__":
    evaluate_model()