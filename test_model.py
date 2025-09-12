#!/usr/bin/env python3
"""
Script de teste para validar o modelo de detecção de danos.
"""

import os
import sys
from PIL import Image
import numpy as np

def test_model_loading():
    """Testa se o modelo pode ser carregado."""
    try:
        from ultralytics import YOLO
        model_path = 'car_damage_best.pt'
        
        if not os.path.exists(model_path):
            print(f"❌ Modelo '{model_path}' não encontrado!")
            return False
            
        print("🔄 Carregando modelo...")
        model = YOLO(model_path)
        print("✅ Modelo carregado com sucesso!")
        
        print(f"📊 Classes do modelo: {model.names}")
        print(f"📏 Número de classes: {len(model.names)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

def test_image_processing():
    """Testa o processamento de uma imagem sintética."""
    try:
        from ultralytics import YOLO
        
        # Criar uma imagem sintética para teste
        test_image = Image.new('RGB', (640, 480), color='blue')
        img_array = np.array(test_image)
        
        model = YOLO('car_damage_best.pt')
        results = model(img_array)
        
        print("✅ Processamento de imagem funcionando!")
        print(f"📊 Número de detecções: {len(results[0].boxes) if len(results[0].boxes) > 0 else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        return False

def main():
    """Função principal de teste."""
    print("🚗 Teste do Sistema de Detecção de Danos")
    print("=" * 50)
    
    # Teste 1: Carregamento do modelo
    print("\n1. Testando carregamento do modelo...")
    if not test_model_loading():
        print("❌ Falha no teste de carregamento!")
        sys.exit(1)
    
    # Teste 2: Processamento de imagem
    print("\n2. Testando processamento de imagem...")
    if not test_image_processing():
        print("❌ Falha no teste de processamento!")
        sys.exit(1)
    
    print("\n✅ Todos os testes passaram!")
    print("🎉 Sistema pronto para uso!")

if __name__ == "__main__":
    main()
