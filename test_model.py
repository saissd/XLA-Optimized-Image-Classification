import torch
import torchvision.models as models

def test_model():
    try:
        model = models.resnet18(pretrained=True)
        model.eval()
        print("✅ ResNet-18 Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Model Loading Failed: {e}")
        exit(1)

if __name__ == "__main__":
    test_model()
