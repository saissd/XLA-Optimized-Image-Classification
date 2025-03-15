from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

app = FastAPI()

# Load model
model = models.resnet18(pretrained=True)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    return {"predicted_class": predicted_class}
