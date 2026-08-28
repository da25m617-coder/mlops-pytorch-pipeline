import io
import os
import sys
from pathlib import Path

# Add src folder to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, HTTPException, UploadFile
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import get_model

app = FastAPI(title="Fashion-MNIST ResNet-18 API")

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/checkpoints/classifier_v1.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2860,), std=(0.3530,))
])

def load_checkpoint():
    global model
    if not MODEL_PATH.exists():
        return False
    try:
        model = get_model(architecture="resnet18", num_classes=10)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_checkpoint()

@app.get("/health")
def health():
    if model is None and not load_checkpoint():
        raise HTTPException(status_code=503, detail="Model checkpoint not loaded")
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None and not load_checkpoint():
        raise HTTPException(status_code=503, detail="Model checkpoint not loaded")
        
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            top_prob, top_class_idx = torch.max(probabilities, dim=0)
            
        prob_dict = {CLASSES[i]: round(probabilities[i].item(), 4) for i in range(len(CLASSES))}
        
        return {
            "prediction": CLASSES[top_class_idx.item()],
            "confidence": round(top_prob.item(), 4),
            "probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")