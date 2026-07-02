# fast api application for x-ray image classification

import torch
from x_ray.ml.model.arch import Net  # Ensure this path is correct
from torchvision.transforms import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File

# Initialize fastapi app
app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load model weights
model = Net().to(device)
model.load_state_dict(torch.load("xray_model.pth", map_location=device))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prediction label mapping
label_map = {
    0: "Normal",
    1: "Pneumonia"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file and convert it to RGB format
    image = Image.open(file.file).convert("RGB")
    # Apply transformation and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction_index = torch.argmax(output, dim=1).item()
        prediction_label = label_map.get(prediction_index, "Unknown")
    
    return {
        "prediction_index": prediction_index,
        "prediction_label": prediction_label
    }
