import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from PIL import Image as Image_pil

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_gaze(frame, model, device):
    image = Image_pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)

    predicted_output = output.cpu().numpy()
    predicted_label = np.argmax(predicted_output, axis=1)
    return predicted_label