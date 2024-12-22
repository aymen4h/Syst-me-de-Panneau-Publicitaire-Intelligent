import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from picamera2 import Picamera2
import os
import requests
import base64
import cv2
import time
import subprocess
import vlc

#camera = Picamera2()
#cam_config = camera.create_still_configuration(main={"size": (1920, 1080)})
#camera.configure(cam_config)
#camera.start()
#camera.capture_file("img.jpg")
#camera.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new_model = models.mobilenet_v2(pretrained=True)

for param in new_model.parameters():
    param.requires_grad = False

# Modify the final classification layer for binary classification
num_classes = 1
new_model.classifier = nn.Sequential(
    nn.Linear(new_model.last_channel, 128),  # Intermediate dense layer
    nn.ReLU(),                              # ReLU activation
    nn.Dropout(0.4),                        # Dropout to prevent overfitting
    nn.Linear(128, num_classes),            # Output layer for a single class (binary)
    nn.Sigmoid()                            # Sigmoid activation
)

# Move the model to the selected device (GPU or CPU)
new_model.to(device)

# Load the trained model weights
new_model.load_state_dict(torch.load('mobilenet.pth', map_location=device))

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

for i in range(1,4):
     
     image_path = f"imm{i}.jpg"
     image_cv = cv2.imread(image_path)
     #cv2.imshow("gg",image_cv)
     #cv2.waitKey(0)
     cv2.destroyAllWindows()
     image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
     image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
     _, buffer = cv2.imencode('.jpg', image_rgb)
     image_base64 = base64.b64encode(buffer).decode('utf-8')

     input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

     # Set the model to evaluation mode
     new_model.eval()

     # Perform prediction
     with torch.no_grad():
          output = new_model(input_tensor)  # Get model output
          print(f"Model output: {output.item()}")

          prediction = (output >= 0.5).int()  # Convert output to binary (0 or 1)

     print(f"Predicted class: {prediction.item()}")
     API_URL = f"http://192.168.1.12:8000/get-video/"
     data = {
          'image': image_base64,
          'rain': output.item()
     }
     SAVE_DIR = "videoss"

     OUTPUT_FILENAME = "downloaded_video1.mp4"

     if not os.path.exists(SAVE_DIR):
          os.makedirs(SAVE_DIR)

     try:
          print("Downloading video...")
          response = requests.post(API_URL, data=data, stream=True)
          response.raise_for_status()  # Vérifier si la requête a réussi

          output_path = os.path.join(SAVE_DIR, OUTPUT_FILENAME)

          with open(output_path, "wb") as video_file:
               for chunk in response.iter_content(chunk_size=8192):  # Lire par blocs
                    if chunk:
                         video_file.write(chunk)

          print(f"Video saved successfully in '{output_path}'")
          subprocess.run(["vlc", output_path, "--play-and-exit", "--fullscreen",], shell=False)
          #, "--play-and-exit"
     
          os.remove(output_path)

     except requests.exceptions.RequestException as e:
          print(f"Failed to download video: {e}")
#os.remove("img.jpg")
