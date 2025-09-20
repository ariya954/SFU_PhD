import torch
from model import PyNET  # model.py defines the architecture

# Load model
pynet = PyNET(level=5)  # You might need to specify the model level, e.g., PyNET(level=5)
pynet.eval()

# Load pre-trained weights
checkpoint = torch.load('pynet_level5.pth', map_location='cpu')  # Replace with actual .pth file path
pynet.load_state_dict(checkpoint)

from PIL import Image
from torchvision import transforms

# Preprocess input
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),  # Match training input size
    transforms.ToTensor()
])

img = Image.open('sample.jpg').convert('RGB')
input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    output = pynet(input_tensor)

# Save or view output
from torchvision.utils import save_image
save_image(output, 'enhanced.jpg')

# extract parameters (weights and biases) from each layer
for name, module in pynet.named_modules():
    if hasattr(module, 'weight'):
        print(f"\nLayer: {name}")
        print(f"  Weight shape: {module.weight.shape}")
        if hasattr(module, 'bias') and module.bias is not None:
            print(f"  Bias shape: {module.bias.shape}")
