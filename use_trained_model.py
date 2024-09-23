import torch
import torchvision.transforms as transforms
from PIL import Image
from SimpleModel import SimpleModel  # Make sure this matches your model file

def load_model(model_path):
    model = SimpleModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, image_path):
    # Define the same transform you used for training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

def main():
    model_path = 'pth/model_checkpoint_epoch_19.pth'  # Adjust this to your saved model path
    model = load_model(model_path)

    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Example usage
    image_path = 'data/3.jpg'  # Replace with your test image path
    prediction = predict(model, image_path)
    print(f"The image is predicted to be: {classes[prediction]}")

if __name__ == "__main__":
    main()