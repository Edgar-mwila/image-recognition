import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from cnnArcithecture import SimpleCNN

# Instantiate your model
model = SimpleCNN()

# Load the trained parameters into the model
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set the model to evaluation mode


# Define the transformation to be applied to the input image
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

# Define the class names for CIFAR-10
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# Function to make predictions
def predict(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]


# Streamlit UI
st.title("Image Recognition")
st.write("This is a simple image recognition app using a trained CIFAR-10 model.")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Perform predictions on the uploaded image
    image = Image.open(uploaded_file)
    st.write("")
    st.write("Classifying...")

    # Make prediction and display the result
    prediction = predict(image)
    st.write(f"Classification: {prediction}")
