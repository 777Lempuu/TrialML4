import streamlit as st
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import random

# Set up Streamlit page
st.set_page_config(page_title="STL-10 Classifier", layout="wide")
st.title("🖼️ STL-10 Image Classifier (UI Prototype)")

# Define STL10 classes
classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# Load the dataset (just a few samples)
@st.cache_data
def load_stl10_samples(num_samples=10):
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
    indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in indices]
    return samples

st.subheader("📄 Dataset Preview and Class Distribution")
st.markdown("STL-10 is an image classification dataset with 10 classes and 96x96 color images.")

# Show class distribution (since we load only some, we simulate it)
st.write("### 🏷️ STL-10 Classes:")
st.write(", ".join(classes))

# Load and display a few image-label pairs
num_preview = st.slider("How many samples to preview?", 1, 20, 5)
samples = load_stl10_samples(num_preview)

st.write("### 🖼️ Random Sample Images:")
cols = st.columns(min(5, num_preview))
for i, (image_tensor, label) in enumerate(samples):
    img = transforms.ToPILImage()(image_tensor)
    with cols[i % len(cols)]:
        st.image(img, caption=f"Label: {classes[label]}", use_column_width=True)

st.markdown("---")

# User input section (no model connected yet)
st.subheader("🤖 Try It Yourself: Image Classification")

st.write("You can upload an image and we’ll use it for prediction (model to be added later).")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.info("Model prediction will appear here in the future.")

# Simulated user guess (for fun/feedback)
user_guess = st.radio(
    "Which class do you think it belongs to?",
    ["Select"] + classes,
    index=0
)

user_confidence = st.slider("How confident are you?", 1, 5, 3)

if uploaded_file and user_guess != "Select":
    st.write(f"Your guess: **{user_guess}** with confidence level **{user_confidence}/5**")
