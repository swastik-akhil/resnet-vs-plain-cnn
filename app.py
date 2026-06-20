import streamlit as st
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image

from models import SimpleCNN

st.set_page_config(
    page_title="CNN vs ResNet18",
    layout="centered"
)

cnn = SimpleCNN()

cnn.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location="cpu"
    )
)

cnn.eval()

resnet = models.resnet18()

resnet.fc = torch.nn.Linear(
    resnet.fc.in_features,
    10
)

resnet.load_state_dict(
    torch.load(
        "resnet_model.pth",
        map_location="cpu"
    )
)

resnet.eval()

classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

st.title("CNN vs ResNet18 Image Classifier")

st.write(
    
    """
    This project compares a traditional Convolutional Neural Network (CNN)
    and ResNet18 on the CIFAR-10 dataset.

    Upload an image belonging to one of the CIFAR-10 classes and compare
    the predictions produced by both models.

    Supported classes: airplane, automobile, bird, cat,
    deer, dog, frog, horse, ship, truck.
    
    Images belonging to other categories will still be assigned to the closest available class.


    """

    # st.markdown("---")

)

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image)

    image = transform(image)
    image = image.unsqueeze(0)


    cnn_output = cnn(image)

    cnn_pred = torch.argmax(
        cnn_output,
        dim=1
    ).item()


    resnet_output = resnet(image)

    resnet_pred = torch.argmax(
        resnet_output,
        dim=1
    ).item()


    st.subheader("CNN Prediction")

    st.write(
        classes[cnn_pred]
    )

    st.subheader("ResNet Prediction")

    st.write(
        classes[resnet_pred]
    )