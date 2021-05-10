import os
import torch
from PIL import Image, UnidentifiedImageError
import requests
from street_view import StreetViewer
import matplotlib.pyplot as plt

import streamlit as st
from model_utils import generator, get_transform, tensor2im


@st.cache()
def load_model():
    """Retrieves the generator model."""
    model = generator(
        3, 3, 64, "instance", True, "xavier", 0.02, False, False, None, []
    )

    load_path = os.path.join("models", "generator_latest.pth")
    state_dict = torch.load(load_path, map_location=str("cpu"))
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    model.load_state_dict(state_dict)

    return model

@st.cache(show_spinner=False)
def lightify(img, model):
    transform = get_transform()
    new_img = transform(img)

    with torch.no_grad():
        output = model(new_img.unsqueeze(0))

    arr = tensor2im(output)
    return arr


if __name__ == "__main__":
    st.title("Christmaslightify")
    st.subheader(
        "Miss the holidays? Want some Christmas cheer? Either enter a street address or upload an image (ideally of a house) to experience some Christmas magic."
    )
    model = load_model()
    address = st.text_input('Place or Street Address to be Lit Up', "")
    if address:
        street_viewer = StreetViewer(api_key=st.secrets["API_KEY"],
                                   location=address)
        street_viewer.get_meta()
        try:
            col1, col2 = st.beta_columns(2)
            img = street_viewer.get_pic()
            if img:
                resized_image = img.resize((256, 256))
                col1.header("December 24th")
                col1.image(resized_image)
                with st.spinner("Processing..."):
                    arr = lightify(img, model)
                # st.subheader("Here it is with some Christmas magic")
                col2.header("December 25th")
                col2.image(arr)
            else:
                st.write("We cannot process your address... looks like you're getting coal for Christmas.")
        except (UnidentifiedImageError):
            st.subheader("We cannot process your address... looks like you're getting coal for Christmas.")

    file = st.file_uploader("Or upload an image to be Lit Up")

    if file:
        try:
            col1, col2 = st.beta_columns(2)
            img = Image.open(file).convert('RGB')
            # st.subheader("Here is the house you've selected")
            resized_image = img.resize((256, 256))
            col1.header("December 24th")
            col1.image(resized_image)
            with st.spinner("Processing..."):
                arr = lightify(img, model)
            # st.subheader("Here it is with some Christmas magic")
            col2.header("December 25th")
            col2.image(arr)
        except (UnidentifiedImageError):
            st.write("We cannot process your image file... looks like you're getting coal for Christmas.")
