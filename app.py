import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ===== UNet Architecture (Kaggle training code se copy — bilkul same hona zaroori hai) =====

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.encoder_blocks.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, features=[512, 256, 128, 64]):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for feature in features:
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[idx](x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = Encoder(in_channels, features)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.decoder = Decoder(features[::-1])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = self.final_conv(x)
        return x


# ===== Model Load Karna (ek hi baar, cache ke sath) =====

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load("best_unet_tversky.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device


model, device = load_model()


# ===== Preprocessing Function (bilkul training jaisa) =====

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_tensor = torch.tensor(img_array).unsqueeze(0)
    return img_tensor


# ===== Streamlit UI =====

st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor Segmentation")
st.caption("U-Net based semantic segmentation on MRI scans")

st.write(
    "Upload a brain MRI scan and the model will predict the tumor region, "
    "pixel by pixel."
)

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Running inference..."):
        input_tensor = preprocess_image(image).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob_mask = torch.sigmoid(output)
            pred_mask = (prob_mask > 0.5).float()

        pred_mask_np = pred_mask.squeeze().cpu().numpy()

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original MRI", use_container_width=True)
    with col2:
        st.image(pred_mask_np, caption="Predicted Tumor Mask", use_container_width=True, clamp=True)

    tumor_pixel_percent = (pred_mask_np.sum() / pred_mask_np.size) * 100

    if tumor_pixel_percent > 0:
        st.success(f"Predicted tumor area: {tumor_pixel_percent:.2f}% of the image")
    else:
        st.info("No tumor region detected in this scan.")

    with st.expander("About this model"):
        st.write(
            "- **Architecture:** U-Net (encoder-decoder with skip connections)\n"
            "- **Loss function:** Tversky Loss\n"
            "- **Test Dice Score:** 84.65%"
        )
else:
    st.info("Please upload an MRI image to get started.")