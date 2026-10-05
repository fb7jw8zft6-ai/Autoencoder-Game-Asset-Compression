from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from PIL import Image

import torch
import math
import io
import base64
import os

from torchvision import transforms

from model import GameAssetAutoencoder


app = Flask(__name__)

CORS(app)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model = GameAssetAutoencoder()

model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "game_asset_autoencoder.pth"),
        map_location="cpu"
    )
)

model.eval()


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "message": "Game Asset Autoencoder Backend is Running!"
    })


# --------------------------------------------------
# RECONSTRUCTION ROUTE
# --------------------------------------------------

@app.route("/reconstruct", methods=["POST"])
def reconstruct():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    file = request.files["image"]

    original_image = Image.open(file).convert("RGB")

    input_tensor = transform(
        original_image
    ).unsqueeze(0)


    # --------------------------------------------------
    # AUTOENCODER PROCESS
    # --------------------------------------------------

    with torch.no_grad():

        # Encode image
        latent = model.encoder(
            input_tensor
        )


        # --------------------------------------------------
        # LATENT QUANTIZATION
        # --------------------------------------------------

        latent_min = latent.min()

        latent_max = latent.max()


        latent_normalized = (
            (latent - latent_min)
            /
            (latent_max - latent_min + 1e-8)
        )


        latent_uint8 = (
            latent_normalized * 255
        ).round().to(torch.uint8)


        # Restore latent representation

        latent_restored = (
            latent_uint8.float()
            / 255.0
        )


        latent_restored = (
            latent_restored
            *
            (latent_max - latent_min)
            +
            latent_min
        )


        # Decode image

        reconstructed = model.decoder(
            latent_restored
        )


    # --------------------------------------------------
    # CALCULATE MSE
    # --------------------------------------------------

    mse = torch.mean(
        (input_tensor - reconstructed) ** 2
    ).item()


    # --------------------------------------------------
    # CALCULATE PSNR
    # --------------------------------------------------

    if mse == 0:

        psnr = float("inf")

    else:

        psnr = 10 * math.log10(
            1.0 / mse
        )


    # --------------------------------------------------
    # COMPRESSION SIZE
    # --------------------------------------------------

    original_size_bytes = 64 * 64 * 3

    latent_size_bytes = latent_uint8.numel()

    compression_ratio = (
        original_size_bytes
        /
        latent_size_bytes
    )


    # --------------------------------------------------
    # CONVERT OUTPUT TO IMAGE
    # --------------------------------------------------

    output_tensor = (
        reconstructed
        .squeeze(0)
        .clamp(0, 1)
    )


    output_image = transforms.ToPILImage()(
        output_tensor
    )


    # --------------------------------------------------
    # CONVERT IMAGE TO BASE64
    # --------------------------------------------------

    buffer = io.BytesIO()

    output_image.save(
        buffer,
        format="PNG"
    )


    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


    # --------------------------------------------------
    # SEND RESULT
    # --------------------------------------------------

    return jsonify({

        "message":
            "Image reconstructed successfully",

        "reconstructed_image":
            "data:image/png;base64,"
            + encoded_image,

        "input_size":
            "64 × 64 RGB",

        "latent_shape":
            list(latent.shape),

        "latent_values":
            int(latent.numel()),

        "original_size_bytes":
            original_size_bytes,

        "latent_size_bytes":
            latent_size_bytes,

        "compression_ratio":
            round(
                compression_ratio,
                2
            ),

        "mse":
            round(
                mse,
                6
            ),

        "psnr":
            round(
                psnr,
                2
            )
    })


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=True
    )