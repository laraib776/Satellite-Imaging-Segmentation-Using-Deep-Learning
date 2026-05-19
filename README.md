```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ███████╗██╗  ██╗██╗   ██╗                                       ║
║   ██╔════╝██║ ██╔╝╚██╗ ██╔╝                                       ║
║   ███████╗█████╔╝  ╚████╔╝                                        ║
║   ╚════██║██╔═██╗   ╚██╔╝                                         ║
║   ███████║██║  ██╗   ██║                                          ║
║   ╚══════╝╚═╝  ╚═╝   ╚═╝                                          ║
║                                                                   ║
║   ██████╗ ██╗██╗  ██╗███████╗██╗                                  ║
║   ██╔══██╗██║╚██╗██╔╝██╔════╝██║                                  ║
║   ██████╔╝██║ ╚███╔╝ █████╗  ██║                                  ║
║   ██╔═══╝ ██║ ██╔██╗ ██╔══╝  ██║                                  ║
║   ██║     ██║██╔╝ ██╗███████╗███████╗                             ║
║   ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝                             ║
║                                                                   ║
║    🛰️  See the Earth differently. One pixel at a time.  🌍       ║
╚═══════════════════════════════════════════════════════════════════╝
```

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![U-Net](https://img.shields.io/badge/U--Net-Deep%20Learning-FF6B6B?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Interactive%20UI-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Powered-FFE66D?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google%20Colab-Compatible-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-C3B1E1?style=for-the-badge)

**[⭐ Star this repo](https://github.com/laraib776/Satellite-Imaging-Segmentation-Using-Deep-Learning)** · **[🐛 Report a Bug](https://github.com/laraib776/Satellite-Imaging-Segmentation-Using-Deep-Learning/issues)** · **[🤝 Contribute](#-contributing)**

</div>

---

## 🛰️ What Does the Earth Actually Look Like From Above?

> **Forests. Deserts. Cities. Rivers. Roads. All hidden in pixels.**
> SkyPixel doesn't just look at satellite images — it *understands* them, segment by segment, feature by feature.
>
> ### 👉 **Upload a satellite image. Watch the Earth reveal itself.** 👈
>
> *Powered by U-Net deep learning. Visualized through a live Gradio interface. No PhD required.*

> [!NOTE]
> SkyPixel is fully **Google Colab compatible** — no local GPU needed. Upload your notebook, run the cells, and start segmenting from the cloud instantly.

---

## ✦ About SkyPixel

> **SkyPixel** is a deep learning project built on the **U-Net architecture** to perform precise segmentation on satellite imagery. It identifies and classifies land cover features — forests, water bodies, urban areas, and more — directly from raw satellite images.
>
> Wrapped in an intuitive **Gradio + Hugging Face** interface, results are visualized in real time. No complex setup, no command-line expertise required. Just upload and see.

---

## ✨ Key Features

| 🌟 Feature | Details |
|---|---|
| 🧠 **U-Net Architecture** | Powerful convolutional network built for fast, precise image segmentation |
| 🖼️ **Interactive UI** | Gradio + Hugging Face interface for real-time upload and visualization |
| ☁️ **Colab Compatible** | Run entirely in the cloud — no local GPU required |
| 🎯 **High Accuracy** | Trained on real satellite datasets for reliable land cover classification |
| ⚡ **Real-Time Results** | Upload an image, get a segmented output instantly in your browser |

---

## 🛠️ Technology Stack

```
  ╭──────────────────┬──────────────────────────────────────────────╮
  │  Layer           │  Technology                                  │
  ├──────────────────┼──────────────────────────────────────────────┤
  │  🐍  Language    │  Python 3.x                                  │
  │  🧠  Model       │  U-Net  (Deep Learning / CNN)                │
  │  🖼️  Interface   │  Gradio  +  Hugging Face                     │
  │  ☁️  Cloud       │  Google Colab                                │
  │  📦  Model File  │  unet_model.h5  (pre-trained weights)        │
  ╰──────────────────┴──────────────────────────────────────────────╯
```

---

##  🚀 Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/laraib776/Satellite-Imaging-Segmentation-Using-Deep-Learning.git
cd skypixel
```

### Step 2 — Create & Activate a Virtual Environment

```bash
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add the Pre-trained Model

```
📥  Place  unet_model.h5  in the root project directory
```

### Step 5 — Launch the App

```bash
gradio app App_satellite_segmentation_prediction.ipynb
```

> 🌐 This launches a live web interface where you can upload satellite images and view segmented outputs instantly.

---

##  ☁️ Run on Google Colab

Prefer the cloud? No problem:

```
  1. 📤  Upload the notebook + unet_model.h5 to Google Colab
  2. ⚙️   Install required dependencies inside the notebook
  3. ▶️   Execute the cells top to bottom
  4. 🖼️  Upload a satellite image and view the segmentation live
```

> [!TIP]
> Google Colab gives you free GPU access — ideal for running inference quickly without any local hardware requirements.

---

## 🎮 Usage Guide

Once the app is running in your browser:

```
  🌍  Step 1  →  Open the Gradio interface in your browser
  📤  Step 2  →  Upload a satellite image
  ⚡  Step 3  →  The model processes and segments the image
  🗺️  Step 4  →  View the colour-coded segmentation output
```

---

##  📁 Project Structure

```
📦 SkyPixel/
 │
 ├── 📄 App_satellite_segmentation_prediction.ipynb  ← Main Gradio app
 ├── 📄 README.md                                    ← You are here 👋
 ├── 📄 requirements.txt                             ← All dependencies
 ├── 🧠 unet_model.h5                                ← Pre-trained U-Net weights
 │
 ├── 📂 dataset/                    ← Satellite training images
 │    └── 🖼️  images/  ·  masks/
 │
 └── 📂 outputs/                    ← Segmentation result previews
```

---

## 🤝 Contributing

Contributions are always welcome and appreciated! 💖

```
  1. 🍴  Fork the repository
  2. 🌿  Create your feature branch
  3. 💾  Commit your changes
  4. 📬  Open a Pull Request
```

Ideas we'd love to see: improved model accuracy, support for new satellite datasets, multi-class segmentation overlays, or a cleaner Gradio UI — all PRs are warmly welcome!

---

##  📜 License

This project is open source under the **MIT License** — free to use, modify, and share.

---

<div align="center">

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   From space, everything looks like a pattern.             ║
║   SkyPixel makes sure you see every single one.  🛰️🌍     ║
║                                                            ║
║               Made with ❤️  by  Laraib Khalid              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

*⭐ Drop a star if SkyPixel helped you see the Earth differently!*

</div>
