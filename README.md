# Next-Gen Computer Vision Studio

An advanced, interactive Computer Vision playground built with Streamlit, incorporating state-of-the-art vision intelligence models like **YOLO11** and **Microsoft Florence-2** alongside OpenCV filters.

## Features

1. **Object Detection & Segmentation (YOLO11)**:
   - Detect 80+ object categories in real-time or from static uploads.
   - Run precise Instance Segmentation to trace object boundaries.
   - Adjust confidence and IoU thresholds dynamically.

2. **OCR & Visual Q&A (Microsoft Florence-2)**:
   - Extract raw text (OCR) from documents, receipts, or scenery.
   - Highlight bounding boxes around read characters.
   - Ask complex questions about image content using Visual Question Answering (VQA).
   - Generate detailed image captions automatically.

3. **Image Enhancement Studio (OpenCV)**:
   - Apply real-time sliding scale adjustments for Brightness, Contrast, and Gaussian Blur.
   - Instantly apply filters: Grayscale, Canny Edges, Sepia, and Sketch effects.

---

## Getting Started Locally

### 1. Clone & Setup
```bash
# Navigate to the project directory
cd cv_studio

# Create a virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## Deployment to Streamlit Cloud & GitHub

### 1. Create Github Repository
Initialize a git repository locally and commit your code:
```bash
git init
git add .
git commit -m "Initial commit of CV Studio"
```

1. Go to your [GitHub account](https://github.com/) and create a new repository named `computer-vision`.
2. Link the local repository to your GitHub repository and push:
```bash
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/computer-vision.git
git push -u origin main
```

### 2. Deploy to Streamlit Community Cloud
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in using your GitHub account.
2. Click **New app**.
3. Select your repository (`computer-vision`), branch (`main`), and path to the entry file (`app.py`).
4. Click **Deploy!** Streamlit will automatically install all dependencies listed in `requirements.txt` and launch your live site.
