import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from vision_utils import run_yolo, run_florence, overlay_ocr_regions

# Page configuration
st.set_page_config(
    page_title="Advanced Computer Vision Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4b4b, #ff7676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Next-Gen Computer Vision Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Harnessing YOLO11 & Microsoft Florence-2 for Advanced Object Analytics, Segmentation, OCR & Visual QA.</div>', unsafe_allow_html=True)

# Sidebar layout
st.sidebar.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
st.sidebar.title("Configuration & Tasks")
task = st.sidebar.selectbox(
    "Choose CV Domain",
    [
        "Object Detection & Segmentation (YOLO11)",
        "OCR & Visual Q&A (Florence-2)",
        "Image Enhancement Studio (OpenCV)"
    ]
)

uploaded_file = st.sidebar.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Grid columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Processed Output")
        
        # Domain 1: YOLO11
        if task == "Object Detection & Segmentation (YOLO11)":
            yolo_mode = st.sidebar.radio("YOLO Task Type", ["Detection", "Segmentation"])
            conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
            iou = st.sidebar.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
            
            task_arg = "segment" if yolo_mode == "Segmentation" else "detect"
            
            with st.spinner("Running YOLO11 inference..."):
                output_image, summary = run_yolo(image, task_type=task_arg, conf=conf, iou=iou)
                
            st.image(output_image, use_container_width=True)
            
            if summary:
                st.write("### Detection Summary")
                df = pd.DataFrame(summary)
                st.dataframe(df.value_counts("label").reset_index(name="Count"))
            else:
                st.info("No objects detected above threshold.")
                
        # Domain 2: Florence-2
        elif task == "OCR & Visual Q&A (Florence-2)":
            florence_task = st.sidebar.selectbox(
                "Select Model Task",
                [
                    "Extract Text (OCR)",
                    "Extract Text with Bounding Boxes (OCR + Region)",
                    "Visual Question Answering (VQA)",
                    "Detailed Image Captioning"
                ]
            )
            
            vqa_query = ""
            if florence_task == "Visual Question Answering (VQA)":
                vqa_query = st.sidebar.text_input("Ask a question about the image:", placeholder="e.g., What is written on the sign?")
                
            run_btn = st.sidebar.button("Run Inference")
            
            if run_btn:
                task_prompt_map = {
                    "Extract Text (OCR)": "<OCR>",
                    "Extract Text with Bounding Boxes (OCR + Region)": "<OCR_WITH_REGION>",
                    "Visual Question Answering (VQA)": "<VQA>",
                    "Detailed Image Captioning": "<DETAILED_CAPTION>"
                }
                
                prompt = task_prompt_map[florence_task]
                
                with st.spinner("Processing with Microsoft Florence-2... (First run might download weights)"):
                    result = run_florence(image, prompt, vqa_query)
                
                if florence_task == "Extract Text (OCR)":
                    text_out = result.get("<OCR>", "")
                    st.text_area("Extracted Text:", text_out, height=300)
                    st.download_button("Download Text", text_out, file_name="extracted_text.txt")
                    
                elif florence_task == "Extract Text with Bounding Boxes (OCR + Region)":
                    st.write("Text highlighted in red bounding boxes:")
                    highlighted_image = overlay_ocr_regions(image, result)
                    st.image(highlighted_image, use_container_width=True)
                    st.json(result)
                    
                elif florence_task == "Visual Question Answering (VQA)":
                    st.write("### Answer:")
                    st.success(result.get("<VQA>", "Could not answer query"))
                    
                elif florence_task == "Detailed Image Captioning":
                    st.write("### Image Caption:")
                    st.info(result.get("<DETAILED_CAPTION>", ""))
            else:
                st.warning("Please click 'Run Inference' in the sidebar to start.")
                
        # Domain 3: OpenCV Enhancements
        elif task == "Image Enhancement Studio (OpenCV)":
            brightness = st.sidebar.slider("Brightness", -100, 100, 0)
            contrast = st.sidebar.slider("Contrast multiplier", 0.5, 3.0, 1.0, 0.1)
            blur_val = st.sidebar.slider("Gaussian Blur size", 1, 31, 1, 2) # must be odd
            
            filter_type = st.sidebar.selectbox(
                "Filter Effect",
                ["None", "Grayscale", "Canny Edges", "Sepia", "Sketch Effect"]
            )
            
            # Convert PIL image to CV2 (numpy array)
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 1. Adjust brightness & contrast
            img_cv = cv2.convertScaleAbs(img_cv, alpha=contrast, beta=brightness)
            
            # 2. Gaussian Blur
            if blur_val > 1:
                img_cv = cv2.GaussianBlur(img_cv, (blur_val, blur_val), 0)
                
            # 3. Filter Application
            if filter_type == "Grayscale":
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            elif filter_type == "Canny Edges":
                img_cv = cv2.Canny(img_cv, 100, 200)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            elif filter_type == "Sepia":
                kernel_sepia = np.array([
                    [0.272, 0.534, 0.131],
                    [0.349, 0.686, 0.168],
                    [0.393, 0.769, 0.189]
                ])
                # CV2 BGR to RGB sepia mapping
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_sepia = cv2.transform(img_rgb, kernel_sepia)
                img_sepia = np.clip(img_sepia, 0, 255).astype(np.uint8)
                img_cv = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
            elif filter_type == "Sketch Effect":
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                inv_gray = 255 - gray
                blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
                sketch = cv2.divide(gray, 255 - blurred, scale=256)
                img_cv = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
            else:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
            # Render final output image
            st.image(img_cv, use_container_width=True)

else:
    st.info("👈 Please upload an image using the sidebar to begin analyzing!")
