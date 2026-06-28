import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
import streamlit as st

# Global configurations / Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

@st.cache_resource
def load_yolo_model(task_type="detect"):
    """
    Loads the YOLO11 model based on detection or segmentation.
    yolo11n.pt = Nano Object Detection
    yolo11n-seg.pt = Nano Instance Segmentation
    """
    if task_type == "segment":
        return YOLO("yolo11n-seg.pt")
    return YOLO("yolo11n.pt")

@st.cache_resource
def load_florence_model():
    """
    Loads Microsoft Florence-2-base model and processor.
    """
    model_id = "microsoft/Florence-2-base"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch_dtype).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

def run_yolo(image: Image.Image, task_type="detect", conf=0.25, iou=0.45):
    """
    Runs YOLO11 model on PIL Image.
    """
    model = load_yolo_model(task_type)
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model.predict(img_cv, conf=conf, iou=iou, device=device)
    
    # Plot results on the image
    annotated_img = results[0].plot()
    # Convert BGR back to PIL
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    
    # Get predictions summary
    boxes = results[0].boxes
    summary = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            conf_val = float(box.conf[0].item())
            summary.append({"label": cls_name, "confidence": conf_val})
            
    return annotated_pil, summary

def run_florence(image: Image.Image, task_prompt: str, text_input: str = None):
    """
    Runs Microsoft Florence-2 inference based on task prompts:
    - OCR: '<OCR>'
    - OCR with Region: '<OCR_WITH_REGION>'
    - Detailed Caption: '<DETAILED_CAPTION>'
    - VQA: '<VQA>'
    """
    model, processor = load_florence_model()
    
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    if task_prompt == "<VQA>" and text_input:
        prompt = f"<VQA> {text_input}"
    else:
        prompt = task_prompt
        
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device).to(torch_dtype) if v.dtype == torch.float else v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)
    
    return parsed_answer

def overlay_ocr_regions(image: Image.Image, ocr_results):
    """
    Draws bounding boxes and text labels for '<OCR_WITH_REGION>'.
    ocr_results is expected to be a dict e.g., {'<OCR_WITH_REGION>': {'quad_boxes': [...], 'labels': [...]}}
    """
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    data = ocr_results.get("<OCR_WITH_REGION>", {})
    quad_boxes = data.get("quad_boxes", [])
    labels = data.get("labels", [])
    
    # Draw quad boxes
    for box, label in zip(quad_boxes, labels):
        # quad_boxes are usually 8 values: [x1, y1, x2, y2, x3, y3, x4, y4]
        if len(box) == 8:
            points = [(box[i], box[i+1]) for i in range(0, 8, 2)]
            draw.polygon(points, outline="red", width=2)
            # Draw label background
            draw.text(points[0], label, fill="blue")
            
    return draw_image
