import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
from gradcam import VizGradCAM
import cv2
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os

#  Page Config 
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="SRC/logo.png",
    layout="wide"
)


#  Load Model 
@st.cache_resource
def load_my_model():
    return load_model("DenseNet_model.keras")

model = load_my_model()

with open('SRC/sa.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


#  Tumor Info & Recommendations 
tumor_info = {
    "glioma": {
        "summary": "Gliomas are tumors that occur in the brain and spinal cord, originating in glial cells.",
        "suggestions": [
            "Consult a neurologist or oncologist for further evaluation.",
            "Follow recommended MRI or CT scan follow-ups regularly.",
            "Maintain a healthy diet rich in antioxidants.",
            "Avoid smoking and excessive alcohol consumption.",
            "Manage stress through yoga or meditation."
        ]
    },
    "meningioma": {
        "summary": "Meningiomas are tumors that arise from the meninges, the membranes surrounding your brain and spinal cord. Most are benign but may cause pressure symptoms.",
        "suggestions": [
            "Schedule an appointment with a neurosurgeon for treatment options.",
            "Report any new headaches, vision issues, or seizures immediately.",
            "Avoid strenuous physical activity without medical approval.",
            "Prioritize sleep and stress management.",
            "Regular check-ups are essential to monitor growth."
        ]
    },
    "normal": {
        "summary": "No abnormal growth detected. The brain scan appears normal.",
        "suggestions": [
            "Continue routine health checkups.",
            "Maintain a balanced diet and regular exercise.",
            "Stay hydrated and prioritize mental wellness.",
            "No special restrictions necessary."
        ]
    },
    "pituitary": {
        "summary": "Pituitary tumors develop in the pituitary gland, which controls vital hormone functions in your body.",
        "suggestions": [
            "Consult an endocrinologist for hormone level assessment.",
            "Follow prescribed hormonal treatments if necessary.",
            "Report symptoms like vision changes, fatigue, or irregular periods (for women).",
            "Avoid self-medicating and maintain regular follow-ups."
        ]
    }
}

#  Hero Section 
st.markdown("""
<div class="hero">
    <h1>Brain Tumor Detection</h1>
    <p>AI-powered system to analyze MRI scans and assist in early tumor detection</p>
</div>
""", unsafe_allow_html=True)

# Upload Section 
col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Upload MRI Image")
    uploadedIMG = st.file_uploader("", type=["png", "jpg", "jpeg", "heif", "heic"])

with col2:
    st.subheader("🌐 Upload via Link")
    url = st.text_input("Paste image URL here")

#  Image Handling 
img = None
if uploadedIMG is not None:
    st.image(uploadedIMG, caption="Uploaded MRI Image", use_container_width=False, width= 400)
    img = Image.open(uploadedIMG).convert("RGB")

elif url:
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")  
        st.image(img, caption="Image from Link", use_container_width=False, width= 400)
    except:
        st.error("Could not load image from the provided link.")

#  Prediction 
if img:
    with st.spinner("🔍 Analyzing MRI scan... please wait..."):
        img = img.resize((240, 240))
        img_array = image.img_to_array(img)

        # GradCAM
        GradCamImage = VizGradCAM(model, img_array, plot_results=False)
        heatmap = GradCamImage - np.min(GradCamImage)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        orig = image.img_to_array(img.resize((240,240)))
        superimposed_img = cv2.addWeighted(orig.astype(np.uint8), 1, heatmap, 0.5, 0)

        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        class_idx = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred) * 100
        classes = ['glioma', 'meningioma', 'normal', 'pituitary']
        predicted_label = classes[class_idx]


        col1, col2 = st.columns(2)
        col1.image(img, caption="Original MRI Image", use_container_width=False, width= 500)
        col2.image(superimposed_img, caption="Grad-CAM Heatmap",use_container_width=False, width= 500)

        st.success(f"### Predicted Label: {classes[class_idx].upper()}")

         # Detailed Report 
        info = tumor_info[predicted_label]
        st.markdown("# MRI Report Summary")
        st.write(info["summary"])

        st.markdown("# Recommendations & Next Steps")
        for sug in info["suggestions"]:
            st.markdown(f"- {sug}")

        # --- Generate PDF Report ---
        if st.button("📥 Download Report as PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                pdf_path = tmpfile.name
                doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                styles = getSampleStyleSheet()
                elements = []

                elements.append(Paragraph("<b>Brain Tumor Detection Report</b>", styles['Title']))
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(f"<b>Prediction:</b> {predicted_label.upper()} ({confidence:.2f}%)", styles['Normal']))
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("<b>Summary:</b>", styles['Heading2']))
                elements.append(Paragraph(info["summary"], styles['Normal']))
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("<b>Recommendations:</b>", styles['Heading2']))
                for sug in info["suggestions"]:
                    elements.append(Paragraph(f"- {sug}", styles['Normal']))
                elements.append(Spacer(1, 10))

                # Save images temporarily
                orig_path = os.path.join(tempfile.gettempdir(), "original_img.jpg")
                heatmap_path = os.path.join(tempfile.gettempdir(), "heatmap_img.jpg")
                Image.fromarray(orig.astype(np.uint8)).save(orig_path)
                Image.fromarray(superimposed_img).save(heatmap_path)

                elements.append(Paragraph("<b>Uploaded MRI Image:</b>", styles['Heading2']))
                elements.append(RLImage(orig_path, width=180, height=180))
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("<b>Grad-CAM Heatmap:</b>", styles['Heading2']))
                elements.append(RLImage(heatmap_path, width=180, height=180))

                doc.build(elements)

                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download Report", f, file_name="Brain_Tumor_Report.pdf", mime="application/pdf")
