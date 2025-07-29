import streamlit as st
import os
import shutil
import pandas as pd
import re
from main import run_fewshot_pipeline

# ------------ CONFIG ------------
TEMP_ROOT = "data_temp"
SUPPORT_ROOT = os.path.join(TEMP_ROOT, "support")
QUERY_ROOT = os.path.join(TEMP_ROOT, "query")
MAX_UPLOAD_MB_PER_CLASS = 25  # Limit for support+query combined
MODEL_OPTIONS = {
    'resnet18': 45,
    'resnet34': 83,
    'resnet50': 98,
    'mobilenet_v2': 14,
    'mobilenet_v3_small': 10,
    'mobilenet_v3_large': 16,
    'efficientnet_b0': 20,
    'efficientnet_b1': 32,
    'densenet121': 33,
    'densenet169': 57,
}
ACCENT = "#21409a"

# ------------ UTILS ------------
def sanitize_filename(name):
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)
    return name.strip()

def reset_dirs_and_save_images(class_data):
    for root in [SUPPORT_ROOT, QUERY_ROOT]:
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)

    for cd in class_data:
        for folder, images in zip([SUPPORT_ROOT, QUERY_ROOT], [cd['support'], cd['query']]):
            class_name = sanitize_filename(cd["name"])
            class_path = os.path.join(folder, class_name)
            os.makedirs(class_path, exist_ok=True)
            for img in images:
                safe_name = sanitize_filename(img.name)
                save_path = os.path.join(class_path, safe_name)
                try:
                    with open(save_path, "wb") as fout:
                        fout.write(img.getbuffer())
                except Exception as e:
                    st.error(f"❌ Failed to save {img.name}: {e}")

def get_total_mb(files):
    return sum([len(f.getbuffer()) for f in files]) / (1024 * 1024)

# ------------ STYLES ------------
st.markdown("""
<style>
.block-container { max-width: 1400px !important; padding: 2rem; }
.class-border {
    border:2px solid #a9b8d6; border-radius:12px; padding:18px 14px 12px 14px;
    margin-bottom:18px; background: #f9fbff;
}
hr { border: none; border-top: 1.4px solid #eef2f6; }
.stButton>button, .stDownloadButton>button {
    background-color: #21409a; color: #fff !important; border-radius: 9px; font-weight: 500;
}
.stButton>button:hover { background-color: #14306c; }
</style>
""", unsafe_allow_html=True)

# ------------ HEADER ------------
st.markdown('<h1 style="margin-bottom:0.18em;">Few-Shot Learning Model Service</h1>', unsafe_allow_html=True)
st.write("<div style='font-size:1.09em;color:#444;'><b>Upload, train, and export your few-shot classifier with just a few clicks.</b></div>", unsafe_allow_html=True)
st.markdown("---")

# ------------ CLASS COUNT ------------
st.subheader("1. Set the number of classes")
num_classes = st.selectbox("Number of classes:", list(range(2, 11)), index=0, key="num_classes")

if "class_inputs" not in st.session_state or len(st.session_state.class_inputs) != num_classes:
    st.session_state.class_inputs = [{"name": "", "support": [], "query": []} for _ in range(num_classes)]

# ------------ IMAGE UPLOAD FORM ------------
st.markdown("---")
st.subheader("2. Upload images for each class")

with st.form("image_form", clear_on_submit=False):
    class_data = []

    for idx in range(num_classes):
        with st.container():
            st.markdown(f"<div class='class-border'>", unsafe_allow_html=True)
            st.markdown(f"### Class {idx+1}")

            name = st.text_input(f"Class Name {idx+1}", value=st.session_state.class_inputs[idx]["name"], key=f"class_name_{idx}")
            support_imgs = st.file_uploader("Support Images (Training)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key=f"support_{idx}")
            query_imgs = st.file_uploader("Query Images (Testing)", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'], key=f"query_{idx}")

            st.session_state.class_inputs[idx]["name"] = name.strip()
            if support_imgs:
                st.session_state.class_inputs[idx]["support"] = support_imgs
            if query_imgs:
                st.session_state.class_inputs[idx]["query"] = query_imgs

            class_data.append({
                "name": st.session_state.class_inputs[idx]["name"],
                "support": st.session_state.class_inputs[idx]["support"],
                "query": st.session_state.class_inputs[idx]["query"],
            })

            st.markdown("</div>", unsafe_allow_html=True)

    if st.form_submit_button("🡒 Upload Images & Continue"):
        names = [cd["name"] for cd in class_data]
        if not all(names):
            st.error("All classes must be named!")
            st.stop()
        s_counts = [len(cd["support"]) for cd in class_data]
        q_counts = [len(cd["query"]) for cd in class_data]
        if min(s_counts) == 0 or min(q_counts) == 0:
            st.error("Each class must have at least one support and one query image.")
            st.stop()
        if len(set(s_counts)) != 1 or len(set(q_counts)) != 1:
            st.error("Support and Query images must be equal across all classes respectively.")
            st.stop()
        for cd in class_data:
            total_mb = get_total_mb(cd["support"] + cd["query"])
            if total_mb > MAX_UPLOAD_MB_PER_CLASS:
                st.error(f"Total upload for class '{cd['name']}' exceeds {MAX_UPLOAD_MB_PER_CLASS} MB limit.")
                st.stop()

        reset_dirs_and_save_images(class_data)
        st.success("✅ Uploaded successfully.")
        st.session_state.uploaded = True
        st.session_state.class_names = names
        st.session_state.support_count = s_counts[0]
        st.session_state.query_count = q_counts[0]

# ------------ MODEL SELECTION & TRAINING ------------
if st.session_state.get("uploaded"):
    st.markdown("---")
    st.subheader("3. Select Model Backbone")
    selected_model = st.selectbox("Choose a backbone model:", list(MODEL_OPTIONS.keys()),
                                  format_func=lambda m: f"{m} ({MODEL_OPTIONS[m]} MB)",
                                  key="model_select")

    st.markdown("---")
    st.subheader("4. Train and Evaluate")

    if st.button("🚀 Train Model", use_container_width=True):
        st.info("Training started... please wait ⏳")

        config = {
            "support_dir": SUPPORT_ROOT,
            "query_dir": QUERY_ROOT,
            "backbone": selected_model
        }
        try:
            results = run_fewshot_pipeline(config)
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            st.stop()

        st.success("🎉 Training complete!")
        st.markdown(f"**Backbone used:** `{selected_model}`")
        st.write(f"**Accuracy:** {results.get('accuracy', 0):.2f}%")

        labels = results.get("labels", [])
        st.write("**Class Labels:**", labels)

        preds = results.get("predicted_labels", [])
        actuals = results.get("true_labels", [])

        if preds and actuals:
            st.dataframe(pd.DataFrame({
                "Predicted": preds,
                "Actual": actuals
            }))

        exp_path = results.get("export_path")
        if exp_path and os.path.exists(exp_path):
            with open(exp_path, "rb") as f:
                st.download_button("⬇️ Download Trained Model (.pt)", f, file_name=os.path.basename(exp_path), mime="application/octet-stream", use_container_width=True)
        else:
            st.warning("Model file not found for download.")

# ------------ FOOTER ------------
st.markdown("<hr style='margin:1.4em 0;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#8392ab;font-size:0.97em;'><b>Developed by Rohit</b> &bull; Powered by Streamlit &bull; Crafted with PyTorch</div>",
    unsafe_allow_html=True
)
