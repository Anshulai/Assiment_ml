import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import io
import base64
from pathlib import Path

from unet_model import ConditionalUNet, create_color_mapping
from inference import PolygonColorInference

# Page configuration
st.set_page_config(
    page_title="Ayna ML Assignment",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    try:
        inferencer = PolygonColorInference(model_path)
        return inferencer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_available_polygons(dataset_root):
    input_dir = os.path.join(dataset_root, 'validation', 'inputs')
    if not os.path.exists(input_dir):
        input_dir = os.path.join(dataset_root, 'training', 'inputs')
    if os.path.exists(input_dir):
        polygons = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        return sorted(polygons), input_dir
    return [], ""

def create_color_buttons():
    colors = ['blue', 'cyan', 'green', 'magenta', 'orange', 'purple', 'red', 'yellow']
    color_hex = {
        'blue': '#0000FF', 'cyan': '#00FFFF', 'green': '#00FF00', 'magenta': '#FF00FF',
        'orange': '#FFA500', 'purple': '#800080', 'red': '#FF0000', 'yellow': '#FFFF00'
    }
    st.markdown('<p class="section-header">🎨 Select Color:</p>', unsafe_allow_html=True)
    cols = st.columns(len(colors))
    for i, color in enumerate(colors):
        selected = st.session_state.get('selected_color') == color
        border = "4px solid red" if selected else "3px solid white"
        with cols[i]:
            if st.button("", key=f"color_btn_{color}", help=color.title()):
                st.session_state.selected_color = color
            st.markdown(f"""
                <div style="
                    background-color: {color_hex[color]};
                    border: {border};
                    border-radius: 50%;
                    width: 60px;
                    height: 60px;
                    margin: auto;
                    box-shadow: 0 0 5px rgba(0,0,0,0.3);
                ">
                </div>
            """, unsafe_allow_html=True)
    return st.session_state.get('selected_color', 'blue')

def create_polygon_buttons(polygons, input_dir):
    st.markdown('<p class="section-header">🔷 Select Polygon Shape:</p>', unsafe_allow_html=True)
    if not polygons:
        st.error("No polygon images found in dataset!")
        return None, None
    cols_per_row = 4
    rows = [polygons[i:i + cols_per_row] for i in range(0, len(polygons), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for i, polygon in enumerate(row):
            polygon_name = polygon.replace('.png', '').title()
            selected = st.session_state.get('selected_polygon') == polygon
            border = "4px solid red" if selected else "2px solid #ccc"
            with cols[i]:
                if st.button("", key=f"poly_{polygon}"):
                    st.session_state.selected_polygon = polygon
                    st.session_state.selected_path = os.path.join(input_dir, polygon)
                st.markdown(f"""
                    <div style="
                        border: {border};
                        padding: 10px;
                        border-radius: 10px;
                        background-color: #111;
                        text-align: center;
                        color: white;
                        font-weight: bold;
                    ">
                        {polygon_name}
                    </div>
                """, unsafe_allow_html=True)
    return st.session_state.get('selected_polygon'), st.session_state.get('selected_path')

def display_prediction(inferencer, image_path, color_name):
    try:
        with st.spinner(f"Generating {color_name} colored polygon..."):
            input_img, pred_img = inferencer.predict(image_path, color_name)
        col1, col2 = st.columns(2)
        with col1:
          st.image(input_img, caption="Original Polygon", use_container_width=True)
        with col2:
          st.image(pred_img, caption=f"{color_name.title()} Colored Polygon", use_container_width=True)

        return True
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")
        return False

def show_color_variations(inferencer, image_path, polygon_name):
    colors = ['blue', 'cyan', 'green', 'magenta', 'orange', 'purple', 'red', 'yellow']
    st.markdown("### 🌈 All Color Variations")
    with st.spinner("Generating all color variations..."):
        rows = [colors[i:i + 4] for i in range(0, len(colors), 4)]
        for row in rows:
            cols = st.columns(4)
            for i, color in enumerate(row):
                with cols[i]:
                    try:
                        _, pred_img = inferencer.predict(image_path, color)
                        st.image(pred_img, caption=color.title(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error with {color}: {str(e)}")

def main():
    st.markdown('<h1 class="main-header">🎓 Ayna ML Assignment</h1>', unsafe_allow_html=True)
    st.markdown("---")
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        model_path = st.text_input("Model Path", value="results/best_model.pth")
        dataset_root = st.text_input("Dataset Root", value="dataset")
        st.markdown("---")
        st.markdown("## 📊 Model Info")
        inferencer = load_model(model_path)
        if inferencer:
            st.success("✅ Model loaded successfully!")
            st.info("🔥 Ready to generate colored polygons!")
        else:
            st.error("❌ Model not found or failed to load")
            return
    if inferencer:
        polygons, input_dir = get_available_polygons(dataset_root)
        if not polygons:
            st.error("No polygon images found! Please check your dataset path.")
            return
        col1, col2 = st.columns([1, 1])
        with col1:
            selected_polygon, selected_path = create_polygon_buttons(polygons, input_dir)
        with col2:
            selected_color = create_color_buttons()
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            generate_single = st.button("🎨 Generate Colored Polygon")
        with col2:
            generate_all = st.button("🌈 Show All Colors")
        with col3:
            if st.button("🔄 Clear Results"):
                st.rerun()
        if selected_polygon and selected_color:
            st.info(f"📋 Current Selection: **{selected_polygon.replace('.png', '').title()}** polygon in **{selected_color.title()}** color")
        st.markdown("---")
        if selected_polygon and selected_path:
            if generate_single and selected_color:
                st.markdown("## 🎯 Generated Result")
                success = display_prediction(inferencer, selected_path, selected_color)
                if success:
                    st.success(f"✅ Successfully generated {selected_color} {selected_polygon.replace('.png', '')}!")
            elif generate_all:
                st.markdown("## 🌈 All Color Variations")
                show_color_variations(inferencer, selected_path, selected_polygon.replace('.png', ''))
        else:
            st.info("👆 Please select a polygon shape and color to get started!")
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>🤖 MADE BY Anshul kumar Singh </p>
            <p>Select a polygon shape and color, then click generate!</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
