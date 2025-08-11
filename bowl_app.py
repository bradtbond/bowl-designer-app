import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for image processing
from scipy.interpolate import PchipInterpolator
import io
import random
import requests # Library to fetch images from URLs

# --- PROFILE LIBRARY (Managed in session_state for dynamic updates) ---
if 'profile_library' not in st.session_state:
    st.session_state.profile_library = {
        "Classic Catenary": {
            "type": "generative", "function": "catenary",
            "description": "A natural, pleasing curve like a hanging chain. Balanced and stable."
        },
        "Sweeping Parabola": {
            "type": "generative", "function": "parabola",
            "description": "A simple, elegant curve that sweeps up quickly from the base."
        },
        "Hellenistic Kylix (Inspired)": {
            "type": "points", "description": "A wide, shallow Ancient Greek drinking cup shape. Dramatic and open.",
            "data": np.array([[0.25,0.00],[0.26,0.05],[0.30,0.10],[0.50,0.18],[0.70,0.23],[0.85,0.28],[0.95,0.50],[0.98,0.75],[1.00,1.00]])
        },
        "Simple Ogee (S-Curve)": {
            "type": "points", "description": "A classic S-shaped curve often seen in architecture and pottery.",
            "data": np.array([[0.30,0.00],[0.32,0.10],[0.50,0.25],[0.60,0.50],[0.55,0.75],[0.70,0.90],[0.85,0.95],[1.00,1.00]])
        },
        "Ellsworth Vessel (Inspired)": {
            "type": "points", "source": "Museum for Art in Wood",
            "description": "Inspired by David Ellsworth's hollow forms. A continuous, elegant curve.",
            "data": np.array([[0.40,0.00],[0.45,0.05],[0.60,0.10],[0.85,0.20],[0.98,0.40],[1.00,0.60],[0.95,0.80],[0.80,0.95],[0.75,1.00]])
        }
    }

# --- CORE FUNCTIONS ---

def calculate_parabolic_curve(bowl_radius, foot_radius, height):
    if bowl_radius == foot_radius: return np.array([foot_radius, bowl_radius]), np.array([0, height])
    a = height / (bowl_radius**2 - foot_radius**2)
    x_coords = np.linspace(foot_radius, bowl_radius, 200)
    y_coords = a * (x_coords**2 - foot_radius**2)
    return x_coords, y_coords

def calculate_catenary_curve(bowl_radius, foot_radius, height, sag):
    x_range = bowl_radius - foot_radius
    if x_range == 0: return calculate_parabolic_curve(bowl_radius, foot_radius, height)
    x_coords = np.linspace(foot_radius, bowl_radius, 200)
    k = sag / x_range 
    cosh_range = np.cosh(k * x_range) - 1.0
    if cosh_range <= 0: return calculate_parabolic_curve(bowl_radius, foot_radius, height)
    y_coords = height * (np.cosh(k * (x_coords - foot_radius)) - 1.0) / cosh_range
    return x_coords, y_coords

def scale_profile_points(points, bowl_radius, foot_radius, height):
    x_norm, y_norm = points[:, 0], points[:, 1]
    x_scaled = foot_radius + (x_norm - x_norm.min()) * (bowl_radius - foot_radius)
    y_scaled = y_norm * height
    interp_func = PchipInterpolator(x_scaled, y_scaled)
    x_smooth = np.linspace(x_scaled.min(), x_scaled.max(), 200)
    y_smooth = interp_func(x_smooth)
    return x_smooth, y_smooth

def plot_bowl(x_coords, y_coords, blank_diameter, blank_height, foot_diameter, units, profile_name):
    bowl_radius, foot_radius = blank_diameter / 2, foot_diameter / 2
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_coords, y_coords, color='black', linewidth=2, label='Bowl Outer Profile'); ax.plot(-x_coords, y_coords, color='black', linewidth=2); ax.plot([-foot_radius, foot_radius], [0, 0], color='black', linewidth=2)
    blank_rect = plt.Rectangle((-bowl_radius, 0), blank_diameter, blank_height, edgecolor='red', facecolor='none', linestyle='--', linewidth=1.5, label='Wood Blank'); ax.add_patch(blank_rect)
    ax.set_aspect('equal', adjustable='box'); ax.grid(True, linestyle=':', color='gray')
    ax.set_xlabel(f"Radius ({units})"); ax.set_ylabel(f"Height ({units})")
    ax.set_title(f"Bowl Design: {profile_name}")
    ax.set_xlim(-bowl_radius * 1.1, bowl_radius * 1.1); ax.set_ylim(-blank_height * 0.1, blank_height * 1.1)
    ax.axvline(0, color='gray', linestyle='-.', linewidth=1); ax.legend()
    st.pyplot(fig)

def extract_profile_from_image(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8); img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, "No contours found."
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour); center_x = x + w / 2
        right_half = np.array([p[0] for p in largest_contour if p[0][0] >= center_x])
        right_half = right_half[right_half[:, 1].argsort()[::-1]]
        x_coords, y_coords = right_half[:, 0], right_half[:, 1]
        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
        unique_x, indices = np.unique(x_norm, return_index=True); unique_y = y_norm[indices]
        normalized_profile = np.vstack((unique_x, unique_y)).T
        contour_img = img.copy()
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 3)
        return normalized_profile, contour_img
    except Exception as e: return None, f"An error occurred: {e}"

def run_generation(profile_name, blank_diameter, blank_height, foot_diameter, units, sag_factor):
    if foot_diameter >= blank_diameter:
        st.error("Error: Foot diameter cannot be larger than the blank diameter."); return
    bowl_radius, foot_radius = blank_diameter / 2, foot_diameter / 2
    selected_profile = st.session_state.profile_library[profile_name]
    if selected_profile["type"] == "generative":
        func = selected_profile["function"]
        if func == "catenary": x, y = calculate_catenary_curve(bowl_radius, foot_radius, blank_height, sag_factor)
        else: x, y = calculate_parabolic_curve(bowl_radius, foot_radius, blank_height)
    else: x, y = scale_profile_points(selected_profile["data"], bowl_radius, foot_radius, blank_height)
    st.session_state.main_col_placeholder.pyplot(plt.gcf())
    with st.session_state.main_col_placeholder.container():
        plot_bowl(x, y, blank_diameter, blank_height, foot_diameter, units, profile_name)

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(layout="wide")
st.title("Woodturning Bowl Designer")

main_col, curator_col = st.columns(2)
with main_col:
    st.header("1. Design a Bowl")
    st.session_state.main_col_placeholder = st.empty()

st.sidebar.header("Bowl Dimensions")
units = st.sidebar.selectbox("Units", ['in', 'mm'])
blank_diameter = st.sidebar.number_input(f"Blank Diameter ({units})", min_value=1.0, value=8.0, step=0.25)
blank_height = st.sidebar.number_input(f"Blank Height ({units})", min_value=1.0, value=3.0, step=0.25)
foot_diameter = st.sidebar.number_input(f"Foot Diameter ({units})", min_value=0.5, value=3.5, step=0.25)
st.sidebar.header("Profile Selection")
profile_name = st.sidebar.selectbox("Choose a Profile", list(st.session_state.profile_library.keys()), key="profile_selector")
st.sidebar.info(st.session_state.profile_library[profile_name]["description"])
sag_factor = 0.15
if st.session_state.profile_library[profile_name].get("function") == "catenary":
    sag_factor = st.sidebar.slider("Catenary Sag", 0.05, 0.5, 0.15, 0.01)

col1, col2 = st.sidebar.columns(2)
if col1.button("Generate Profile", type="primary"):
    run_generation(profile_name, blank_diameter, blank_height, foot_diameter, units, sag_factor)
if col2.button("Surprise Me!"):
    if st.session_state.profile_library:
        random_profile_name = random.choice(list(st.session_state.profile_library.keys()))
        st.success(f"Lucky choice: {random_profile_name}")
        random_sag = 0.15 
        if st.session_state.profile_library[random_profile_name].get("function") == "catenary": random_sag = sag_factor
        run_generation(random_profile_name, blank_diameter, blank_height, foot_diameter, units, random_sag)
    else: st.warning("Profile library is empty!")

with curator_col:
    st.header("2. Add a Profile with AI")
    input_tab1, input_tab2 = st.tabs(["Upload an Image", "From a URL"])
    with input_tab1:
        st.write("Upload a clear, side-profile image of a bowl against a plain background.")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="uploader")
        if uploaded_file:
            st.session_state.image_bytes = uploaded_file.getvalue()
    with input_tab2:
        st.write("Find an image online, right-click and 'Copy Image Address', then paste it below.")
        image_url = st.text_input("Paste Image URL here:")
        if st.button("Fetch & Analyze from URL"):
            if image_url:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}
                    response = requests.get(image_url, stream=True, timeout=10, headers=headers)
                    response.raise_for_status()
                    st.session_state.image_bytes = response.content
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not fetch image: {e}")
            else: st.warning("Please paste a URL.")

    if 'image_bytes' in st.session_state and st.session_state.image_bytes:
        st.image(st.session_state.image_bytes, caption="Source Image", use_column_width=True)
        profile_data, processed_image = extract_profile_from_image(st.session_state.image_bytes)
        if profile_data is not None:
            st.image(processed_image, caption="AI Detected Contour", use_column_width=True)
            st.success("Profile extracted successfully!")
            new_profile_name = st.text_input("Enter a name for this new profile:")
            if st.button("Add to Library"):
                if new_profile_name:
                    st.session_state.profile_library[new_profile_name] = {
                        "type": "points", "description": f"Custom profile from user input", "data": profile_data
                    }
                    st.success(f"'{new_profile_name}' added! It is now available in the dropdown.")
                    del st.session_state.image_bytes 
                    st.rerun()
                else: st.warning("Please enter a name for the profile.")
        else:
            st.error(f"Could not extract profile: {processed_image}")
        if 'image_bytes' in st.session_state:
            del st.session_state.image_bytes