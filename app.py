import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from fpdf import FPDF
from datetime import datetime

# ---------------------------------------------------------
# STEP 1: GLOBAL CALIBRATION VALUES (FROM COLAB)
# ---------------------------------------------------------
TRAIN_MEAN = 0.012446394482234157
TRAIN_STD = 0.06560105234766174

CALIBR_MIN = 0.13409189879894257
CALIBR_MAX = 0.586746096611023

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance of Bearings",
    
    layout="wide"
)

# ---------------------------------------------------------
# AUTOENCODER MODEL ARCHITECTURE
# ---------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=1028):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

@st.cache_resource
def load_model():
    model = Autoencoder(1028)
    try:
        model.load_state_dict(torch.load("autoencoder_weights.pth", map_location="cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Error: 'autoencoder_weights.pth' not found. Please upload the retrained model file.")
        return None

model = load_model()

# ---------------------------------------------------------
# SIGNAL PROCESSING FUNCTIONS
# ---------------------------------------------------------
def bandpass_filter(signal, fs=48000, low=500, high=10000):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def create_windows(signal, window_size=2048):
    windows = []
    for i in range(0, len(signal)-window_size, window_size):
        windows.append(signal[i:i+window_size])
    return np.array(windows)

def ema(signal, alpha=0.1):
    out = [signal[0]]
    for i in range(1, len(signal)):
        out.append(alpha * signal[i] + (1-alpha) * out[i-1])
    return np.array(out)

def generate_pdf_report(filename, condition, hi, status, dominance):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(190, 10, "Maintenance Audit Report", 0, 1, 'C')
    
    pdf.set_font("Arial", '', 10)
    pdf.cell(190, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "1. System Context", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(50, 10, f"Filename: {filename}", 0, 1)
    pdf.cell(50, 10, f"Load Condition: {condition}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "2. Diagnostics", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(50, 10, f"Health Index (HI): {hi:.4f}", 0, 1)
    pdf.cell(50, 10, f"Status: {status}", 0, 1)
    pdf.cell(50, 10, f"Fault Type: {dominance}", 0, 1)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(190, 10, "3. Recommendations", 0, 1)
    pdf.set_font("Arial", 'I', 11)
    
    if hi > 0.6:
        rec = "CRITICAL: Immediate shutdown advised. Inspect bearing elements for spalling."
    elif hi > 0.35:
        rec = "WARNING: Schedule maintenance. Check looseness, mounting bolts, and alignment."
    else:
        rec = "NORMAL: System is healthy. Continue routine monitoring."
        
    pdf.multi_cell(190, 10, rec)
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------------------------------
# ADVANCED EXPERT SYSTEM CHATBOT
# ---------------------------------------------------------
def get_advanced_response(question, hi, status, fault_text, band_data):
    """
    Inputs:
        question: User's text
        hi: Current Health Index (float)
        status: "Healthy", "Warning", "Critical"
        fault_text: "Bearing Fault", "Looseness", etc.
        band_data: Dictionary {'low': val, 'mid': val, 'high': val}
    """
    q = question.lower()
    
    # --- LOGIC BRANCH 1: "WHY" (Root Cause & Physics) ---
    if "why" in q:
        if hi < 0.35:
            return (f"**Why is it Healthy?**\n"
                    f"The calculated Health Index is {hi:.3f}, which is very low. "
                    f"The Autoencoder successfully reconstructed the signal, meaning the vibration patterns match "
                    f"the 'healthy' training data perfectly. No anomalies were found in the frequency spectrum.")
        
        elif "Bearing" in fault_text:
            return (f"**Why is there a Bearing Fault?**\n"
                    f"The model detected a high reconstruction error ({band_data['high']:.3f}) in the High-Frequency band (600Hz+). "
                    f"In rolling element bearings, defects on the Inner or Outer race cause repetitive impacts. "
                    f"These impacts excite high-frequency resonances that the Autoencoder (trained on healthy data) "
                    f"does not recognize, resulting in this specific fault classification.")
            
        elif "Looseness" in fault_text:
            return (f"**Why is it Looseness?**\n"
                    f"The error is highest in the Mid-Frequency band ({band_data['mid']:.3f}). "
                    f"Mechanical looseness (e.g., loose bolts, soft foot) typically generates harmonics at 2x, 3x, or 4x the running speed. "
                    f"These mid-range harmonics are distinct from high-frequency bearing noise.")
            
        elif "Imbalance" in fault_text:
            return (f"**Why is it Imbalance/Misalignment?**\n"
                    f"The Low-Frequency band has the highest error ({band_data['low']:.3f}). "
                    f"Unbalance and misalignment create strong forces at 1x RPM (Running Speed). "
                    f"Since this is a low-frequency phenomenon, the model flags it as an operational issue rather than a component failure.")

    # --- LOGIC BRANCH 2: "WHAT" (Definitions & Status) ---
    if "what" in q:
        if "fault" in q or "wrong" in q or "issue" in q:
            return (f"**Current Diagnosis:** {fault_text}.\n"
                    f"**Severity:** {status}.\n"
                    f"**Health Index:** {hi:.3f} (0 is best, 1 is worst).")
            
        if "hi" in q or "index" in q:
            return ("**What is the Health Index (HI)?**\n"
                    "It is a normalized score between 0 and 1 representing the 'Anomaly Level'.\n"
                    "• **0.0 - 0.35:** Healthy (Normal Operation)\n"
                    "• **0.35 - 0.65:** Warning (Developing Fault)\n"
                    "• **0.65 - 1.0:** Critical (Failure Imminent)\n"
                    "It is derived from the Mean Squared Error of the Autoencoder's reconstruction.")
            
        if "autoencoder" in q or "model" in q:
            return ("**What is the Model?**\n"
                    "I am using an Unsupervised Deep Autoencoder. I compress the vibration data into 32 latent features "
                    "and try to reconstruct it. If I cannot reconstruct the signal accurately, it means the machine behavior "
                    "has deviated from the healthy baseline.")

    # --- LOGIC BRANCH 3: "HOW" (Actionable Maintenance) ---
    if "how" in q or "fix" in q or "action" in q or "recommend" in q:
        if "Healthy" in status:
            return "**Recommendation:** No maintenance required. Continue routine monitoring and data collection."
        
        if "Bearing" in fault_text:
            return ("⚠️ **Maintenance Plan (Bearing Fault):**\n"
                    "1. Schedule machine downtime.\n"
                    "2. Perform vibration spectrum analysis to confirm inner vs. outer race defect.\n"
                    "3. Inspect the bearing for spalling or pitting.\n"
                    "4. Prepare a replacement bearing (verify part number).")
            
        if "Looseness" in fault_text:
            return ("⚠️ **Maintenance Plan (Structural):**\n"
                    "1. Check all foundation bolts with a torque wrench.\n"
                    "2. Inspect the base frame for cracks.\n"
                    "3. Verify 'soft foot' conditions using feeler gauges.\n"
                    "4. Tighten any loose guarding or external components.")
            
        if "Imbalance" in fault_text:
            return ("⚠️ **Maintenance Plan (Operational):**\n"
                    "1. Check coupling alignment using laser alignment tools.\n"
                    "2. Clean the fan/impeller blades (dirt buildup causes imbalance).\n"
                    "3. Verify the shaft is not bent.")

    # --- FALLBACK (General Summary) ---
    return (f"I am analyzing the machine vibration.\n"
            f"**Status:** {status} (HI: {hi:.3f})\n"
            f"**Dominant Pattern:** {fault_text}\n"
            f"You can ask me **Why** this is happening, **What** the Health Index means, or **How** to fix it.")

# ---------------------------------------------------------
# SIDEBAR – INPUT
# ---------------------------------------------------------
st.sidebar.title("Input Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload  vibration file (.mat)",
    type=["mat"]
)

current_condition = st.sidebar.selectbox(
    "Operating Load Condition",
    ["LOW_LOAD", "MID_LOAD", "HIGH_LOAD"]
)

# CONDITION MAP
cond_map = {
    "LOW_LOAD":  np.array([1797/2000, 0/4, 1.0]),
    "MID_LOAD":  np.array([1772/2000, 1/4, 1.0]),
    "HIGH_LOAD": np.array([1750/2000, 2/4, 1.0])
}

# ---------------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------------
st.title("🛠️ Predictive Maintenance of Bearings")
st.caption("Unsupervised vibration based health monitoring using condition-aware autoencoders")

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
if uploaded_file is not None and model is not None:

    try:
        data = loadmat(uploaded_file)
        # Handle different file key naming conventions
        key = [k for k in data.keys() if "DE_time" in k][0]
        signal = data[key].squeeze()
    except IndexError:
        st.error("Could not find 'DE_time' data in this file. Please check the .mat structure.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # 1. Preprocessing (Global Normalization)
    signal = (signal - TRAIN_MEAN) / TRAIN_STD
    
    windows = create_windows(signal)
    if len(windows) == 0:
        st.error("Signal too short for analysis.")
        st.stop()

    filtered = np.array([bandpass_filter(w) for w in windows])
    envelope = np.abs(hilbert(filtered, axis=1))
    fft = np.abs(np.fft.rfft(envelope, axis=1))
    fft_log = np.log1p(fft)

    # 2. Add Condition Vector (Context)
    selected_cond_vector = cond_map[current_condition]
    cond_features = np.tile(selected_cond_vector, (fft_log.shape[0], 1))
    
    # Combine FFT + Condition
    X = np.hstack([fft_log, cond_features]).astype(np.float32)

    # 3. Model Inference
    with torch.no_grad():
        recon = model(torch.tensor(X)).numpy()

    # 4. Global HI Calculation
    recon_error = np.mean((X - recon)**2, axis=1)
    recon_error = np.maximum(recon_error, CALIBR_MIN)
    HI = (recon_error - CALIBR_MIN) / (CALIBR_MAX - CALIBR_MIN)
    HI_smooth = ema(HI)

    # 5. Band-wise Analysis
    low_idx, mid_idx, high_idx = slice(0,200), slice(200,600), slice(600,1025)
    
    re_low = np.mean((fft_log[:,low_idx]-recon[:,low_idx])**2, axis=1)
    re_mid = np.mean((fft_log[:,mid_idx]-recon[:,mid_idx])**2, axis=1)
    re_high= np.mean((fft_log[:,high_idx]-recon[:,high_idx])**2, axis=1)

    # Smooth raw errors
    raw_low_s = ema(re_low)
    raw_mid_s = ema(re_mid)
    raw_high_s = ema(re_high)

    # 6. Dominance Logic
    last_idx = -1
    
    # Logic: Which band has the highest error?
    if raw_high_s[last_idx] > raw_mid_s[last_idx] and raw_high_s[last_idx] > raw_low_s[last_idx]:
        dominance_text = "Bearing Fault (High-frequency dominant)"
    elif raw_mid_s[last_idx] > raw_low_s[last_idx]:
        dominance_text = "Structural Degradation (Looseness)"
    else:
        dominance_text = "Operational Issue (Imbalance / Misalignment)"
    
    # 7. Final Status Decision
    current_hi = HI_smooth[-1]
    
    if current_hi < 0.35: 
        status_text = "Healthy System"
        status_color = "green"
        dominance_text = "None (Healthy)" # Override dominance
    elif current_hi < 0.65:
        status_text = "Warning (Incipient Fault)"
        status_color = "orange"
    else:
        status_text = "Critical Fault"
        status_color = "red"

    # Data dictionary for chatbot to use
    band_data = {
        'low': raw_low_s[-1],
        'mid': raw_mid_s[-1],
        'high': raw_high_s[-1]
    }

    # -----------------------------------------------------
    # DASHBOARD VISUALIZATION
    # -----------------------------------------------------
    col1, col2 = st.columns([2,1])

    with col1:
        st.subheader("Health Index Trend")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(HI_smooth, label="Health Index", color=status_color, linewidth=2)
        ax.axhline(y=0.35, color='g', linestyle='--', alpha=0.5, label="Healthy Limit")
        ax.axhline(y=0.65, color='r', linestyle='--', alpha=0.5, label="Critical Limit")
        
        ax.set_title("Health Index Trend")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Usage Windows")
        ax.set_ylabel("Health Index")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.subheader("Diagnostics")
        st.metric("Current Health Index", f"{current_hi:.3f}")
        st.markdown(f"**System Status:** <span style='color:{status_color}; font-weight:bold'>{status_text}</span>", unsafe_allow_html=True)
        st.info(f"Dominant Pattern: {dominance_text}")

    st.divider()
    st.subheader("Band-wise Error Contribution")
    fig2, ax2 = plt.subplots(figsize=(8, 2))
    ax2.plot(raw_low_s, label="Low Freq (Misalignment)")
    ax2.plot(raw_mid_s, label="Mid Freq (Looseness)")
    ax2.plot(raw_high_s, label="High Freq (Bearing Fault)", linewidth=2)
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # -----------------------------------------------------
    # NEW FEATURE: DOWNLOADABLE RESULTS
    # -----------------------------------------------------
    st.divider()
    st.subheader("Export Results")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        # Generate PDF
        pdf_bytes = generate_pdf_report(uploaded_file.name, current_condition, current_hi, status_text, dominance_text)
        st.download_button(
            label="📄 Download Audit Report (PDF)",
            data=pdf_bytes,
            file_name="maintenance_report.pdf",
            mime="application/pdf"
        )
        
    with d_col2:
        # Generate CSV
        csv_data = pd.DataFrame({
            "Window_Index": range(len(HI_smooth)),
            "Health_Index": HI_smooth,
            "Low_Freq_Error": raw_low_s,
            "Mid_Freq_Error": raw_mid_s,
            "High_Freq_Error": raw_high_s
        }).to_csv(index=False)
        st.download_button(
            label="💾 Download Raw Data (CSV)",
            data=csv_data,
            file_name="sensor_data.csv",
            mime="text/csv"
        )

    # -----------------------------------------------------
    # IMPROVED CHATBOT (Context-Aware)
    # -----------------------------------------------------
    st.divider()
    st.subheader("🤖 Maintenance Assistant")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat Input
    user_q = st.text_input("Ask 'Why', 'What', or 'How' about the machine:", placeholder="e.g., 'Why is the health index high?'")

    if user_q:
        # Get smart response
        response = get_advanced_response(user_q, current_hi, status_text, dominance_text, band_data)
        
        # Add to history
        st.session_state.chat_history.append(("User", user_q))
        st.session_state.chat_history.append(("Assistant", response))

    # Display Chat History (Most recent at bottom)
    if st.session_state.chat_history:
        for role, text in st.session_state.chat_history:
            if role == "User":
                st.info(f"{text}")
            else:
                st.success(f"{text}")

else:
    st.info("⬅ Please upload a  .mat vibration file to begin analysis.")