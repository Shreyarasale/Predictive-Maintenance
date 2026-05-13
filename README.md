# Predictive Maintenance of Deep Groove Ball Bearings using Vibration Analysis

An AI-powered predictive maintenance system that detects and localizes bearing faults using vibration signal analysis and deep learning. The project leverages a **Condition-Aware Autoencoder (CA-AE)** with physics-informed diagnostics to identify anomalies under varying operating conditions without requiring labeled fault data. 

---

## 🚀 Features

* Condition-Aware Autoencoder for unsupervised anomaly detection
* Physics-informed fault localization using spectral band analysis
* Vibration signal preprocessing using FFT & Hilbert Transform
* Adaptive Health Index generation for real-time monitoring
* Load & RPM aware diagnostics to reduce false positives
* Fault classification for:

  * Inner Race Defects
  * Outer Race Defects
  * Bearing Resonance Issues
  * Misalignment & Looseness

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch**
* **NumPy**
* **SciPy**
* **Matplotlib**
* **Scikit-Learn**
* **Streamlit** (Dashboard/UI)

---

## 📂 Project Structure

```bash id="isdr4m"
├── train_model.py
├── app.py
├── preprocessing.py
├── autoencoder_model.py
├── diagnostics.py
├── config.py
├── utils.py
├── dataset/
├── checkpoints/
└── reports/
```

---

## ⚙️ Methodology

1. Raw vibration signals collected from the **CWRU Bearing Dataset**
2. Signal preprocessing using:

   * Bandpass Filtering
   * Hilbert Envelope Analysis
   * FFT & Log-Spectrograms
3. Condition vectors (Load, RPM) injected into the neural network
4. Autoencoder trained only on healthy vibration data
5. Reconstruction error used to generate a **Health Index**
6. Physics-based frequency analysis used for fault localization



---

## 📊 Results

* Achieved **100% classification precision**
* AUC Score: **1.0**
* Robust fault separation under varying load conditions
* Reduced false alarms using adaptive thresholding
* Effective latent space clustering for healthy vs faulty states



---

## 🧠 Applications

* Industrial Manufacturing
* Wind Turbine Monitoring
* Electric Vehicles
* Aerospace Systems
* Smart Infrastructure
* Oil & Gas Machinery



---

## 📈 Future Enhancements

* Real-time IoT sensor integration
* Cloud-based monitoring dashboard
* Remaining Useful Life (RUL) prediction
* Edge AI deployment for industrial systems
* Transfer learning for multiple bearing types

---

## 📚 Dataset

* **Case Western Reserve University (CWRU) Bearing Dataset**

---



---


