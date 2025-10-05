# 🛰️ VISTA Project: Robust Visual Odometry

**Authors:** [Shay Rozin](https://github.com/shayro9) & [Nitay Amiel](https://github.com/NitayAmiel)  
**Supervisor:** Tamir Shor  

---

## 📖 Overview

This Project explores methods to improve the **robustness of visual odometry (VO)** for drones by introducing an **optimized projected spotlight** in the scene.  
Our approach leverages **differentiable optimization** and **adversarial insights** to improve localization accuracy under adversarial noise conditions.

---

## 🎯 Motivation

We knew that CNN based Visual odometry systems are sensitive to adversarial attacks.  
To address this, we propose **projecting a static, distinctive spotlight pattern** that remains visible to the drone and enhances localization consistency across frames.

---

## ⚙️ Methodology

### 🔄 Pipeline
1. **Multiple Track Generation** – Simulate drone trajectories in Blender using ray tracing.
2. **Simulate Noise** - Applay an adversarial attack to each drone trajectory.
3. **Light Optimization** – Optimize a projected light texture visible to the drone to reduce loss.  
4. **Evaluation** – Measure robustness improvements using the **DPVO** (Deep Patch Visual Odometry) model.

---

## 🧠 Foundational Concepts

### 🧩 DPVO (Deep Patch Visual Odometry)
- RNN based model for visual odometry
- Tracks sparse patch correspondences across frames. 
- State-of-the-art method.  

### ⚡ Noise Injection (PGD)
- Uses **Projected Gradient Descent (PGD)** to optimize adversarial noise.  
- Maximizes VO loss under bounded norm constraints.  
- Frame-based, without physical restrictions — serves as an upper-bound baseline.

### 🟥 Square Attack
- **Black-box** adversarial method with no internal model access.  
- Iteratively adds localized squares for maximum CNN impact.  
- Provides insights for our **“Square Defense”** method.

---

## 💡 Our Approach: Light Optimization

We introduce a **universal, physically grounded defense** mechanism based on light projection.

**Key ideas:**
- 🟦 **Static square-pattern spotlight** optimized using feedback from multiple trajectories.  
- 🧮 **Universal texture** – effective across different tracks.  
- 🧱 **Square Defense** – inspired by adversarial attack properties.  
- ☀️ Optimization performed in Blender using ray tracing and DPVO feedback loss.

---

## 🔬 Experimental Setup

| Component | Description |
|------------|-------------|
| **Simulation** | Blender scenes with ray tracing enabled |
| **Tracks** | 5 unique synthesized drone trajectories |
| **Loss Function** | 6D pose error |
| **Noise Bound** | 5% max-norm constraint |
| **Epochs** | 10 |
| **Texture** | Square-pattern spotlight |
| **Evaluation** | Average loss across all tracks |

---

## 📊 Results Summary

| Condition | Description |
|------------|-------------|
| **CLEAN** | Standard DPVO performance |
| **PGD** | Degraded accuracy under adversarial noise |
| **DEFEND** | Improved robustness using optimized spotlight |

✅ The optimized spotlight enhances localization robustness  
✅ The defense generalizes across tracks  
✅ The method complements DPVO’s patch-based structure  

---

## 🚀 Future Work

- 🔧 **Advanced Noise Models:** Explore physically consistent universal textures.  
- 🟥 **Square Attack Analysis:** Study the defense/attack relationship in depth.  
- 🤖 **Multi-Sensor Fusion:** Integrate IMU/LiDAR for improved resilience.

---

## 📁 Repository Structure

VISTA-Project-Robust-Visual-Odometry/
├── data
├── src
├── results
├── presentation
└── README.md
