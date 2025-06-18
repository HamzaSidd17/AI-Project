# TORCS Autonomous Driver – NeuroEvolution & Behaviour Cloning

**Final Year AI Project – Spring 2025**  
**Submitted on:** May 11, 2025

## 👥 Group Members
- **Muhammad Daniyal Aziz** – 22i-0753  
- **Syed Hussain Ali Zaidi** – 22i-0902  
- **Hamza Ahmad** – 22i-1339

---

## 🚗 Project Overview

This project explores two distinct approaches for training an autonomous driver in **TORCS** (The Open Racing Car Simulator):

- **NeuroEvolution using Genetic Algorithms (GA)**
- **Behaviour Cloning (BC) via Supervised Learning**

Each method was implemented, tested, and evaluated for **driving performance, training speed, and adaptability**.

---

## 🧬 NeuroEvolution: Genetic Algorithm with CNN

In this approach, a **Genetic Algorithm evolves a Convolutional Neural Network (CNN)** that maps real-time sensor inputs to driving actions (steering, throttle, brake). Instead of traditional backpropagation, weights are optimized through evolution.

### 🛠️ Implementation
- **Developers**: Muhammad Daniyal Aziz & Syed Hussain Ali Zaidi
- **Inputs**: Speed, angle, track position, opponent distance (sensor data)
- **Outputs**: Steering, throttle, brake
- **Fitness Function**: Rewards survival time, smooth driving, fewer collisions

### ✅ Pros
- No labeled data needed; learning is performance-based
- Capable of generalizing to unseen tracks
- Finds creative strategies without explicit programming

### ❌ Cons
- Very slow due to simulation limits (max 10 parallel TORCS instances)
- Requires many generations and a large population size
- High computational cost

---

## 🧠 Behaviour Cloning: Supervised Learning from Expert Bot

Behaviour Cloning trains a CNN to **mimic an expert driver**. Using the Ahura bot, we recorded driving data to train a model that maps image frames to control actions.

### 🛠️ Implementation
- **Developers**: Muhammad Daniyal Aziz, Syed Hussain Ali Zaidi & Hamza Ahmad
- **Tools**: PyTorch, CUDA
- **Training Data**: CSV files containing image frames and corresponding control values from Ahura bot
- **Loss Function**: Mean Squared Error (MSE)

### ✅ Pros
- Much faster training process
- Replicates expert behavior effectively in familiar environments
- Scales well with additional expert data and compute power

### ❌ Cons
- Weak generalization to unfamiliar tracks
- Prone to error accumulation during long or complex races

---

## 🧾 Conclusion

Both strategies bring unique strengths to autonomous driving:

| Method             | Strengths                                  | Weaknesses                                      |
|--------------------|---------------------------------------------|-------------------------------------------------|
| **NeuroEvolution** | Strong generalization, creative strategies | Very slow, computationally expensive            |
| **Behaviour Cloning** | Fast, accurate on known tracks          | Poor generalization, error compounding          |

Given our limited training time and compute, **Behaviour Cloning** emerged as the more practical choice. However, future directions could include **hybrid models** or **reinforcement learning** to merge performance with generalization.


---

## 📌 Technologies Used
- Python, PyTorch, CUDA
- OpenCV, NumPy, Pandas
- TORCS simulator
- Genetic Algorithms

---

> _"Learning to drive in pixels and sensors – one crash at a time."_


![image](https://github.com/user-attachments/assets/1948c31c-573b-4a88-a09f-3edda9b4cc74)
![image](https://github.com/user-attachments/assets/5147d088-4f2b-4f14-ba43-585621512e81)

