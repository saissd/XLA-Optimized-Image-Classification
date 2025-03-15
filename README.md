# XLA-Optimized-Image-Classification

This project implements **ResNet-18** for image classification with **XLA acceleration on TPU** using Google Colab. The model processes images efficiently using **torch-xla**, making inference faster on TPUs.

## 📌 Features  
✔️ Uses **ResNet-18** pretrained on ImageNet  
✔️ Optimized inference using **XLA on TPU**  
✔️ Supports **custom image classification**  
✔️ **Google Colab-compatible** (ready to run)  

---

## 🔧 Installation & Setup  
### **1️⃣ Clone the Repository**  
bash
git clone https://github.com/saissd/XLA-Optimized-Image-Classification.git

cd XLA-Image-Classification



### **2️⃣ Open Google Colab & Upload the Notebook**
Open XLA_Optimized_Image_Classification.ipynb in Google Colab.
### **3️⃣ Install Dependencies**
Run this in Google Colab:

!pip install torch torchvision torch-xla
## **🚀 How to Use**
Upload an image (.jpg or .png).
Run all cells in the Colab notebook.
The model predicts the image class from ImageNet’s 1,000 categories.
The output shows:
🔹 Predicted class label

🔹 Inference time comparison (CPU vs TPU)

🔹 Image with classification result



## 📸 Example Output

![image](https://github.com/user-attachments/assets/281c6d37-8995-4117-993d-cfb1febafb90)


##  🛠 Technologies Used
PyTorch - Deep Learning Framework

Torchvision - Pretrained Models & Image Processing

torch-xla - XLA Optimization for TPU

Google TPU - Fast model inference
---
## ✨ Future Improvements
🔹 Fine-tune ResNet-18 for custom datasets

🔹 Deploy as an API using FastAPI/Flask

🔹 Optimize inference on AWS Inferentia


