---

### 📢 Image-to-Image Translation with Pix2Pix (cGAN)

📌 *Internship Task - Prodigy InfoTech (Generative AI Intern)*

As part of my Generative AI Internship at *Prodigy InfoTech, I developed an image-to-image translation system using **Pix2Pix, a type of **Conditional Generative Adversarial Network (cGAN)*. This task provided deep insights into adversarial learning, computer vision, and image synthesis.

---

### 🛠 Implementation Overview

✅ *Pix2Pix Architecture (cGAN)*
Built a deep learning pipeline with:

* A *U-Net based Generator* that translates input images (e.g., edge maps or sketches) to photo-realistic outputs
* A *PatchGAN Discriminator* that evaluates 70×70 patches for realism and structure

✅ *Dataset Handling*

* Used paired datasets from facades, maps via *TensorFlow Datasets*
* Designed fallback support for *custom datasets* and *synthetic data*

✅ *Training & Loss*

* Combined *Adversarial Loss* and *L1 Loss* to balance realism and accuracy
* Visual output after every few epochs, showing progression of generator performance

✅ *Visualization and Testing*

* Side-by-side display: Input Image | Ground Truth | Generated Image
* Added loss tracking plots and custom test image evaluation pipeline

---

### 🖼 Output Sample

Generated images that translate structural representations into detailed visuals using learned pixel mappings.
![Stable Diffusion Output](https://github.com/Vidyyaaakshaya3/PRODIGY_GA_04/blob/main/Task_4_ouput.png)
---

### 💡 Features

🎯 *Conditional Generation*: Learns mappings from input to target images
🧠 *Skip Connections*: U-Net enhances low-level detail retention
📊 *Loss Plots*: Adversarial vs. reconstruction loss monitoring
🖥 *CLI Training Interface*: Easy to train, visualize, and extend

---

### 📌 Internship Credit

Task-04: Part of the *Generative AI Internship at Prodigy InfoTech*
Model: *Pix2Pix (cGAN)* using TensorFlow
Role: *GenAI Intern*
Date: *July 2025*

---
