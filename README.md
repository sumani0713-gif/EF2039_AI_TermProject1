# EF2039 AI Term Project 1 – Image Super Resolution using Real-ESRGAN

This project performs image super-resolution using the Real-ESRGAN model and pretrained ×4 weights.  
A low-resolution image is upscaled into a high-resolution output using PyTorch.

---

# 1. Project Overview
- Model: Real-ESRGAN (RRDBNet backbone)
- Upscale Factor:** ×4
- Weight Used: `RealESRGAN_x4plus.pth`
- Framework: PyTorch
- Goal: Convert low-resolution images into high-quality, detailed high-resolution images.

---

# 2. Repository Structure

- `run_esrgan.py` — Main inference script  
- `realesrgan/` — Real-ESRGAN source code (imported from official repo)  
- `RealESRGAN_x4plus.pth` — Pretrained ×4 model weight  
- `low.jpg` — Input sample image  
- `output.png` — Output super-resolution result

---

# 3. How to Run

#3.1 Install dependencies
```bash
pip install -r requirements.txt
```

#3.2 Run the inference script
```bash
python run_esrgan.py
```

Output
The result will be saved as output.png in the same directory.

---
# 4. Model Description

Real-ESRGAN restores details using:
- Residual-in-Residual Dense Blocks (RRDB)
- Generative Adversarial Training
- Robust to real-world compressed images, noise, blur, low-quality inputs
- The model used here (x4plus) is designed for ×4 super-resolution.

---
 # 5. Results
Example input/output:
- Input: low-resolution (low.jpg)
- Output: high-resolution (output.png)
(Include images if needed)

---
# 6. Development Process
This repository includes:
Commit history of development
Raw model source code
Inference script
Generated sample output

---
# 7. How to Package (for final submission)
Zip all files except weight and large images:
```bash
EF2039_Proj01_<YourID>_<Name>.zip
```

---
