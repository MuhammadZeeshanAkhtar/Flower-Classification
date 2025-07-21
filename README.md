# 🌸 Flower Classification using Deep Learning (ResNet18 + Fastai)

This project demonstrates a deep learning-based image classification model that can identify different types of flowers using photographs. We used transfer learning with the ResNet50 architecture, powered by Fastai.

---

## 🚀 Project Features

- Classifies 102 types of flowers
- Trained using Fastai’s high-level API
- Uses **ResNet18** (transfer learning) and **ConvNeXt Tiny**
- Interactive web UI powered by **Gradio**
- Real-time predictions with high accuracy

---

## 🧠 Technologies Used

- Python
- Jupyter Notebook
- Fastai & PyTorch
- Pandas
- Gradio (for UI)
- Matplotlib (for visualization)

---

## 📁 Dataset

- **Source:** `fastai.data.external.URLs.FLOWERS`
- 102 flower categories
- Includes labeled training and validation sets
- Label mapping from `flower_name.csv`

---

## 🛠️ How It Works

1. **Preprocessing:**
   - Random resized crop, normalization
   - DataBlock API for input pipeline

2. **Model Training:**
   ```python
   learn = vision_learner(dls, resnet50, metrics=error_rate)
   learn.fine_tune(4)

3. Evaluation:

Confusion matrix

Top losses visualization

4. Deployment:

      gr.Interface(fn=predict, inputs=gr.Image(shape=(512, 512)), outputs=gr.Label(num_top_classes=3)).launch()

📊 Results
Achieved high classification accuracy

Cleaned dataset using ImageClassifierCleaner

Interactive Gradio app supports real-time predictions

🔮 Future Work
Add more advanced models (e.g., EfficientNet)

Support for mobile deployment

Improve UI with confidence scores and visualization

👤 Author
Muhammad Zeeshan Akhtar
BS Software Engineering — Area of Interest: Data Science
Feel free to reach out for feedback, collaboration, or contributions!

