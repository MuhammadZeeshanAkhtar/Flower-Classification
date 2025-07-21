
from fastai.vision.all import * 
import gradio as gr
import warnings

# Suppress FastAI pickle warning
warnings.filterwarnings("ignore", category=UserWarning, module='fastai.learner')
# Load the trained model
learn_inf = load_learner('export.pkl')

# Get the labels
labels = learn_inf.dls.vocab

# Define prediction function
def predict(img):
    pred, pred_idx, probs = learn_inf.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Gradio interface
title = "Flower Classifier"
description = "Flower classifier for 102 types of flowers"


# Launch the interface
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", image_mode='RGB', height=512, width=512),
    outputs=gr.Label(num_top_classes=3),
    title=title,
    description=description,
   
).launch()
