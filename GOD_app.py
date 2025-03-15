# -*- coding: utf-8 -*-
# import some common libraries
import streamlit as st
import detectron2
import numpy as np
import cv2
import random
from PIL import Image
import torch

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

trained_classes = ["bird","flower"]

#paths to config and weight file of trained model
MODEL_CONFIG = "/home/advait/Desktop/machine learning/config.yml"
MODEL_WEIGHTS = "/home/advait/Desktop/machine learning/model_final.pth"

@st.cache(allow_output_mutation= True)
def create_config_file(threshold , model_config, model_weights) :
  '''
    This function creates a config file of the model and defines a predictor
  '''

  cfg = get_cfg()
  
  cfg.merge_from_file(model_config)
  cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set threshold for this model

  cfg.MODEL.WEIGHTS = (model_weights)

  predictor = DefaultPredictor(cfg)
  return cfg , predictor


def detect_objects(img, model_config, model_weights , threshold = 0.5 ):
  ''' 
    This function used to detect objects in a given image.

    params :
      threshold : confidence threshold used for prediction in model
      model_config : path to config file of pretrained model
      model_weights : path to .pth file of pretrained models
      img : image in which objects have to be detected

    returns :
      image with bounding boxes , classes that were detected in that particular image
  '''
  image = np.asarray(img)
  cfg, predictor = create_config_file(threshold = threshold, model_config= model_config , model_weights = model_weights) 
  outputs = predictor(image)
  v = Visualizer(img_rgb = image, metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=trained_classes))
  predicted_classes = outputs["instances"].pred_classes
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  return out.get_image(), predicted_classes


def main():
  st.title("General Object Detection")
  st.write("This Model can currently detect " + str(len(trained_classes)) + " objects : " + ", ".join(trained_classes))

  st.sidebar.title("Parameters")
  threshold = st.sidebar.slider("Threshold" ,value = 0.7,min_value= 0.0 , max_value = 1.0 , step= 0.1)


  #Receiving and extracting image
  uploaded_images = st.file_uploader("Choose a png or jpg image", type=["jpg", "png", "jpeg"] , accept_multiple_files= True)
  images = []

  for uploaded_image in uploaded_images :
    if uploaded_image is not None:
      image = Image.open(uploaded_image)
      st.image(image, caption="Uploaded Image.", use_column_width=True)
      image = image.convert("RGB")# Make sure image is RGB
      images.append(image)

  if st.button("Detect Objects"):
    for image in images :
      with st.spinner("Detecting Objects ...") :

        result_image, predictions  = detect_objects(image , model_config= MODEL_CONFIG, model_weights=MODEL_WEIGHTS , threshold = threshold)
        st.image(result_image)
      classes = set(predictions.cpu().numpy())
      st.success("Objects Detected : " + ", ". join([trained_classes[i] for i in classes]))
    

if __name__ == '__main__':
  main()