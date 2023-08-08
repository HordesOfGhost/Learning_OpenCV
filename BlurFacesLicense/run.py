"""

    Main Function
    
    Input Arguments:
    -i, --input  : Folder containing images to blur
    -o, --output : Folder to write blurred images to
    
    Ouptut:
    Blurred images to the folder.
    
"""
from main import BlurFacesLicense
# from main_v7 import BlurFacesLicense
# from main_v8 import BlurFacesLicense
import cv2
import argparse
import os
import glob
import time
import torch
from ultralytics import YOLO
import pandas as pd
import csv

'''

    ArgParsers

'''
ag = argparse.ArgumentParser()
ag.add_argument("-i", "--input", type = str, required = True, help = "Input Folder for images" )
ag.add_argument("-o", "--output", type = str, required = True, help = "Ouput Folder for images")
ag.add_argument("-l", "--log", type = str, required = False, help = "for log purposes")
ag.add_argument("-d","--device", type=str, required = False, help = "set device")

arguments = ag.parse_args()
input_folder = arguments.input
output_folder = arguments.output
log_file = arguments.log
device = arguments.device

'''

    Runing in CPU or GPU

'''
if device == 'cuda':
    device = torch.device("cuda:0")
elif device == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

''' 

    Loading License Detection model in  license_detection_model

'''
license_detection_model = YOLO('models/license_15k.pt')
license_detection_model.to(device)

'''

    Loading Vehicle Detection Model in yolo_model

'''

path = 'yolov7/yolov7-e6e.pt'
yolo_model_v7 = torch.hub.load("WongKinYiu/yolov7","custom",f"{path}",trust_repo=True,verbose=False)
yolo_model_v7.to(device)
yolo_model_v8 = YOLO("models/yolov8x.pt")
yolo_model_v8.to(device)


'''
    Line - Break
'''

if input_folder[-1] != os.path.sep:
    input_folder = input_folder + os.path.sep

if not os.path.exists(output_folder):
    # Create the folder
    os.mkdir(output_folder)

if output_folder[-1] != os.path.sep:
    output_folder = output_folder + os.path.sep
    
file_types = ['JPG','JPEG','PNG','JFIF']
image_paths = glob.glob(input_folder + "*.jpg")
print("Bluring Started \n")
print("---------------------------- \n")

df_list = []

for i,image_path in enumerate(image_paths):
    start = time.time()
    
    print(f"{image_path.split(os.path.sep)[-1]} \n")
    image = cv2.imread(image_path)
    blured_image , total_vehicles_detected, vehicles_detected_v7, vehicles_detected_v8, license_plates_detected, faces_detected = BlurFacesLicense(image = image, yolo_model_v7 = yolo_model_v7, yolo_model_v8 = yolo_model_v8, license_detection_model = license_detection_model).blur_license_plates_and_faces()
    
    end = time.time()
    time_elapsed = end - start
    df_list.append([image_path.split(os.path.sep)[-1], total_vehicles_detected, vehicles_detected_v7, vehicles_detected_v8, license_plates_detected, faces_detected, round(time_elapsed,2)])
    # df_log.append({'image_file' : image_path.split(os.path.sep)[-1] ,'vehicles_detected' : vehicles_detected ,'license_plates_detected' : license_plates_detected,'time_elapsed':time_elapsed},ignore_index = True)
    print(f"{i+1} images Blurred")
    print(f"Time Esplaced {time_elapsed}")
    print("Saving...")
    print("---------------------------- \n")
    
    cv2.imwrite(output_folder + image_path.split(os.path.sep)[-1], blured_image)

'''

    Logs

'''
if len(log_file) != 0:
    df_log = pd.DataFrame(df_list,columns=['image_file','total_vehicles_detected','vehicles_detected_v7','vehicles_detected_v8','license_plates_detected','faces_detected','time_elapsed'])
    df_log.to_csv(f'logs/{log_file}.csv',index = False)


print("Blurring Completed")