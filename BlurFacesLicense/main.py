'''

    Importing Libraries

'''
import cv2
import torch

import traceback
from retinaface import RetinaFace
import matplotlib.pyplot as plt



class BlurFacesLicense:
    '''

    Class  : BlurFacesLicense
    Inputs : image, yolo_model, license_detection_model
            image                   : the image to be blurred
            yolo_model              : model to detect vehicle
            license_detection_model : model to detect license 

    '''
    
    def __init__(self, image, yolo_model_v7, yolo_model_v8, license_detection_model):
        self.image = image
        self.yolo_model_v7 = yolo_model_v7
        self.yolo_model_v8 = yolo_model_v8
        self.license_detection_model = license_detection_model
        self.license_plates_detected = 0
        self.faces_detected = 0
        self.total_vehicles_detected = 0
        self.vehicles_detected_v8 = 0
        self.vehicles_detected_v7 = 0
        # self.plot_image(self.image)
    
    def plot_image(self,image):
        '''

            Input   : image
                      image : image to be plotted
            Display : Plot of input image

        '''
        plt.axis('off');
        plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));
        plt.show();
    
    def blur_faces(self):
        '''
        
            Blurs People's Face using RetinaFace
        
        '''
        
        image = self.image.copy()

        '''
        
            RetinaFace.detect_faces : Detects faces on given image and returns detected faecs with cordinate value as dict.
            detected_faces          : Consists of information on detected faces as dictionary type.
        
        '''
        detected_faces = RetinaFace.detect_faces(image)
        if type(detected_faces) == dict:
            for key in detected_faces.keys():
                    self.faces_detected += 1
                    # Get Cordinatese of each detected faces.
                    each_face  = detected_faces[key]
                    facial_area = each_face["facial_area"]

                    x1,y1,x2,y2 = facial_area[0],facial_area[1],facial_area[2],facial_area[3]
                    # roi = image[facial_area[1]:facial_area[3], facial_area[0]:facial_area[2]]
                    roi = image[y1:y2,x1:x2]
                    # Apply Gaussian blur to the ROI
                    blurred_roi = cv2.GaussianBlur(roi, (29, 29),11,11,cv2.BORDER_DEFAULT )
                    
                    # Blur each detected face.
                    # blur_image = cv2.GaussianBlur(roi,(15,15),0)
                    image[y1:y2, x1:x2] = blurred_roi
        
        self.image = image.copy()
    
    def blur_license_plates(self,vehicle_detected_region):
        '''

            Blurs License Plate from Vehicle Detected Regions
            Input  : vehicle_detected_region
                     vehicle_detected_region : ROI of Vehicle Detected Region
            Output : img
                     img : ROI of Vehicle Detected Region with bounding box around number plates
                     
        '''
        image = vehicle_detected_region
        detected_plates = self.license_detection_model(image)
        
        try:
            # print('here')
            detected_plates[0].boxes.data = torch.stack([box for box in detected_plates[0].boxes.data if (box[4] > 0.25)])
            boxes = detected_plates[0].boxes.data
            # confidences = torch.tensor([confidence[4] for confidence in detected_plates[0].boxes.data])
            # detected_plates[0].boxes.data = non_max_suppression(np.array(detected_plates[0].boxes.data),probs=confidences)
            # detected_plates[0].boxes.data = nms_pytorch(detected_plates[0].boxes.data,  0.5)
            boxes = detected_plates[0].boxes.data
            
            for box in boxes:
                self.license_plates_detected += 1
                x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])

                roi = image[y1:y2,x1:x2]
                # Apply Gaussian blur to the ROI
                blurred_roi = cv2.GaussianBlur(roi, (29, 29),11,11,cv2.BORDER_DEFAULT )  # Adjust the kernel size as needed

                # Replace the ROI with the blurred content
                image[y1:y2, x1:x2] = blurred_roi
            # self.plot_image(img)
            return image

        except Exception:
            traceback.print_exc(limit=1)
            return image
    
    def blur_license_plates_and_faces(self):
        '''
        
            Calls both the function blur_faces and blur_license_plates.
            Passes whole image to blur_faces function
            Detects Vehicle first and returns ROI of detected vehicles for license plate detection
            Returns the drawn image
        '''
        self.blur_faces()
    
        # Predicts on a image
        predict_image_v7 = self.yolo_model_v7(self.image)
        predict_image_v8 = self.yolo_model_v8(self.image)
        
        # Bounding list from v7 model                
        predict_image_v7_list = predict_image_v7.pandas().xyxy[0].values.tolist() 
        predict_image_v7_list = [[int(item) if index < 4 else item for index, item in enumerate(sublist)] for sublist in predict_image_v7_list]
        predict_image_v7_list = [list[:-1] for list in predict_image_v7_list]

        
        
        ### Checks if prediction is empty for all classes
        if (len(predict_image_v7_list) != 0 or len(predict_image_v8[0].boxes.data) != 0):
            try:
                
                # Filter only vehicles
                boxes_v7 = [box for box in predict_image_v7_list if ((int(box[5]) == 2) or (int(box[5]) == 3) or (int(box[5]) == 4) or (int(box[5]) == 5) or (int(box[5]) == 6) or (int(box[5]) == 7) or (int(box[5]) == 8)) and (box[4] > 0.25)]
                boxes_v7 = [[int(item) if index < 4 else item for index, item in enumerate(sublist)] for sublist in boxes_v7]
                
                # Filter only vehicles    
                boxes_v8 = torch.stack([box for box in predict_image_v8[0].boxes.data if ((int(box[5]) == 2) or (int(box[5]) == 3) or (int(box[5]) == 4) or (int(box[5]) == 5) or (int(box[5]) == 6) or (int(box[5]) == 7) or (int(box[5]) == 8)) and (box[4] > 0.25)])

                # Combine
                combined_boxes = boxes_v7 + boxes_v8.tolist()
                self.vehicles_detected_v7 = len(boxes_v7)
                self.vehicles_detected_v8 = len(boxes_v8)

                boxes = combined_boxes
                self.total_vehicles_detected = len(boxes)
                for box in boxes:
                    x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])

                    roi =  self.image[y1 : y2, x1 : x2]
                    
                    # self.plot_image(roi) 
                    self.image[y1 : y2, x1 : x2] = self.blur_license_plates(roi)

                # self.plot_image(self.image)
            except Exception as e:
                print(e)
                traceback.print_exc(limit=1)
                # self.plot_image(self.image)
        else:
            print("No Predictions")
            # self.plot_image(self.image)
        return self.image,self.total_vehicles_detected,self.vehicles_detected_v7,self.vehicles_detected_v8,self.license_plates_detected,self.faces_detected
             
    


