import base64
import datetime
import math
import os
import smtplib
import time
from threading import Thread
#import TextRecognitionModule
import sys
import torchvision
import pyodbc
import imutils
from PIL import Image
import cv2
import skimage
from tkinter import *
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import io
import torch
import numpy as np
from time import sleep
import pytorch_mask_rcnn as pmr
#from __future__ import unicode_literals
from PIL import ImageFont, ImageDraw, Image
ListCarNumbers=[]



def sqlQuery():

    server = '10.57.0.21'
    database = 'cars'
    username = 'sa'
    password = '********'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    cursor.execute("select * from car")
    ListCarNumbers.clear()
    print('Обновление номеров в памяти завершено')
    for row in cursor.fetchall():
        carID=str(row[0]).strip()
        carFIO = str(row[1]).strip()
        carNumber = str(row[2]).strip()
        carLocation = str(row[3]).strip()
        carPermissionTime = str(row[4]).strip()
        OnlyEnter = str(row[5]).strip()
        ListCarNumbers.append([carID,carFIO,carNumber,carLocation,carPermissionTime,OnlyEnter])



#now = datetime.datetime.now()
#dt_string = datetime.datetime.now.strftime("%d/%m/%Y %H:%M:%S")
#print("date and time =", dt_string)


def sqlQueryInsertEvents(Entrance,NR,user):
    dt=str(datetime.date.today())
    server = '10.57.0.21'
    database = 'cars'
    username = 'sa'
    password = '********'
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    cursor.execute("insert into[Cars].[dbo].[Events](DT, PlateNumber, [User], Lacation, Entrance)Values('"+datetime.datetime.now().strftime("%m-%d-%Y %H:%M:%S")+"', '"+NR+"', '"+user+"', 'ОБ', "+str(Entrance)+")")
    cnxn.commit()

def sql_insert_evcent_thread(Entrance,NR,user):
    try:
        sqlQueryInsertEvents(Entrance,NR,user)
    except:
        print("Insert event error")

def sql_thread():
    while True:
        sqlQuery()
        sleep(600)
th_sql = Thread(target=sql_thread)
th_sql.start()

from torchvision import transforms



@torch.no_grad()
def procRec(model,device,args,image):

    torch.cuda.synchronize()
    output = model(image)
    return output

@torch.no_grad()
def procRecAuto(model,d_test,device,args,image):

    #torch.cuda.synchronize()
    output = model(image)
    return output
@torch.no_grad()
def procRecAuto2(model,d_test,device,args,image):


    torch.cuda.synchronize()
    output = model(image)
    return output

@torch.no_grad()
def procRecAuto3(model,d_test,device,args,image):

    torch.cuda.synchronize()
    output = model(image)
    return output




  # S = time.time()
  # print('Dur: '+str(S-T))
#TextRecognitionModule.TextRec()

#exec(open("TextRecognition.py").read())

import requests








trafficlightEntr='red'
trafficlightExt='red'
svetoforExt  = 'green'
svetoforEntr  = 'green'
project_dir = os.path.dirname(os.path.abspath(__file__))
trafficRedFoto = cv2.imread(project_dir+'/red.jpg')
trafficGreenFoto = cv2.imread(project_dir+'/green.jpg')
url = "http://127.0.0.1:8080/Predict/"



#pastebin_url = r.text
#print("The pastebin URL is:%s" % pastebin_url)
"""
from tkinter import *
# create a tkinter window
root = Tk()
# Open window having dimension 100x100
root.geometry('200x100')
# Create a Button
btn = Button(root, text='Click me !', bd='5',
             command=root.destroy)
# Set the position of button on the top of window.
btn.pack(side='top')
root.mainloop()
"""

IPESP32='http://10.57.21.1/'
CamReadingPermission=False
trafficLightEntrance='red'
trafficLightExit='red'
requests.get(IPESP32 + 'offled5')
ReadNumberPlatePermission=True
def ShlakbaumOpen():
    try:
            res = requests.get(IPESP32 + 'onled5')
            print('Ответ - '+str(res.status_code))
            #sleep(0.2)
            #        if res.status_code == 200:
            res = requests.get(IPESP32+'on1')
            if res.status_code == 200:
                #requests.get(IPESP32 + 'onled5')
                print(' Команда но открытие шлакбаума отправлена! ')
                #global ReadNumberPlatePermission
                #ReadNumberPlatePermission=False
                global CamReadingPermission

                sleep(2.0)
                CamReadingPermission = True
                #requests.get(IPESP32 + 'onled5')
                WaitMoovingCarAfterOpeniniBarrierProcThread = Thread(target=WaitMoovingCarAfterOpeniniBarrierProc)
                WaitMoovingCarAfterOpeniniBarrierProcThread.start()
                sleep(1.5)
                #requests.get(IPESP32 + 'onled5')
                global isBarrierOpened
                isBarrierOpened=True

                print(' Шлакбаум отпрыт! ')
            else:
                print('не смог открыть шлакбаум')
    except:
        print('error1')
#timerPause=False
def WaitMoovingCarAfterOpeniniBarrierProc():
    print('запуск ожиджания движенния машины')
    try:
        global WaitMoovingCarAfterOpeniniBarrier
        WaitMoovingCarAfterOpeniniBarrier=True
        start_time = time.time()
        while (time.time() - start_time) < 24.0:
            if WaitMoovingCarAfterOpeniniBarrier ==False:
                print('breack')
                break

            sleep(0.1)
        global trafficlightEntr
        trafficlightEntr = 'red'
        global trafficlightExt
        trafficlightExt = 'red'
        sleep(4)
        WaitMoovingCarAfterOpeniniBarrier = False
        print('WaitMoovingCarAfterOpeniniBarrierProc  ожидание окончено')
    except:
        print('error221')



ShlakbaumOpenThread = Thread(target=ShlakbaumOpen)
NumberPlateText = ''
isBarrierOpened=False

IsMaveAutosCam3=True
IsMaveAutosCam2=True
IsMaveAutosCam1=True
IsMaveAutosCam4=True

testFlagCam1=False
testFlagCam2=False
testFlagCam3=False
testFlagCam4=False
EmergencyMode=False

#ChangeToRedFlag4=False

WaitMoovingCarAfterOpeniniBarrier=False




def main(args):




   device = torch.device("cuda")
   if device.type == "cuda":
        pmr.get_gpu_prop(show=True)
   #d_test = pmr.datasets(args.dataset, args.data_dir, "val2017", train=True) # set train=True for eval")
   device1 = torch.device("cuda:1")
   device2 = torch.device("cuda:2")
   num_classes = 2
   #num_classes_auto = 81

   model = pmr.maskrcnn_resnet50(False, num_classes).to(device2)
   model2 = pmr.maskrcnn_resnet50(False, num_classes).to(device1)

   #model_auto = pmr.maskrcnn_resnet50(False, num_classes_auto).to(device)
   #model_auto_2 = pmr.maskrcnn_resnet50(False, num_classes_auto).to(device)
   #model_auto_3 = pmr.maskrcnn_resnet50(False, num_classes_auto).to(device)



   checkpoint = torch.load(args.ckpt_path, map_location=device)
   checkpoint2 = torch.load(args.ckpt_path, map_location=device1)
   #checkpoint_auto1 = torch.load(args.ckpt_path_auto, map_location=device)
   #checkpoint_auto2 = torch.load(args.ckpt_path_auto, map_location=device)
   #checkpoint_auto3 = torch.load(args.ckpt_path_auto, map_location=device)

   model.load_state_dict(checkpoint["model"])
   model2.load_state_dict(checkpoint2["model"])
   #model_auto.load_state_dict(checkpoint_auto1["model"])
   #model_auto_2.load_state_dict(checkpoint_auto2["model"])
   #model_auto_3.load_state_dict(checkpoint_auto3["model"])

   del checkpoint
   del checkpoint2
   #del checkpoint_auto1
   #del checkpoint_auto2
   #del checkpoint_auto3
   torch.cuda.empty_cache()
   model.eval()
   model2.eval()
   #model_auto.eval()
   #model_auto_2.eval()
   #model_auto_3.eval()

   #model_FRCNN =torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
   model_FRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   model_FRCNN_1 =  torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   model_FRCNN_2 = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   """ 
   COCO_INSTANCE_CATEGORY_NAMES = [
       '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
       'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
       'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
       'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
       'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
       'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
       'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
       'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
   ]
   """
   model_FRCNN.eval()
   model_FRCNN_1.eval()
   model_FRCNN_2.eval()

   model_FRCNN.cuda(0)
   model_FRCNN_1.cuda(1)
   model_FRCNN_2.cuda(2)





   T = time.time()
   ########--------------------------------------------------------------------
   imagesArray=[]
   imagesArray_exitBarrier = []
   imagesArray_auto_1 = []
   imagesArray_auto_2 = []
   imagesArray_auto_3 = []
   imagesArray_auto_4 = []
   imagesArray_CamVerifyAutoUnderBarrier = []



   #th.start()

   #def func():
   def CamVerifyAutoUnderBarrier():
       cap = cv2.VideoCapture('rtsp://admin:gJ94Hp8z@10.57.20.245:554/ISAPI/Streaming/Channels/101')
       while True:

           try:
               #if isBarrierOpened == False:
                   #cap = cv2.VideoCapture('rtsp://10.57.20.245:554/user=admin&password=gJ94Hp8z&channel=1&stream=0?.sdp')
                   #cap = cv2.VideoCapture('rtsp://admin:gJ94Hp8z@10.57.20.245:554/ISAPI/Streaming/Channels/101')
                   #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                   #cap.set(cv2.CAP_PROP_FPS, 2)

                   while cap.isOpened() :
                       success, frame_inp_0 = cap.read()
                       if not success:
                           print("Ignoring empty camera frame. Datchik")
                           cap = cv2.VideoCapture('rtsp://admin:gJ94Hp8z@10.57.20.245:554/ISAPI/Streaming/Channels/101')
                           # cap = cv2.VideoCapture('rtsp://admin:gJ94Hp8z@10.57.0.67:554/ISAPI/Streaming/Channels/101')
                           cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                           continue

                       #frame_inp_0=imutils.resize(frame_inp_0, width=1280)
                       image_ = Image.fromarray(cv2.cvtColor(frame_inp_0, cv2.COLOR_BGR2RGB))
                       # image_=Image.open('21.jpg')
                       im = cv2.cvtColor(((np.array(image_))), cv2.COLOR_RGB2BGR)
                       image = transforms.ToTensor()(image_)
                       #global imagesArray_CamVerifyAutoUnderBarrier
                       if len(imagesArray_CamVerifyAutoUnderBarrier) > 10:
                           del imagesArray_CamVerifyAutoUnderBarrier[0]

                       im = im[200:880, 200:1620, :]
                       image = image[:, 200:880, 200:1620]

                       imagesArray_CamVerifyAutoUnderBarrier.append([image, im])
                       #if isBarrierOpened==True:
                       #    imagesArray_CamVerifyAutoUnderBarrier.clear()
                       #    cap.release()
                       #    break
                       sleep(0.01)

                       #cv2.imshow('imageBarrier', imutils.resize(im, width=800))
                       #cv2.waitKey(1)
                   else:
                        cap = cv2.VideoCapture('rtsp://admin:gJ94Hp8z@10.57.20.245:554/ISAPI/Streaming/Channels/101')
                        imagesArray_CamVerifyAutoUnderBarrier.clear()
                        sleep(0.1)


            #   else:
            #   sleep(0.1)

           except:
               print('error2')
           #if isBarrierOpened == True:
           #    break
           imagesArray_CamVerifyAutoUnderBarrier.clear()
           sleep(0.1)
       #cap.release()
       #cv2.destroyAllWindows()

   def CamEntrance():
       cap = cv2.VideoCapture('rtsp://10.57.20.8:554/user=admin&password=&channel=1&stream=0?.sdp')
       while True:
           try:

                   #cap = cv2.VideoCapture('rtsp://10.57.20.8:554/user=admin&password=&channel=1&stream=0?.sdp')
                   #cap = cv2.VideoCapture('rtsp://admin:gJ94Hp8z@10.57.0.67:554/ISAPI/Streaming/Channels/101')
                   #cap = cv2.VideoCapture('rtsp://admin:gJ94Hp8z@10.57.20.245:554/ISAPI/Streaming/Channels/101')

                   #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                   while cap.isOpened():
                       success, frame_inp_0 = cap.read()
                       if isBarrierOpened == False:
                           if not success:
                               print("Ignoring empty camera frame.Entrance")
                               #cap = cv2.VideoCapture('rtsp://10.57.20.8:554/user=admin&password=&channel=1&stream=0?.sdp')
                               #cap = cv2.VideoCapture('rtsp://admin:gJ94Hp8z@10.57.20.245:554/ISAPI/Streaming/Channels/101')
                               cap = cv2.VideoCapture('rtsp://10.57.20.8:554/user=admin&password=&channel=1&stream=0?.sdp')
                               cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                               continue

                           # frame_inp_0=imutils.resize(frame_inp_0, width=1280)

                           image_ = Image.fromarray(cv2.cvtColor(frame_inp_0, cv2.COLOR_BGR2RGB))
                           # image_=Image.open('21.jpg')
                           im = cv2.cvtColor(((np.array(image_))), cv2.COLOR_RGB2BGR)

                           image = transforms.ToTensor()(image_)
                           if len(imagesArray) > 10:
                               del imagesArray[0]
                           imagesArray.append([image, im])
                           #if isBarrierOpened == True:
                           #    imagesArray.clear()
                           #    cap.release()
                           #    break

                           sleep(0.01)
                       else:
                           sleep(0.09)

                       cv2.imshow('CamExt', imutils.resize(frame_inp_0, width=1024))
                       cv2.waitKey(1)

                   else:
                       cap = cv2.VideoCapture('rtsp://10.57.20.8:554/user=admin&password=&channel=1&stream=0?.sdp')
                       imagesArray.clear()
                       sleep(0.1)


           except:
               print('error3')
           #if isBarrierOpened == True:
           #    break
           sleep(0.1)

       #cv2.destroyAllWindows()
   def CamExitBarrier():
       cap = cv2.VideoCapture('rtsp://admin:12345678@10.57.20.78:554/ISAPI/Streaming/Channels/101')
       while True:
           try:

                   #cap = cv2.VideoCapture('rtsp://admin:12345678@10.57.20.78:554/ISAPI/Streaming/Channels/101')
                   #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                   while cap.isOpened():
                       success, frame_inp_0 = cap.read()
                       if isBarrierOpened == False:
                           if not success:
                               print("Ignoring empty camera frame.Exit")
                               cap = cv2.VideoCapture('rtsp://admin:12345678@10.57.20.78:554/ISAPI/Streaming/Channels/101')
                               cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                               continue

                           # frame_inp_0=imutils.resize(frame_inp_0, width=1280)

                           image_ = Image.fromarray(cv2.cvtColor(frame_inp_0, cv2.COLOR_BGR2RGB))
                           # image_=Image.open('21.jpg')
                           im = cv2.cvtColor(((np.array(image_))), cv2.COLOR_RGB2BGR)

                           image = transforms.ToTensor()(image_)
                           if len(imagesArray_exitBarrier) > 10:
                               del imagesArray_exitBarrier[0]
                           imagesArray_exitBarrier.append([image, im])
                           #sleep(0.08)
                           #if isBarrierOpened==True:
                           #    imagesArray_exitBarrier.clear()
                           #    cap.release()
                           #    break
                           sleep(0.01)
                       else:
                           sleep(0.09)

                       #cv2.imshow('CamExt1', imutils.resize(frame_inp_0, width=1024))
                       #cv2.waitKey(1)
                   else:
                       cap = cv2.VideoCapture('rtsp://admin:12345678@10.57.20.78:554/ISAPI/Streaming/Channels/101')
                       imagesArray_exitBarrier.clear()
                       sleep(0.1)

           except:
               print('error4')

           sleep(0.1)
           #if isBarrierOpened == True:
           #    break
       #cap.release()
   def CamEntranceAdvanced1():
       while True:
           try:
               #cap = cv2.VideoCapture('rtsp://10.57.20.45:554/user=admin&password=admin&channel=1&stream=0?.sdp')
               cap = cv2.VideoCapture('rtsp://admin:admin@10.57.20.45:554/RVi/1/1')
               cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
               while cap.isOpened():
                   success, frame_inp_0 = cap.read()
                   if not success:
                       print("Ignoring empty camera frame.")
                       #cap = cv2.VideoCapture('rtsp://10.57.20.45:554/user=admin&password=admin&channel=1&stream=0?.sdp')
                       cap = cv2.VideoCapture('rtsp://admin:admin@10.57.20.45:554/RVi/1/1')
                       cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                       continue

                   frame_inp_0=imutils.resize(frame_inp_0, width=1280)
                   image_ = Image.fromarray(cv2.cvtColor(frame_inp_0, cv2.COLOR_BGR2RGB))
                   # image_=Image.open('21.jpg')
                   im = cv2.cvtColor(((np.array(image_))), cv2.COLOR_RGB2BGR)
                   image = transforms.ToTensor()(image_)
                   if len(imagesArray_auto_1) > 10:
                       del imagesArray_auto_1[0]
                   imagesArray_auto_1.append([image, im])
                   sleep(0.01)

                   #image_ = cv2.cvtColor(np.array(image_), cv2.COLOR_RGB2BGR)
                   #global NumberPlateText
                   #font = cv2.FONT_HERSHEY_SIMPLEX
                   #org = (100, 100)
                   #fontScale = 3
                   #color = (0, 0, 255)
                   #thickness = 8
                   #if (len(NumberPlateText)) >= 8:
                   #    image_ = cv2.putText(image_, NumberPlateText, org, font,
                   #                         fontScale, color, thickness, cv2.LINE_AA)

                   #cv2.imshow('CamEntranceAdvanced', imutils.resize(image_, width=1024))
                   #cv2.waitKey(1)

                   # im=cv2.cvtColor(splash, cv2.COLOR_BGR2GRAY)

           except:
               imagesArray_auto_1.clear()
               print('error5')
           imagesArray_auto_1.clear()
           print('--------------------------------------------------')
           sleep(0.05)


   def CamEntranceAdvanced3():
       cap = cv2.VideoCapture('rtsp://admin:@10.57.20.176:554/ISAPI/Streaming/Channels/101')
       while True:
           try:
               #cap = cv2.VideoCapture('rtsp://admin:@10.57.20.176:554/ISAPI/Streaming/Channels/101')

               #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
               while cap.isOpened():
                   success, frame_inp_0 = cap.read()
                   if not success:
                       print("Ignoring empty camera frame.Cam3")
                       cap = cv2.VideoCapture('rtsp://admin:@10.57.20.176:554/ISAPI/Streaming/Channels/101')
                       cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                       continue

                   frame_inp_0=imutils.resize(frame_inp_0, width=1280)
                   image_ = Image.fromarray(cv2.cvtColor(frame_inp_0, cv2.COLOR_BGR2RGB))

                   # image_=Image.open('21.jpg')
                   im = cv2.cvtColor(((np.array(image_))), cv2.COLOR_RGB2BGR)
                   image = transforms.ToTensor()(image_)

                   if len(imagesArray_auto_3) > 8:
                       del imagesArray_auto_3[0]
                   imagesArray_auto_3.append([image, im])
                   sleep(0.01)

                   #cv2.imshow('CamEntranceAdvanced3', imutils.resize(frame_inp_0, width=1024))
                   #cv2.waitKey(1)
               else:
                   cap = cv2.VideoCapture('rtsp://admin:@10.57.20.176:554/ISAPI/Streaming/Channels/101')
                   imagesArray_auto_3.clear()
                   sleep(0.1)



           except:
               imagesArray_auto_3.clear()
               print('error6')
           imagesArray_auto_3.clear()
           print('--------------------------------------------------')
           sleep(0.05)

   def CamEntranceAdvanced2():
       cap = cv2.VideoCapture('rtsp://admin:admin1987@10.57.20.237:554/0')
       while True:
           try:
               #cap = cv2.VideoCapture('rtsp://10.57.20.28:554/user=admin&password=12345678&channel=1&stream=0?.sdp')
               #cap = cv2.VideoCapture('rtsp://admin:admin1987@10.57.20.237:554/0')

               #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
               while cap.isOpened():
                   success, frame_inp_0 = cap.read()
                   if not success:
                       print("Ignoring empty camera frame.Cam2")
                       #cap = cv2.VideoCa`pture('rtsp://10.57.28.45:554/user=admin&password=12345678&channel=1&stream=0?.sdp')
                       cap = cv2.VideoCapture('rtsp://admin:admin1987@10.57.20.237:554/0')
                       cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                       continue

                   frame_inp_0 = imutils.resize(frame_inp_0, width=1280)
                   image_ = Image.fromarray(cv2.cvtColor(frame_inp_0, cv2.COLOR_BGR2RGB))
                   # image_=Image.open('21.jpg')
                   im = cv2.cvtColor(((np.array(image_))), cv2.COLOR_RGB2BGR)
                   image = transforms.ToTensor()(image_)
                   if len(imagesArray_auto_2) >8:
                       del imagesArray_auto_2[0]
                   imagesArray_auto_2.append([image, im])
                   sleep(0.01)

                   #image_ = cv2.cvtColor(np.array(image_), cv2.COLOR_RGB2BGR)
                   # global NumberPlateText
                   # font = cv2.FONT_HERSHEY_SIMPLEX
                   # org = (100, 100)
                   # fontScale = 3
                   # color = (0, 0, 255)
                   # thickness = 8
                   # if (len(NumberPlateText)) >= 8:
                   #    image_ = cv2.putText(image_, NumberPlateText, org, font,
                   #                         fontScale, color, thickness, cv2.LINE_AA)

                   # cv2.imshow('CamEntranceAdvanced', imutils.resize(image_, width=1024))
                   # cv2.waitKey(1)

                   # im=cv2.cvtColor(splash, cv2.COLOR_BGR2GRAY)
               else:
                   cap = cv2.VideoCapture('rtsp://admin:admin1987@10.57.20.237:554/0')
                   imagesArray_auto_2.clear()
                   sleep(0.1)

           except:
               print('error7')
               imagesArray_auto_2.clear()
           imagesArray_auto_2.clear()
           print('--------------------------------------------------')
           sleep(0.05)
   def CamEntranceAdvanced4():
       cap = cv2.VideoCapture('rtsp://admin:12345678@10.57.20.28:554/0')
       while True:
           try:
               #cap = cv2.VideoCapture('rtsp://10.57.20.45:554/user=admin&password=admin&channel=1&stream=0?.sdp')
               #cap = cv2.VideoCapture('rtsp://admin:12345678@10.57.20.28:554/0')
               #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
               while cap.isOpened():
                   success, frame_inp_0 = cap.read()
                   if not success:
                       print("Ignoring empty camera frame.Cam4")
                       #cap = cv2.VideoCapture('rtsp://10.57.20.45:554/user=admin&password=admin&channel=1&stream=0?.sdp')
                       cap = cv2.VideoCapture('rtsp://admin:12345678@10.57.20.28:554/0')
                       cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                       continue

                   frame_inp_0=imutils.resize(frame_inp_0, width=1280)
                   image_ = Image.fromarray(cv2.cvtColor(frame_inp_0, cv2.COLOR_BGR2RGB))
                   # image_=Image.open('21.jpg')
                   im = cv2.cvtColor(((np.array(image_))), cv2.COLOR_RGB2BGR)
                   image = transforms.ToTensor()(image_)
                   if len(imagesArray_auto_4) > 8:
                       del imagesArray_auto_4[0]
                   imagesArray_auto_4.append([image, im])
                   sleep(0.01)

                   #image_ = cv2.cvtColor(np.array(image_), cv2.COLOR_RGB2BGR)
                   #global NumberPlateText
                   #font = cv2.FONT_HERSHEY_SIMPLEX
                   #org = (100, 100)
                   #fontScale = 3
                   #color = (0, 0, 255)
                   #thickness = 8
                   #if (len(NumberPlateText)) >= 8:
                   #    image_ = cv2.putText(image_, NumberPlateText, org, font,
                   #                         fontScale, color, thickness, cv2.LINE_AA)

                   #cv2.imshow('CamEntranceAdvanced', imutils.resize(image_, width=1024))
                   #cv2.waitKey(1)

                   # im=cv2.cvtColor(splash, cv2.COLOR_BGR2GRAY)
               else:
                   cap = cv2.VideoCapture('rtsp://admin:12345678@10.57.20.28:554/0')
                   imagesArray_auto_4.clear()
                   sleep(0.1)
           except:
               imagesArray_auto_4.clear()
               print('error81')
           imagesArray_auto_4.clear()
           print('--------------------------------------------------')
           sleep(0.05)
   def getRoad1(x1,y1,x2,y2,x,y):


      #x2=1280
      #y2=500
      #y1=220
      yfunc=((y2-y1)/x2)*x+y1
      if y<yfunc:
          return True
      else:
          return False


   def CamEntranceAdvanced1Read():
       global IsMaveAutosCam1
       global testFlagCam1
       ListCars = []
       while True:
           #start_time = time.time()
           try:
               image = imagesArray_auto_1[len(imagesArray_auto_1) - 1][0]
               im_ = imagesArray_auto_1[len(imagesArray_auto_1) - 1][1]
               image = image.cuda()
               #image = image.to(device)
               #predict = procRecAuto(model_auto, d_test, device, args, image)
               x1 = 0
               y1 = 90
               x2 = 1280
               y2 = 620

               x21 = 0
               y21 = 200
               x22 = 450
               y22 = 720

               x31 = 0
               y31 = 187
               x32 = 1280
               y32 = 120


               tik = 2
               BadRetX1 = 430
               BadRetY1 = 260
               BadRetX2 = 480
               BadRetY2 = 290

               BadRetX11 = 180
               BadRetY11 = 165
               BadRetX12 = 240
               BadRetY12 = 200

               BadRetX21 = 0
               BadRetY21 = 710
               BadRetX22 = 450
               BadRetY22 = 720

               im_ = cv2.rectangle(im_, (BadRetX1, BadRetY1), (BadRetX2, BadRetY2), (255, 0, 0), tik)
               im_ = cv2.rectangle(im_, (BadRetX11, BadRetY11), (BadRetX12, BadRetY12), (255, 0, 0), tik)
               im_ = cv2.rectangle(im_, (BadRetX21, BadRetY21), (BadRetX22, BadRetY22), (255, 0, 0), tik)

               if CamReadingPermission == True:
                   pred = model_FRCNN([image])

                   if len([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]) > 0:
                    pass


                   cars = False
                   for i in range(len([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())])):
                       score=list(pred[0]['scores'].detach().cpu().numpy())[i]
                       #if not (int(list(pred[0]['labels'].cpu().numpy())[i])  in [51,9,15,16,34,62,1,35,27,86,72,28,7,31,36,33,42,64,44,17,41,2,11,84,38,16,37,76,18]):
                       if (int(list(pred[0]['labels'].cpu().numpy())[i]) in [3,4,6,8]):
                                 if score>0.60:
                                   #print('1111111111111111111111111'+str(predict['labels'].cpu()))
                                   box = pred[0]['boxes'].cpu()[i]
                                   box= [int(box[0]),int(box[1]),int(box[2]),int(box[3])]
                                   #MainAutoPoint = (int(box[0]), int(box[1]) + int((int(box[3]) - int(box[1])) / 2))
                                   MainAutoPoint = (int(box[0]) + int((int(box[2]) - int(box[0])) / 2), int(box[3]))









                                   if ((not (getRoad3(x21,y21,x22,y22,MainAutoPoint[0],MainAutoPoint[1])))&((pred[0]['labels'].cpu().numpy())[i]==3)) or  (getRoad3(x31,y31,x32,y32,MainAutoPoint[0],MainAutoPoint[1])) or getRoad1(x1,y1,x2,y2,MainAutoPoint[0],MainAutoPoint[1]):
                                     im_ = cv2.circle(im_, MainAutoPoint, 5, (0, 0, 255), 2)
                                     im_ = cv2.rectangle(im_, (box[0],box[1]), (box[2],box[3]), (255,0,0), tik)
                                   else:
                                       if (MainAutoPoint[0] > BadRetX1) & (MainAutoPoint[0] < BadRetX2) & (
                                               MainAutoPoint[1] > BadRetY1) & (MainAutoPoint[1] < BadRetY2) or ((MainAutoPoint[0] > BadRetX21) & (MainAutoPoint[0] < BadRetX22) & (
                                               MainAutoPoint[1] > BadRetY21) & (MainAutoPoint[1] < BadRetY22))or ((MainAutoPoint[0] > BadRetX11) & (MainAutoPoint[0] < BadRetX12) & (
                                               MainAutoPoint[1] > BadRetY11) & (MainAutoPoint[1] < BadRetY12)):
                                           im_ = cv2.circle(im_, MainAutoPoint, 5, (0, 0, 255), 2)
                                           im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), tik)
                                       else:
                                            testFlagCam1 = True
                                            cars = True
                                            im_ = cv2.circle(im_, MainAutoPoint, 5, (0, 0, 255), 2)
                                            im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), tik)

                                   font = cv2.FONT_HERSHEY_SIMPLEX
                                   org = (100, 100)
                                   fontScale = 1
                                   color = (0, 0, 255)
                                   thickness = 3
                                   im_ = cv2.putText(im_, str(int(list(pred[0]['labels'].cpu().numpy())[i])), MainAutoPoint,
                                                     font,fontScale, color, thickness, cv2.LINE_AA)
                                   #im_ = cv2.putText(im_, str(list((pred[0]['scores'].cpu().detach()).numpy())[i]), MainAutoPoint, font,
                                   #      fontScale, color, thickness, cv2.LINE_AA)
                   if cars:
                       ListCars.append(1)
                   else:
                       ListCars.append(0)

                   if len(ListCars) > 15:
                       del ListCars[0]


                   if len(ListCars) >= 15:
                       if 1 in ListCars:
                           IsMaveAutosCam1 = True
                       else:
                           IsMaveAutosCam1 = False
                   else:
                       IsMaveAutosCam1 = True
               else:
                   ListCars.clear()
                   IsMaveAutosCam1 = True


               im_ = cv2.line(im_, (x1,y1), (x2,y2), (0,255,0), 2)
               im_ = cv2.line(im_, (x21, y21), (x22, y22), (0, 255, 0), 2)
               im_ = cv2.line(im_, (x31, y31), (x32, y32), (0, 255, 0), 2)
               #cv2.imshow('CamEntranceAdvancedRead', imutils.resize(im_, width=1024))
               #cv2.waitKey(1)
               if EmergencyMode==True:
                   b, g, r, a = 0, 0, 255, 0
                   fontpath = "a_Albionic.ttf"
                   font = ImageFont.truetype(fontpath, 32)
                   img_pil = Image.fromarray(im_)
                   draw = ImageDraw.Draw(img_pil)
                   draw.text((50, 80), "Система перешла в аварийный режим.", font=font, fill=(b, g, r, a))
                   draw.text((100, 120), " Шлакбаум заблокирован!", font=font, fill=(b, g, r, a))
                   im_ = np.array(img_pil)
               cv2.startWindowThread()
               cv2.namedWindow("CamEntranceAdvancedRead1")
               cv2.imshow("CamEntranceAdvancedRead1", imutils.resize(im_, width=840))

               if CamReadingPermission == False:
                   sleep(0.1)
               #cv2.imshow('a frame', im_)  # a frame is the title name of the display window
               #cv2.waitKey(0)
               #cv2.destroyAllWindows()

           except:
               cv2.startWindowThread()
               cv2.namedWindow("CamEntranceAdvancedRead1")
               cv2.imshow("CamEntranceAdvancedRead1", imutils.resize(np.zeros((720,1280,3), np.uint8), width=840))

               ListCars.clear()
               IsMaveAutosCam1 = True
               # NumberPlateText=''
               print('error8')
               sleep(0.05)
           #print(str( (time.time() - start_time)))
   def getRoad2(x1,y1,x2,y2,x,y):
      #x2=1280
      #y2=500
      #y1=220
      yfunc=((y2-y1)/x2)*x+y1
      if y<yfunc:
          return True
      else:
          return False
   def getRoadUpDown(x1,y1,x2,y2,x,y):
      #x2=1280
      #y2=500
      #y1=220
      #yfunc=((y2-y1)/x2)*x+y1
      yfunc =(x-x1)/(x2-x1)*(y2-y1)+y1
      if y<yfunc:
          return True
      else:
          return False
   def getRoad3(x1,y1,x2,y2,x,y):
      #x2=1280
      #y2=500
      #y1=220
      #yfunc=((y2-y1)/x2)*x+y1
      #yfunc =(x-x1)/(x2-x1)*(y2-y1)+y1
      xfunc = (y - y1) / (y2 - y1) * (x2 - x1) + x1
      if x>xfunc:
          return True
      else:
          return False
   def isLineCarIntoRec(x1,y1,x2,y2,x21,y21,x22,y22,p1,p2):
       #if p2[1]>=720:
       #    return True
       xP1=p1[0]
       yP1=p1[1]
       xP2=p2[0]
       yP2=p2[1]
       #1 вариант обе точки слева выше обеих линий
       if getRoad3(x1,y1,x2,y2,xP1,yP1) & getRoad3(x21,y21,x22,y22,xP1,yP1)&  getRoad3(x1,y1,x2,y2,xP2,yP2) & getRoad3(x21,y21,x22,y22,xP2,yP2):
           return False
       # 2 вариант верхняя точка выше обеих линий . нижняя точка между лдиниями находится
       if getRoad3(x1,y1,x2,y2,xP1,yP1) & ((getRoad3(x21,y21,x22,y22,xP1,yP1)))& (not  getRoad3(x1,y1,x2,y2,xP2,yP2)) & getRoad3(x21,y21,x22,y22,xP2,yP2):
           return True
       # 3 вариант обе левые точки между линиями находятся
       if (not(getRoad3(x1, y1, x2, y2, xP1, yP1))) & ((getRoad3(x21, y21, x22, y22, xP1, yP1))) & (not getRoad3(x1, y1, x2, y2, xP2, yP2)) & getRoad3(x21, y21, x22, y22, xP2, yP2):
           return True

       # 4 вариант верхняя точка между линиями, нижняя ниже обеих линий
       if (not (getRoad3(x1, y1, x2, y2, xP1, yP1))) & ((getRoad3(x21, y21, x22, y22, xP1, yP1))) & (not getRoad3(x1, y1, x2, y2, xP2, yP2)) & (not(getRoad3(x21, y21, x22, y22, xP2, yP2))):
               return True

       # 5 вариант верхняя точка выше вех лиий, нижняя ниже обеих линий
       if ( (getRoad3(x1, y1, x2, y2, xP1, yP1))) & ((getRoad3(x21, y21, x22, y22, xP1, yP1))) & (not getRoad3(x1, y1, x2, y2, xP2, yP2)) & (not (getRoad3(x21, y21, x22, y22, xP2, yP2))):
           return True
       # 6 вариант обе точки ниже всех линий
       if (not(getRoad3(x1, y1, x2, y2, xP1, yP1))) & (not(getRoad3(x21, y21, x22, y22, xP1, yP1))) & (not getRoad3(x1, y1, x2, y2, xP2, yP2)) & (not (getRoad3(x21, y21, x22, y22, xP2, yP2))):
               return False




   def CamEntranceAdvanced2Read():
       ListCars = []
       global IsMaveAutosCam2
       global testFlagCam2
       while True:
           #start_time = time.time()
           try:
               image = imagesArray_auto_2[len(imagesArray_auto_2) - 1][0]
               im_ = imagesArray_auto_2[len(imagesArray_auto_2) - 1][1]
               x21 = 445
               y21 = 0
               x22 = 560
               y22 = 720

               x1 = 485
               y1 = 0
               x2 = 620
               y2 = 740

               x31 = 0
               y31 = 200
               x32 = 1280
               y32 = 300

               x41 = 0
               y41 = 250
               x42 = 1280
               y42 = 280


               tik = 2
               #BadRetX1 = 885
               #BadRetY1 = 70
               #BadRetX2 = 1280
               #BadRetY2 = 380

               #BadRetX21 = 0
               #BadRetY21 = 20
               #BadRetX22 = 100
               #BadRetY22 = 720

               #BadRetX31 = 830
               #BadRetY31 = 70
               #BadRetX32 = 1280
               #BadRetY32 = 220

               #im_ = cv2.rectangle(im_, (BadRetX1, BadRetY1), (BadRetX2, BadRetY2), (255, 0, 0), tik)
               #im_ = cv2.rectangle(im_, (BadRetX21, BadRetY21), (BadRetX22, BadRetY22), (255, 0, 0), tik)
               #im_ = cv2.rectangle(im_, (BadRetX31, BadRetY31), (BadRetX32, BadRetY32), (255, 0, 0), tik)
               if CamReadingPermission == True:

                   image = image.cuda(1)
                   pred = model_FRCNN_1([image])

                   if len([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]) > 0:
                    pass


                   cars=False
                   for i in range(len([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())])):
                       score=list(pred[0]['scores'].detach().cpu().numpy())[i]
                       #if not (int(list(pred[0]['labels'].cpu().numpy())[i])  in [51,9,15,16,34,62,1,35,27,86,72,28,7,31,36,33,42,64,44,17,41,2,11,84,38,16,37,76,18]):
                       if (int(list(pred[0]['labels'].cpu().numpy())[i]) in [3,4,6,8]):
                           if score > 0.68:
                               # print('1111111111111111111111111'+str(predict['labels'].cpu()))
                               box = pred[0]['boxes'].cpu()[i]
                               box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                               MainAutoPoint1 = (int(box[0]), int((box[1]) + (int(int(box[3]) - int(box[1])) / 2)))
                               MainAutoPoint2 = (int(box[2]), int((box[1]) + (int(int(box[3]) - int(box[1])) / 2)))

                               if (not (isLineCarIntoRec(x1, y1, x2, y2, x21, y21, x22, y22, MainAutoPoint2,MainAutoPoint1)))or (getRoadUpDown(x41, y41, x42, y42, box[2], box[3])):
                                   im_ = cv2.circle(im_, MainAutoPoint1, 3, (255, 0, 0), 3)
                                   im_ = cv2.circle(im_, MainAutoPoint2, 3, (255, 0, 0), 3)
                                   im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), tik)
                                   if not getRoadUpDown(x31, y31, x32, y32, box[2], box[3]):
                                       testFlagCam2 = True
                                       im_ = cv2.circle(im_, MainAutoPoint1, 3, (255, 255, 0), 3)
                                       im_ = cv2.circle(im_, MainAutoPoint2, 3, (255, 255, 0), 3)
                                       im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), tik)
                               else:
                                   # global     WaitMoovingCarAfterOpeniniBarrier
                                   # WaitMoovingCarAfterOpeniniBarrier = False
                                   testFlagCam2 = True
                                   cars = True
                                   im_ = cv2.circle(im_, MainAutoPoint1, 3, (0, 255, 0), 3)
                                   im_ = cv2.circle(im_, MainAutoPoint2, 3, (0, 255, 0), 3)
                                   im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), tik)

                               CenterCarPoint1 = (int(box[0]) + int((int(box[2]) - int(box[0])) / 3),
                                                  int((box[1]) + (int(int(box[3]) - int(box[1])) / 2) * 1))
                               CenterCarPoint = (int(box[0]) + int((int(box[2]) - int(box[0])) / 3 * 2),
                                                 int((box[1]) + (int(int(box[3]) - int(box[1])) / 2)))

                               if (((isLineCarIntoRec(x1, y1, x2, y2, x21, y21, x22, y22, (CenterCarPoint),(CenterCarPoint1))))) and  (not((getRoadUpDown(x41, y41, x42, y42,CenterCarPoint1[0],CenterCarPoint1[1])))):
                                   # print('333333333333333333333333333333333333333333333333333')
                                   global WaitMoovingCarAfterOpeniniBarrier
                                   WaitMoovingCarAfterOpeniniBarrier = False
                                   im_ = cv2.line(im_, CenterCarPoint, CenterCarPoint1, (0, 255, 0), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint, 3, (0, 255, 0), 3)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 3, (0, 255, 0), 3)
                                   global trafficlightEntr
                                   trafficlightEntr = 'red'
                                   global trafficlightExt
                                   trafficlightExt = 'red'
                                   # cv2.imwrite('temp/1' + str(time.time()) + '.jpg', im_)
                               else:
                                   im_ = cv2.line(im_, CenterCarPoint, CenterCarPoint1, (255, 0, 0), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint, 3, (255, 0, 0), 3)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 3, (255, 0, 0), 3)

                               """ 
                               if not getRoad3(x1, y1, x2, y2, CenterCarPoint[0], CenterCarPoint[1])or not getRoad3(x1, y1, x2, y2, CenterCarPoint1[0], CenterCarPoint1[1]):
                                   global WaitMoovingCarAfterOpeniniBarrier
                                   WaitMoovingCarAfterOpeniniBarrier = False
                                   im_ = cv2.circle(im_, CenterCarPoint, 2, (0, 0, 255), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 2, (0, 0, 255), 2)
                                   global trafficlightEntr
                                   trafficlightEntr = 'red'
                               else:
                                   im_ = cv2.circle(im_, CenterCarPoint, 2, (255, 0, 255), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 2, (255, 0, 255), 2)
                               """
                               # im_ = cv2.circle(im_, (box[0], box[1]), 5, (0, 0, 255), 2)
                               # im_ = cv2.circle(im_, (box[0], box[3]), 5, (0, 0, 255), 2)

                               # im_ = cv2.circle(im_, (box[2], box[1]), 5, (0, 255, 255), 2)
                               # im_ = cv2.circle(im_, (box[2], box[3]), 5, (0, 255, 255), 2)

                               font = cv2.FONT_HERSHEY_SIMPLEX
                               fontScale = 1
                               color = (0, 0, 255)
                               thickness = 3
                               im_ = cv2.putText(im_, str(score),
                                                 (box[0], box[1] + 20),
                                                 font, fontScale, color, thickness, cv2.LINE_AA)


                   if cars:
                       ListCars.append(1)
                   else:
                       ListCars.append(0)

                   if len(ListCars)>8:
                       del ListCars[0]

                   if len(ListCars) >= 8:
                       if 1 in ListCars:
                           IsMaveAutosCam2 = True
                       else:
                           IsMaveAutosCam2 = False
                   else:
                       IsMaveAutosCam2 = True
               else:

                   ListCars.clear()
                   IsMaveAutosCam2 = True


               im_ = cv2.line(im_, (x1, y1), (x2, y2), (0, 255, 0), 2)
               im_ = cv2.line(im_, (x21, y21), (x22, y22), (0, 255, 0), 2)
               im_ = cv2.line(im_, (x31, y31), (x32, y32), (255, 0, 0), 1)
               im_ = cv2.line(im_, (x41, y41), (x42, y42), (0, 255, 0), 2)
               global trafficRedFoto
               global trafficGreenFoto
               if trafficlightEntr=='red':
                    im_[50+0:50+151, 50+0:50+81] = trafficRedFoto
               else:

                   im_[50 + 0:50 + 151, 50 + 0:50 + 81] = trafficGreenFoto

               if trafficlightExt=='red':
                    im_[50+0:50+0+151, 50+1100:50+1100+81] = trafficRedFoto
               else:
                   im_[50 + 0:50 +0+ 151, 50 + 1100:50 +1100+ 81] = trafficGreenFoto


               #cv2.imshow('CamEntranceAdvancedRead2', imutils.resize(im_, width=1024))
               #cv2.waitKey(1)
               if EmergencyMode==True:
                   b, g, r, a = 0, 0, 255, 0
                   fontpath =project_dir+ "/a_Albionic.ttf"
                   font = ImageFont.truetype(fontpath, 32)
                   img_pil = Image.fromarray(im_)
                   draw = ImageDraw.Draw(img_pil)
                   draw.text((50, 80), "Система перешла в аварийный режим.", font=font, fill=(b, g, r, a))
                   draw.text((100, 120), " Шлакбаум заблокирован!", font=font, fill=(b, g, r, a))
                   im_ = np.array(img_pil)
               cv2.startWindowThread()
               cv2.namedWindow("CamEntranceAdvancedRead2")
               cv2.imshow("CamEntranceAdvancedRead2", imutils.resize(im_, width=800))

               if CamReadingPermission == False:
                   sleep(0.1)


           except:
               cv2.startWindowThread()
               cv2.namedWindow("CamEntranceAdvancedRead2")
               cv2.imshow("CamEntranceAdvancedRead2", imutils.resize(np.zeros((720,1280,3), np.uint8), width=800))

               ListCars.clear()
               IsMaveAutosCam2 = True
               # NumberPlateText=''
               print('error9')
               sleep(0.3)
           #print(str( (time.time() - start_time)))
               #sleep(0.01)

   def CamEntranceAdvanced3Read():
       ListCars = []
       global IsMaveAutosCam3
       global testFlagCam3
       while True:
           #start_time = time.time()
           try:
               image = imagesArray_auto_3[len(imagesArray_auto_3) - 1][0]
               im_ = imagesArray_auto_3[len(imagesArray_auto_3) - 1][1]
               x1 = 515
               y1 = 0
               x2 = 620
               y2 = 720

               x31 = 500
               y31 = 0
               x32 = 1280
               y32 = 400

               x41 = 0
               y41 = 180+15
               x42 = 1280
               y42 = 222+15


               x21 = 460
               y21 = 0
               x22 = 540
               y22 = 720
               """ 
               BadRetX1 = 200
               BadRetY1 = 100
               BadRetX2 = 460
               BadRetY2 = 300

               BadRetX21 = 450
               BadRetY21 = 150
               BadRetX22 = 540
               BadRetY22 = 240

               BadRetX31 = 750
               BadRetY31 = 690
               BadRetX32 = 1280
               BadRetY32 = 720
               """

               tik = 2
               #im_ = cv2.rectangle(im_, (BadRetX1, BadRetY1), (BadRetX2, BadRetY2), (255, 0, 0), tik)
               #im_ = cv2.rectangle(im_, (BadRetX21, BadRetY21), (BadRetX22, BadRetY22), (255, 0, 0), tik)
               #im_ = cv2.rectangle(im_, (BadRetX31, BadRetY31), (BadRetX32, BadRetY32), (255, 0, 0), tik)

               if CamReadingPermission == True:

                   image = image.cuda(2)
                   pred = model_FRCNN_2([image])

                   if len([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]) > 0:
                    pass

                   cars = False
                   for i in range(len([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())])):
                       score=list(pred[0]['scores'].detach().cpu().numpy())[i]
                       #if not (int(list(pred[0]['labels'].cpu().numpy())[i])  in [51,9,15,16,34,62,1,35,27,86,72,28,7,31,36,33,42,64,44,17,41,2,11,84,38,16,37,76,18]):
                       if (int(list(pred[0]['labels'].cpu().numpy())[i]) in [3,4,6,8]):
                           if score > 0.68:
                               # print('1111111111111111111111111'+str(predict['labels'].cpu()))
                               box = pred[0]['boxes'].cpu()[i]
                               box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                               MainAutoPoint1 = (int(box[0]) , int((box[1])+(int(int(box[3])-int(box[1]))/2)))
                               MainAutoPoint2 = (int(box[2]) , int((box[1])+(int(int(box[3])-int(box[1]))/2)))

                               if (not (isLineCarIntoRec(x1, y1, x2, y2, x21, y21, x22, y22, MainAutoPoint2, MainAutoPoint1)))or (getRoadUpDown(x41, y41, x42, y42, box[2], box[3])) :
                                   im_ = cv2.circle(im_, MainAutoPoint1, 3, (255, 0, 0), 3)
                                   im_ = cv2.circle(im_, MainAutoPoint2, 3, (255, 0, 0), 3)
                                   im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), tik)
                                   if not getRoadUpDown(x31, y31, x32, y32, box[2], box[3]):
                                       testFlagCam3 = True
                                       im_ = cv2.circle(im_, MainAutoPoint1, 3, (255, 255, 0), 3)
                                       im_ = cv2.circle(im_, MainAutoPoint2, 3, (255, 255, 0), 3)
                                       im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), tik)
                               else:
                                   # global     WaitMoovingCarAfterOpeniniBarrier
                                   # WaitMoovingCarAfterOpeniniBarrier = False
                                   testFlagCam3 = True
                                   cars = True
                                   im_ = cv2.circle(im_, MainAutoPoint1, 3, (0, 255, 0), 3)
                                   im_ = cv2.circle(im_, MainAutoPoint2, 3, (0, 255, 0), 3)
                                   im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), tik)

                               CenterCarPoint1 = (int(box[0])+int((int(box[2])-int(box[0]))/3) , int((box[1])+(int(int(box[3])-int(box[1]))/2)*1))
                               CenterCarPoint = (int(box[0])+int((int(box[2])-int(box[0]))/3*2) , int((box[1])+(int(int(box[3])-int(box[1]))/2)))

                               if (( (isLineCarIntoRec(x1, y1, x2, y2, x21, y21, x22, y22, (CenterCarPoint),(CenterCarPoint1)))))and (not((getRoadUpDown(x41, y41, x42, y42,CenterCarPoint1[0],CenterCarPoint1[1])))):
                                   #print('333333333333333333333333333333333333333333333333333')
                                   global WaitMoovingCarAfterOpeniniBarrier
                                   WaitMoovingCarAfterOpeniniBarrier = False
                                   im_ = cv2.line(im_, CenterCarPoint, CenterCarPoint1, (0, 255, 0), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint, 3, (0, 255, 0), 3)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 3, (0, 255, 0), 3)
                                   global trafficlightEntr
                                   trafficlightEntr = 'red'
                                   global trafficlightExt
                                   trafficlightExt = 'red'
                                   #cv2.imwrite('temp/1' + str(time.time()) + '.jpg', im_)
                               else:
                                   im_ = cv2.line(im_, CenterCarPoint, CenterCarPoint1, (255, 0, 0), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint, 3, (255, 0, 0), 3)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 3, (255, 0, 0), 3)

                               """ 
                               if not getRoad3(x1, y1, x2, y2, CenterCarPoint[0], CenterCarPoint[1])or not getRoad3(x1, y1, x2, y2, CenterCarPoint1[0], CenterCarPoint1[1]):
                                   global WaitMoovingCarAfterOpeniniBarrier
                                   WaitMoovingCarAfterOpeniniBarrier = False
                                   im_ = cv2.circle(im_, CenterCarPoint, 2, (0, 0, 255), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 2, (0, 0, 255), 2)
                                   global trafficlightEntr
                                   trafficlightEntr = 'red'
                               else:
                                   im_ = cv2.circle(im_, CenterCarPoint, 2, (255, 0, 255), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 2, (255, 0, 255), 2)
                               """
                               # im_ = cv2.circle(im_, (box[0], box[1]), 5, (0, 0, 255), 2)
                               # im_ = cv2.circle(im_, (box[0], box[3]), 5, (0, 0, 255), 2)

                               # im_ = cv2.circle(im_, (box[2], box[1]), 5, (0, 255, 255), 2)
                               # im_ = cv2.circle(im_, (box[2], box[3]), 5, (0, 255, 255), 2)

                               font = cv2.FONT_HERSHEY_SIMPLEX
                               fontScale = 1
                               color = (0, 0, 255)
                               thickness = 3
                               im_ = cv2.putText(im_, str(score),
                                                 (box[0], box[1] + 20),
                                                 font, fontScale, color, thickness, cv2.LINE_AA)

                   if cars:
                       ListCars.append(1)
                   else:
                       ListCars.append(0)

                   if len(ListCars) > 8:
                       del ListCars[0]


                   if len(ListCars)>=8:
                       if 1 in ListCars:
                           IsMaveAutosCam3 = True
                       else:
                           IsMaveAutosCam3 = False
                   else:
                       IsMaveAutosCam3 = True
               else:

                   ListCars.clear()
                   IsMaveAutosCam3 = True

               im_ = cv2.line(im_, (x1, y1), (x2, y2), (0, 255, 0), 2)
               im_ = cv2.line(im_, (x1, y1), (x2, y2), (0, 255, 0), 2)
               im_ = cv2.line(im_, (x21, y21), (x22, y22), (0, 255, 0), 2)
               im_ = cv2.line(im_, (x31, y31), (x32, y32), (255, 0, 0), 1)
               im_ = cv2.line(im_, (x41, y41), (x42, y42), (0, 255, 0), 2)

               #cv2.imshow('CamEntranceAdvancedRead3', imutils.resize(im_, width=1024))
               #cv2.waitKey(1)
               if EmergencyMode==True:
                   b, g, r, a = 0, 0, 255, 0
                   fontpath = "a_Albionic.ttf"
                   font = ImageFont.truetype(fontpath, 32)
                   img_pil = Image.fromarray(im_)
                   draw = ImageDraw.Draw(img_pil)
                   draw.text((50, 80), "Система перешла в аварийный режим.", font=font, fill=(b, g, r, a))
                   draw.text((100, 120), " Шлакбаум заблокирован!", font=font, fill=(b, g, r, a))
                   im_ = np.array(img_pil)
               cv2.startWindowThread()
               cv2.namedWindow("CamEntranceAdvancedRead3")
               cv2.imshow("CamEntranceAdvancedRead3", imutils.resize(im_, width=800))

               if CamReadingPermission == False:
                   sleep(0.1)
               #sleep(0.1)
           except:
               cv2.startWindowThread()
               cv2.namedWindow("CamEntranceAdvancedRead3")
               cv2.imshow("CamEntranceAdvancedRead3", imutils.resize(np.zeros((720,1280,3), np.uint8), width=800))

               ListCars.clear()
               IsMaveAutosCam3 = True
               # NumberPlateText=''
               print('error10')
               sleep(0.3)
           #print(str( (time.time() - start_time)))
          # sleep(0.01)

   def CamEntranceAdvanced4Read():
       global IsMaveAutosCam4
       global testFlagCam4
       ListCars = []
       while True:
           #start_time = time.time()
           try:
               image = imagesArray_auto_4[len(imagesArray_auto_4) - 1][0]
               im_ = imagesArray_auto_4[len(imagesArray_auto_4) - 1][1]
               image = image.cuda()
               # image = image.to(device)
               # predict = procRecAuto(model_auto, d_test, device, args, image)
               x21 = 830
               y21 = 0
               x22 = 700
               y22 = 720

               x31 = 0
               y31 = 240
               x32 = 1280
               y32 = 240


               x41 = 0
               y41 = 272
               x42 = 1280
               y42 = 272

               x1 = 890
               y1 = 0
               x2 = 760
               y2 = 720

               tik = 2
               #BadRetX1 = 650
               #BadRetY1 = 00
               #BadRetX2 = 1280
               #BadRetY2 = 200
               #im_ = cv2.rectangle(im_, (BadRetX1, BadRetY1), (BadRetX2, BadRetY2), (255, 0, 0), tik)

               if CamReadingPermission == True:
                   pred = model_FRCNN([image])

                   if len([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]) > 0:
                       pass

                   cars = False
                   for i in range(
                           len([[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())])):
                       score = list(pred[0]['scores'].detach().cpu().numpy())[i]
                       # if not (int(list(pred[0]['labels'].cpu().numpy())[i])  in [51,9,15,16,34,62,1,35,27,86,72,28,7,31,36,33,42,64,44,17,41,2,11,84,38,16,37,76,18]):
                       if (int(list(pred[0]['labels'].cpu().numpy())[i]) in [ 3, 4, 6, 8]):
                           if score > 0.68:
                               # print('1111111111111111111111111'+str(predict['labels'].cpu()))
                               box = pred[0]['boxes'].cpu()[i]
                               box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                               MainAutoPoint1 = (int(box[0]), int((box[1]) + (int(int(box[3]) - int(box[1])) / 2)))
                               MainAutoPoint2 = (int(box[2]), int((box[1]) + (int(int(box[3]) - int(box[1])) / 2)))

                               if (not (isLineCarIntoRec(x1, y1, x2, y2, x21, y21, x22, y22, MainAutoPoint2,MainAutoPoint1)))or (getRoadUpDown(x41, y41, x42, y42, box[2], box[3])) or ( MainAutoPoint1[1]<y41) :
                                   im_ = cv2.circle(im_, MainAutoPoint1, 3, (255, 0, 0), 3)
                                   im_ = cv2.circle(im_, MainAutoPoint2, 3, (255, 0, 0), 3)
                                   im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), tik)
                                   if not getRoadUpDown(x31, y31, x32, y32, box[2], box[3]):
                                       testFlagCam4 = True
                                       im_ = cv2.circle(im_, MainAutoPoint1, 3, (255, 255, 0), 3)
                                       im_ = cv2.circle(im_, MainAutoPoint2, 3, (255, 255, 0), 3)
                                       im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), tik)
                               else:
                                   # global     WaitMoovingCarAfterOpeniniBarrier
                                   # WaitMoovingCarAfterOpeniniBarrier = False
                                   testFlagCam4 = True
                                   cars = True
                                   im_ = cv2.circle(im_, MainAutoPoint1, 3, (0, 255, 0), 3)
                                   im_ = cv2.circle(im_, MainAutoPoint2, 3, (0, 255, 0), 3)
                                   im_ = cv2.rectangle(im_, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), tik)

                               CenterCarPoint1 = (int(box[0]) + int((int(box[2]) - int(box[0])) / 3),
                                                  int((box[1]) + (int(int(box[3]) - int(box[1])) / 2) * 1))
                               CenterCarPoint = (int(box[0]) + int((int(box[2]) - int(box[0])) / 3 * 2),
                                                 int((box[1]) + (int(int(box[3]) - int(box[1])) / 2)))

                               if (((isLineCarIntoRec(x1, y1, x2, y2, x21, y21, x22, y22, (CenterCarPoint),(CenterCarPoint1)))))and (not((getRoadUpDown(x41, y41, x42, y42,CenterCarPoint1[0],CenterCarPoint1[1])))):
                                   # print('333333333333333333333333333333333333333333333333333')
                                   global WaitMoovingCarAfterOpeniniBarrier
                                   WaitMoovingCarAfterOpeniniBarrier = False
                                   im_ = cv2.line(im_, CenterCarPoint, CenterCarPoint1, (0, 255, 0), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint, 3, (0, 255, 0), 3)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 3, (0, 255, 0), 3)
                                   global trafficlightEntr
                                   trafficlightEntr = 'red'
                                   global trafficlightExt
                                   trafficlightExt = 'red'
                                   # cv2.imwrite('temp/1' + str(time.time()) + '.jpg', im_)
                               else:
                                   im_ = cv2.line(im_, CenterCarPoint, CenterCarPoint1, (255, 0, 0), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint, 3, (255, 0, 0), 3)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 3, (255, 0, 0), 3)

                               """ 
                               if not getRoad3(x1, y1, x2, y2, CenterCarPoint[0], CenterCarPoint[1])or not getRoad3(x1, y1, x2, y2, CenterCarPoint1[0], CenterCarPoint1[1]):
                                   global WaitMoovingCarAfterOpeniniBarrier
                                   WaitMoovingCarAfterOpeniniBarrier = False
                                   im_ = cv2.circle(im_, CenterCarPoint, 2, (0, 0, 255), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 2, (0, 0, 255), 2)
                                   global trafficlightEntr
                                   trafficlightEntr = 'red'
                               else:
                                   im_ = cv2.circle(im_, CenterCarPoint, 2, (255, 0, 255), 2)
                                   im_ = cv2.circle(im_, CenterCarPoint1, 2, (255, 0, 255), 2)
                               """
                               # im_ = cv2.circle(im_, (box[0], box[1]), 5, (0, 0, 255), 2)
                               # im_ = cv2.circle(im_, (box[0], box[3]), 5, (0, 0, 255), 2)

                               # im_ = cv2.circle(im_, (box[2], box[1]), 5, (0, 255, 255), 2)
                               # im_ = cv2.circle(im_, (box[2], box[3]), 5, (0, 255, 255), 2)

                               font = cv2.FONT_HERSHEY_SIMPLEX
                               fontScale = 1
                               color = (0, 0, 255)
                               thickness = 3
                               im_ = cv2.putText(im_, str(score),
                                                 (box[0], box[1] + 20),
                                                 font, fontScale, color, thickness, cv2.LINE_AA)

                   if cars:
                       ListCars.append(1)
                   else:
                       ListCars.append(0)

                   if len(ListCars) > 8:
                       del ListCars[0]

                   if len(ListCars) >= 8:
                       if 1 in ListCars:
                           IsMaveAutosCam4 = True
                       else:
                           IsMaveAutosCam4 = False
                   else:
                       IsMaveAutosCam4 = True
               else:
                   ListCars.clear()
                   IsMaveAutosCam4 = True

               im_ = cv2.line(im_, (x1, y1), (x2, y2), (0, 255, 0), 2)
               im_ = cv2.line(im_, (x21, y21), (x22, y22), (0, 255, 0), 2)
               im_ = cv2.line(im_, (x31, y31), (x32, y32), (255, 0, 0), 1)
               im_ = cv2.line(im_, (x41, y41), (x42, y42), (0, 255, 0), 2)



               # cv2.imshow('CamEntranceAdvancedRead', imutils.resize(im_, width=1024))
               # cv2.waitKey(1)
               if EmergencyMode==True:
                   b, g, r, a = 0, 0, 255, 0
                   fontpath = "a_Albionic.ttf"
                   font = ImageFont.truetype(fontpath, 32)
                   img_pil = Image.fromarray(im_)
                   draw = ImageDraw.Draw(img_pil)
                   draw.text((50, 80), "Система перешла в аварийный режим.", font=font, fill=(b, g, r, a))
                   draw.text((100, 120), " Шлакбаум заблокирован!", font=font, fill=(b, g, r, a))
                   im_ = np.array(img_pil)
               cv2.startWindowThread()
               cv2.namedWindow("CamEntranceAdvancedRead4")
               cv2.imshow("CamEntranceAdvancedRead4", imutils.resize(im_, width=800))

               if CamReadingPermission == False:
                   sleep(0.1)
               # cv2.imshow('a frame', im_)  # a frame is the title name of the display window
               # cv2.waitKey(0)
               # cv2.destroyAllWindows()

           except:
               cv2.startWindowThread()
               cv2.namedWindow("CamEntranceAdvancedRead4")
               cv2.imshow("CamEntranceAdvancedRead4", imutils.resize(np.zeros((720,1280,3), np.uint8), width=800))
               ListCars.clear()
               IsMaveAutosCam4 = True
               # NumberPlateText=''
               print('error80')
               sleep(0.05)
           #print(str( (time.time() - start_time)))
   def CloseBarrier():
       #команда на зарытие
       global isBarrierOpened
       global CamReadingPermission
       #global imagesArray_auto_1
       #global imagesArray_auto_2
       #global imagesArray_auto_3
       if isBarrierOpened == True:
           print('cl1')
           try:
               if (len(imagesArray_auto_4) >= 8)&(len(imagesArray_auto_2) >= 8)&(len(imagesArray_auto_3) >= 8):
                   print('cl2')
                   if DatchikNoCars()==True:
                       global IsMaveAutosCam3
                       global IsMaveAutosCam2
                       #global IsMaveAutosCam1
                       global IsMaveAutosCam4
                       #if IsMaveAutosCam1 == False:
                       if IsMaveAutosCam2 == False:
                               if IsMaveAutosCam3 == False:
                                 if IsMaveAutosCam4 == False:
                                   if CamReadingPermission ==True:
                                       res = requests.get(IPESP32+'off1')
                                       if res.status_code == 200:
                                           start_time2 = time.time()

                                           imagesArray.clear()
                                           imagesArray_exitBarrier.clear()
                                           imagesArray_CamVerifyAutoUnderBarrier.clear()

                                           #global isBarrierOpened
                                           isBarrierOpened = False
                                           #th_chehkEmergencyUPBarrier = Thread(target=CheckEmergencyUPBarrier(start_time2))
                                           #th_chehkEmergencyUPBarrier.start()
                                           print(' Команда н на закрытие шлакбаума отправлена! ')
                                           start_time = time.time()
                                           Alyarm = False
                                           while (time.time() - start_time) < 4.0:
                                               if  (IsMaveAutosCam2 == True) or (IsMaveAutosCam4 == True)or (IsMaveAutosCam3 == True):
                                                   #if  IsMaveAutosCam1: print('аварийный - IsMaveAutosCam1')
                                                   if IsMaveAutosCam2: print('аварийный - IsMaveAutosCam2')
                                                   if IsMaveAutosCam3: print('аварийный - IsMaveAutosCam3')
                                                   if IsMaveAutosCam4: print('аварийный - IsMaveAutosCam4')
                                                   res = requests.get(IPESP32+'on1')
                                                   if res.status_code == 200:
                                                       print(
                                                           'Время от команды на закрытие - до команды на аварийный подъм: ' + str(
                                                               time.time() - start_time2))
                                                       print('Авармийный подём !!!!!!!!!!!!!!!!!!!!! ')
                                                       #global isBarrierOpened
                                                       isBarrierOpened = False
                                                       sleep(5)
                                                       print('Авармийный подём закончен')
                                                       isBarrierOpened = True
                                                       #global Caa
                                                       # ssion
                                                       CamReadingPermission = True
                                                       Alyarm = True
                                                   else:
                                                       print('не смог открыть шлакбаум')

                                               sleep(0.1)
                                           if (Alyarm == False):
                                               global ReadNumberPlatePermission



                                               # sleep(3)
                                               CamReadingPermission = False
                                               isBarrierOpened = False

                                               #if res.status_code == 200:
                                               print('Закрытие закончено')
                                               #тест на работу датчика
                                               print('Выкл авто режим')
                                               sleep(1.1)
                                               res = requests.get(IPESP32 + 'offled5')
                                               ReadNumberPlatePermission = True

                                               th16 = Thread(target=testDatchikOnEmergency())
                                               th16.start()


                                           #sleep(1)
                                           #th = Thread(target=CamEntrance)
                                           #global th
                                           #th.start()
                                           #th1 = Thread(target=CamExitBarrier)
                                           #global th1
                                           #th1.start()
                                           #th7 = Thread(target=CamVerifyAutoUnderBarrier)
                                           #global th7
                                           #th7.start()
                                           #print(' Команда н на закрытие шлакбаума отправлена! ')



                                           #sleep(3)
                                           #print('Закрытие закончен')
                                           #if isBarrierOpened == True:
                                           #CamReadingPermission=False
                                       else:
                                           print('не смог закрыть шлакбаум')
                   else:
                       sleep(5)
                       print('Пауза после срабатывания датчика закончена')


               #else:
               #    sleep(1)


           except:
               print('error11')

   def DatchikNoCars():
       if isBarrierOpened == True:
           try:
               image1=imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier)-1][0]
               im_1 = imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier) - 1][1]

               FN1, NR1 = IsNumberA000AA99(image1,im_1)
               #sleep(0.15)
               #image1 = imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier) - 1][0]
               #im_1 = imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier) - 1][1]
               #FN2, NR2 = IsNumberA000AA99(image1, im_1)

               if FN1 & NR1:
                   print('NO CARS')
                   return True
               else:
                   print('CARS is YES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                   return False

           except:

               #NumberPlateText=''
               print('error12')
               sleep(0.03)
       return False

   def IsNumberA000AA99(image,im_):
       findedNumber=False
       NumberRecognized = False
       try:

           #cv2.imshow('image1212121', imutils.resize(im_, width=1024))
           #cv2.waitKey(1)

           image = image.to(device2)
           predict = procRec(model, device, args, image)
           scores = predict['scores'].cpu()


           if len(predict['masks'].cpu()) > 0:
               tmpL = []
               for x in range(len(scores)):
                   tmpL.append(float(scores[x]))
               tmpL = sorted(tmpL, reverse=True)
               # print('111111111111111111111111111111111111111111111111111111111111111')
               res = tmpL[0]
               try:
                    res1 = tmpL[1]
               except:
                    res1 = 0
               try:
                    res2 = tmpL[2]
               except:
                    res2 = 0
               if (res >= 0.85)&(res1 < 0.85)&(res2 < 0.85):
                   print('вижу только один номер scores  ' + str(res) + '  ' + str(res1)+'  ' + str(res2))
               if (res >= 0.85)&(res1 >= 0.85)&(res2 < 0.85):
                   print('вижу только два номера scores  ' + str(res) + '  ' + str(res1)+'  ' + str(res2))
               #print('max score '+str(scores[ind] ))
               if (res >= 0.85)&(res1 >= 0.85)&(res2 >= 0.85):
                   print('вижу три номера scores  ' + str(res) + '  ' + str(res1)+'  ' + str(res2))
                   findedNumber = True
                   NumberRecognized = True
                   return findedNumber, NumberRecognized

                   # print('000000000000000000000')
                   mask = predict['masks'][ind].cpu()
                   box = predict['boxes'][ind].cpu()
                   splash = color_splash(im_, mask, box,1)
                   if splash is not None:
                       # splash = imutils.resize(splash, width=640)

                       # im1 = cv2.imread('img.jpg')
                       # im_resize = cv2.resize(im1 (500, 500))
                       is_success, im_buf_arr = cv2.imencode(".jpg", splash)
                       byte_im = im_buf_arr.tobytes()
                       r = requests.post(url, data=byte_im)
                       global NumberPlateText
                       print(str(r.text))
                       if r.text[len(r.text)-4:] == 'AA96':
                           NumberRecognized = True

                       """
                       if len(r.text) > 7:
                           if r.text[0].isnumeric() == False:
                               if r.text[4].isnumeric() == False:
                                   if r.text[5].isnumeric() == False:
                                       if int(r.text[1]) >= 0:
                                           if int(r.text[2]) >= 0:
                                               if int(r.text[3]) >= 0:
                                                   if int(r.text[6]) >= 0:
                                                       if int(r.text[7]) >= 0:
                                                           if ((len(r.text) >= 8) & (len(r.text) <= 9)):
                                                               print('Номер машины: ' + r.text)
                                                               if r.text[4:] == 'AA96':
                                                                   #return True
                                                                   NumberRecognized=True
                                                           else:
                                                               # NumberPlateText = ''
                                                               pass
                       """
                   #if (splash is not None):
                   #    cv2.imshow('Finish', splash)
                   #    cv2.waitKey(1)
       except:
         print('error13')
       return findedNumber,NumberRecognized

   def TryCloseBarrier():
       while True:
          try:
              if EmergencyMode == False:
                    global IsMaveAutosCam4
                    global IsMaveAutosCam3
                    global IsMaveAutosCam2
                    #global IsMaveAutosCam1
                    global isBarrierOpened
                    global WaitMoovingCarAfterOpeniniBarrier

                    if isBarrierOpened==True:
                        #print(str(WaitMoovingCarAfterOpeniniBarrier))
                        if WaitMoovingCarAfterOpeniniBarrier == False:
                               #if IsMaveAutosCam1==False:
                                   #print(str(IsMaveAutosCam2)+str(IsMaveAutosCam3)+str(IsMaveAutosCam4))
                                   if IsMaveAutosCam2 == False:
                                       if IsMaveAutosCam3 == False:
                                           if IsMaveAutosCam4 == False:
                                           #if   (len(imagesArray)>=10 & len(imagesArray_exitBarrier)>=10):
                                                global trafficlightEntr
                                                trafficlightEntr = 'red'
                                                global trafficlightExt
                                                trafficlightExt = 'red'
                                                #print("ДЕРЬМО")
                                                CloseBarrier()
                                   #pass



          except:
              print('error14')
          sleep(0.1)
   """ 
   def CheckEmergencyUPBarrier(start_time2):
       print(' Команда н на закрытие шлакбаума отправлена! ')
       start_time = time.time()
       Alyarm=False
       while (time.time() - start_time) < 3.0:
           if (IsMaveAutosCam1 == True)or(IsMaveAutosCam2 == True)or(IsMaveAutosCam3 == True):
               res = requests.get(IPESP32+'on1')
               if res.status_code != 200:
                   print('Время от команды на закрытие - до команды на аварийный подъм: '+str(time.time() - start_time2))
                   print('Авармийный подём !!!!!!!!!!!!!!!!!!!!! ')
                   global isBarrierOpened
                   isBarrierOpened = False
                   sleep(10)
                   print('Авармийный подём закончен')
                   isBarrierOpened = True
                   global CamReadingPermission
                   CamReadingPermission = True
                   Alyarm = True

           sleep(0.1)
       and (ReadNumberPlatePermission == True)if (Alyarm==False):
         #sleep(3)
         CamReadingPermission = False
         isBarrierOpened = False
         print('Закрытие закончен')
   """
   def  Exit_barrir_NumberRec():
       while True:
           global WaitMoovingCarAfterOpeniniBarrier
           global CamReadingPermission
           global ReadNumberPlatePermission
           if (WaitMoovingCarAfterOpeniniBarrier == False) and (isBarrierOpened == False) and (CamReadingPermission == False) and (ReadNumberPlatePermission == True):
               try:
                   if len(imagesArray_exitBarrier)>8:
                       # print(str(len(imagesArray)))
                       image = imagesArray_exitBarrier[len(imagesArray_exitBarrier) - 1][0]
                       im_ = imagesArray_exitBarrier[len(imagesArray_exitBarrier) - 1][1]

                       # tensor_image = image.view(image.shape[1], image.shape[2], image.shape[0])

                       # im=cv2.cvtColor(tensor_image.numpy(), cv2.COLOR_RGB2BGR)

                       # im_ = image

                       image = image.to(device1)

                       predict = procRec(model2, device1, args, image)
                       if len(predict['masks'].cpu()) > 0:
                           # print('111111111111111111111111111111111111111111111111111111111111111')
                           scores = predict['scores'].cpu()
                           tmpL = []

                           for x in range(len(scores)):
                               tmpL.append(float(scores[x]))
                           res = max(tmpL)
                           ind = tmpL.index(res)

                           if float(scores[ind]) >= 0.9997:
                               # print('000000000000000000000')
                               mask = predict['masks'][ind].cpu()
                               box = predict['boxes'][ind].cpu()
                               # print('score ' + str(scores[ind]))
                               splash = color_splash(im_, mask, box, 0)
                               if splash is not None:
                                   # splash = imutils.resize(splash, width=640)

                                   # im1 = cv2.imread('img.jpg')
                                   # im_resize = cv2.resize(im1 (500, 500))
                                   is_success, im_buf_arr = cv2.imencode(".jpg", splash)
                                   byte_im = im_buf_arr.tobytes()

                                   # data = open('img.jpg', 'rb').read()
                                   # data = splash.tobytes('C')
                                   r = requests.post(url, data=byte_im)
                                   global NumberPlateText
                                   if len(r.text) > 7:
                                       if r.text[0].isnumeric() == False:
                                           if r.text[4].isnumeric() == False:
                                               if r.text[5].isnumeric() == False:
                                                   if int(r.text[1]) >= 0:
                                                       if int(r.text[2]) >= 0:
                                                           if int(r.text[3]) >= 0:
                                                               if int(r.text[6]) >= 0:
                                                                   if int(r.text[7]) >= 0:
                                                                       if ((len(r.text) >= 8) & (len(r.text) <= 9)):
                                                                           print('Номер машины: ' + r.text)
                                                                           NumberPlateText = r.text
                                                                           # http: // 10.57.0.63 / on1
                                                                           # res = requests.get('http://10.57.0.63/on1')
                                                                           # if res.status_code==200:
                                                                           #    print(' Команда но открытие шлакбаума отправлена! ')

                                                                           # if NumberPlateText=='H905CP37':

                                                                           global ShlakbaumOpenThread
                                                                           numbRec = False
                                                                           NR = NumberToRussian(NumberPlateText)
                                                                           global trafficlightExt
                                                                           for xx in range(len(ListCarNumbers)):
                                                                               if NR == ListCarNumbers[xx][2]:
                                                                                   print('Выезд: Номер в базе найден, владелец '+str( ListCarNumbers[xx][1]))
                                                                                   if ListCarNumbers[xx][5].lower()!='да':


                                                                                       carPermissionTimeBegin = ListCarNumbers[xx][4][:5]
                                                                                       carPermissionTimeEnd = ListCarNumbers[xx][4][6:]
                                                                                       datetime_object_begin = datetime.datetime.strptime(carPermissionTimeBegin, '%H:%M')
                                                                                       datetime_object_end = datetime.datetime.strptime(carPermissionTimeEnd,'%H:%M')
                                                                                       datetime_object_now = datetime.datetime.strptime(datetime.datetime.now().strftime("%H:%M:%S"), '%H:%M:%S')
                                                                                       if (datetime_object_now >= datetime_object_begin)&(datetime_object_now <= datetime_object_end):
                                                                                           if ShlakbaumOpenThread.is_alive() == False:
                                                                                               #res = requests.get(IPESP32 + 'onled5')
                                                                                               ShlakbaumOpenThread = Thread(
                                                                                                   target=ShlakbaumOpen)
                                                                                               ShlakbaumOpenThread.start()
                                                                                               #global ReadNumberPlatePermission
                                                                                               ReadNumberPlatePermission == False
                                                                                               trafficlightExt = 'green'
                                                                                               #global testFlagCam1
                                                                                               #testFlagCam1 = False
                                                                                               global testFlagCam4
                                                                                               testFlagCam4 = False
                                                                                               th14 = Thread(target=testCam3andCam4)
                                                                                               th14.start()
                                                                                               sleep(5)
                                                                                               numbRec = True
                                                                                               th_isert_event_sql = Thread(target=sql_insert_evcent_thread(0,NR,str( ListCarNumbers[xx][1])))
                                                                                               th_isert_event_sql.start()
                                                                                       else:
                                                                                           print('Нет разрешения на выезд в это время '+str(datetime.datetime.now()))
                                                                                   else:
                                                                                        print('Нет разрешения на автоматический выезд')
                                                                                   #numbRec = False
                                                                                   break
                                                                           if numbRec == False:
                                                                               #global trafficlightExt
                                                                               if trafficlightExt != 'green':
                                                                                   print('Выезд: Номера в базе нет')
                                                                                   print('Выезд: перемигиваю светофор ')
                                                                                   requests.get(IPESP32 + 'offled1')
                                                                                   sleep(0.05)
                                                                                   requests.get(IPESP32 + 'onled1')
                                                                                   #sleep(0.1)
                                                                                   #requests.get(IPESP32 + 'offled1')
                                                                                   #sleep(0.1)
                                                                                   #requests.get(IPESP32 + 'onled1')
                                                                                   #sleep(0.1)


                                                                       else:
                                                                           # NumberPlateText = ''
                                                                           pass
                               # if (splash is not None):
                               # cv2.imshow('Finish', splash)
                   else:            # cv2.waitKey(1)
                        sleep(0.1)
                   # global NumberPlateText
                   #font = cv2.FONT_HERSHEY_SIMPLEX
                   #org = (100, 100)
                   #fontScale = 3
                   #color = (0, 0, 255)
                   #thickness = 8
                   #if (len(NumberPlateText)) >= 8:
                   #    im_ = cv2.putText(im_, NumberPlateText, org, font,
                   #                      fontScale, color, thickness, cv2.LINE_AA)
                   #cv2.imshow('cam2', imutils.resize(im_, width=640))
                   #cv2.waitKey(1)
                   # print(str(len(imagesArray)))
               #except:
               except Exception as e:
                   #print(e)
                   #print(e, file=sys.stderr)
                   #print(str(im_))
                   #print(str(image))

                   # NumberPlateText=''
                   print('error15')
           sleep(0.05)

   #DatchikNoCars
   def testDatchik():
       while True:
          try:
           #start_time = time.time()
           image1 = imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier) - 1][0]
           im_1 = imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier) - 1][1]
           FN1, NR1 = IsNumberA000AA99(image1, im_1)
           #sleep(0.15)
           #image1 = imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier) - 1][0]
           #im_1 = imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier) - 1][1]
           #FN2, NR2 = IsNumberA000AA99(image1, im_1)



           if FN1 & NR1:
               print('NO CARS')
           else:
                   print('CARS is YES!!!!1111111111!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

           #print(str( (time.time() - start_time)))
          except:
                print('errror100')

   def SvetoforChange():
       global trafficlightEntr
       global svetoforEntr
       global trafficlightExt
       global svetoforExt

       while True:
          try:
              if (trafficlightEntr == 'green') & (svetoforEntr != 'green'):
                  res = requests.get(IPESP32 + 'offled2')
                  if res.status_code == 200:
                      svetoforEntr = 'green'
                      print(' Вход - зелёный ')
                  else:
                      print(' Вход - зелёный - Error')
              if (trafficlightEntr == 'red') & (svetoforEntr != 'red'):
                  res = requests.get(IPESP32 + 'onled2')
                  if res.status_code == 200:
                      svetoforEntr = 'red'
                      print(' Вход - красный ')
                  else:
                    print(' Вход - красный  - Error')

              if (trafficlightExt == 'green') & (svetoforExt != 'green'):
                  res = requests.get(IPESP32 + 'offled1')
                  if res.status_code == 200:
                      svetoforExt = 'green'
                      print(' Выход - зелёный ')
                  else:
                      print(' Выход - зелёный - Error')
              if (trafficlightExt == 'red') & (svetoforExt != 'red'):
                  res = requests.get(IPESP32 + 'onled1')
                  if res.status_code == 200:
                      svetoforExt = 'red'
                      print(' Выход - красный ')
                  else:
                    print(' Выход - красный  - Error')

          except:
                print('errror112')
          sleep(0.1)



   def testDatchikOnEmergency():
       start_time = time.time()
       DatchikOK=False
       while (time.time() - start_time) < 5:
          try:
           image1 = imagesArray_CamVerifyAutoUnderBarrier[len(imagesArray_CamVerifyAutoUnderBarrier) - 1][0]
           image = image1.to(device2)

           predict = procRec(model, device, args, image)
           """
           if len(predict['masks'].cpu()) == 2:
               scores = predict['scores'].cpu()
               if float(scores[0]) >= 0.96 & float(scores[1]) >= 0.96:
                    print('датчиков    ' + str(len(predict['masks'].cpu())))
           if len(predict['masks'].cpu()) == 0:
                print('датчиков    0' )
           """



           if len(predict['masks'].cpu()) >= 1:
               scores = predict['scores'].cpu()
               TMPLIST=[]
               for x in range(len(scores)):
                   if float(scores[x])>=0.96:
                    TMPLIST.append(1)
               if len(TMPLIST)==2:
                    #print('датчиков    ' + str(len(predict['masks'].cpu())))
                    DatchikOK = True
                    #print(' Шлакбаум перекрыл 2 ой  номер ' +str(scores[0]))


          except:
                print('errror111')
       if DatchikOK != True:
           #apllicationBlock()
           print('Датчик не прошёл провекрку - входим в аварийный режим ')
       else:
           print('Проверка датчика - выполнена')


   def testCam3andCam4():
       global testFlagCam3
       global testFlagCam4
       TestCamPass3=False
       TestCamPass4 = False
       start_time = time.time()
       while (time.time() - start_time) < 10.0:

              #print('testcam14  '+str(testFlagCam1)+'  '+str(testFlagCam4)+' '+str((time.time() - start_time)))
              if testFlagCam3:
                 TestCamPass3=True
              if testFlagCam4:
                 TestCamPass4=True

              sleep(0.2)
       if (TestCamPass3 == False):
           print('Тест не прошла камера №3 ')
       if (TestCamPass4 == False):
           print('Тест не прошла камера №4 ')

       if  (TestCamPass3==False) or (TestCamPass4==False):
       #if (TestCamPass4 == False):
           #apllicationBlock()
           print('блокируем рпаботу программы')
       else:
           print('Тест пройдллен камерами:  №3 и №4')
       #print('Время теста 1 и 4 '+str( (time.time() - start_time)))

   def testCam2andCam3():
       global testFlagCam2
       #global testFlagCam3
       TestCamPass2 = False
       #TestCamPass3 = False
       start_time = time.time()
       while (time.time() - start_time) < 10.0:

           #print('testcam23  ' + str(testFlagCam2) + '  ' + str(testFlagCam3)+' '+str((time.time() - start_time)))
           if testFlagCam2:
               TestCamPass2 = True
           #if testFlagCam3:
               #TestCamPass3 = True

           sleep(0.2)
       if (TestCamPass2 == False):
           print('Тест не прошла камера №2 ')
       #if (TestCamPass3 == False):
       #    print('Тест не прошла камера №3 ')

       #if (TestCamPass2 == False) or (TestCamPass3 == False):
       ##if (TestCamPass2 == False) :
           #apllicationBlock()
           print('блокируем рпаботу программы')
       else:
           print('Тест пройдллен камерами: №2')
       #print('Время теста 2 и 3 ' + str((time.time() - start_time)))

   def apllicationBlock():

       global EmergencyMode
       EmergencyMode = True
       global IPESP32
       res = requests.get(IPESP32 + 'offled5')
       print('Выключаю авто режим')
       global trafficlightEntr
       trafficlightEntr = 'red'
       global trafficlightExt
       trafficlightExt = 'red'

       IPESP32 = 'http://10.57.22.1/'
       res = requests.get('http://10.57.21.1/' + 'onled2')
       res = requests.get('http://10.57.21.1/' + 'onled1')
       try:
        im2 = imagesArray_auto_2[len(imagesArray_auto_2) - 1][1]
        im3 = imagesArray_auto_3[len(imagesArray_auto_3) - 1][1]
        cv2.imwrite('temp/2.jpg',im2)
        cv2.imwrite('temp/3.jpg', im3)
        SendAlarm('Блокировка работы шлакбаума',project_dir+'/temp/2.jpg',project_dir+'/temp/3.jpg',str(datetime.date.today()))
       except:
           print('SendAlarm почему то не сработал')
   def OFFapllicationBlock():

       global EmergencyMode
       EmergencyMode = False
       global IPESP32
       #res = requests.get(IPESP32 + 'offled5')

       global trafficlightEntr
       trafficlightEntr = 'red'
       global trafficlightExt
       trafficlightExt = 'red'

       IPESP32 = 'http://10.57.21.1/'
       res = requests.get('http://10.57.21.1/' + 'onled2')
       res = requests.get('http://10.57.21.1/' + 'onled1')
       try:
        im2 = imagesArray_auto_2[len(imagesArray_auto_2) - 1][1]
        im3 = imagesArray_auto_3[len(imagesArray_auto_3) - 1][1]
        cv2.imwrite('temp/1.jpg',im2)
        cv2.imwrite('temp/3.jpg', im3)
        SendAlarm('Блокировка работы шлакбаума отключена',project_dir+'/temp/2.jpg',project_dir+'/temp/3.jpg',str(datetime.date.today()))
       except:
           print('SendAlarm почему то не сработал')

   def ButtonThread():
       def change():
           # global EmergencyMode
           # EmergencyMode=True
           apllicationBlock()
       def change2():
           # global EmergencyMode
           # EmergencyMode=True
           OFFapllicationBlock()
       def ManuallyCloseProc():
           try:
               res = requests.get(IPESP32 + 'off1')
               if res.status_code == 200:
                   imagesArray.clear()
                   imagesArray_exitBarrier.clear()
                   imagesArray_CamVerifyAutoUnderBarrier.clear()
                   global isBarrierOpened
                   isBarrierOpened = False
                   print(' Команда на ручное закрытие шлакбаума отправлена! ')
                   global CamReadingPermission
                   CamReadingPermission = False
                   print('Закрытие закончено')
                   requests.get(IPESP32 + 'offled5')

               else:
                   print('не смог вручную  закрыть шлакбаум')
           except:
               print('error ManuallyCloseProc')

       def ManuallyOpenProc():
           try:
               res = requests.get(IPESP32 + 'on1')
               if res.status_code == 200:
                   print(' Команда на ручное открытие шлакбаума отправлена! ')
                   global CamReadingPermission
                   CamReadingPermission = False
                   sleep(0.1)
                   requests.get(IPESP32 + 'offled5')
                   global isBarrierOpened
                   isBarrierOpened = False
                   print(' Шлакбаум вручную отпрыт! ')
               else:
                   print('не смог вручную открыть шлакбаум')
           except:
               print('error ManuallyOpenProc')


       root = Tk()
       b1 = Button(text=" Включить Авариный останов ",
                   width=25, height=3)
       b1.config(command=change)
       b1.pack()
       b2 = Button(text="Отключить Авариный останов ",
                   width=25, height=3)
       b2.config(command=change2)
       b2.pack()
       bManuallyOpen = Button(text=" Открыть вручную ",
                   width=25, height=3)
       bManuallyOpen.config(command=ManuallyOpenProc)
       bManuallyOpen.pack()
       bManuallyClose = Button(text=" Закрыть вручную ",
                   width=25, height=3)
       bManuallyClose.config(command=ManuallyCloseProc)
       bManuallyClose.pack()
       root.mainloop()

   thButton = Thread(target=ButtonThread)
   thButton.start()

   th = Thread(target=CamEntrance)
   th.start()
   th1 = Thread(target=CamExitBarrier)
   th1.start()
   #th2 = Thread(target=CamEntranceAdvanced1)
   #th2.start()

   #th3 = Thread(target=CamEntranceAdvanced1Read)
   #th3.start()
   th4 = Thread(target=CamEntranceAdvanced2)
   th4.start()
   th5 = Thread(target=CamEntranceAdvanced2Read)
   th5.start()

   th6 = Thread(target=TryCloseBarrier)
   th6.start()

   th7 = Thread(target=CamVerifyAutoUnderBarrier)
   th7.start()

   th8 = Thread(target=CamEntranceAdvanced3)
   th8.start()
   th9 = Thread(target=CamEntranceAdvanced3Read)
   th9.start()
   th11 = Thread(target=CamEntranceAdvanced4Read)
   th11.start()
   th10 = Thread(target=Exit_barrir_NumberRec)
   th10.start()
   th12 = Thread(target=CamEntranceAdvanced4)
   th12.start()

   th13 = Thread(target=testDatchik)
   #th13.start()

   #th14 = Thread(target=testCam1andCam4)
   #th15 = Thread(target=testCam2andCam3)

   th16 = Thread(target=SvetoforChange)
   th16.start()
   def LatToRus(a):
       if a == 'A': return 'а'
       if a == 'B': return 'в'
       if a == 'E': return 'е'
       if a == 'K': return 'к'
       if a == 'M': return 'м'
       if a == 'H': return 'н'
       if a == 'O': return 'о'
       if a == 'P': return 'р'
       if a == 'C': return 'с'
       if a == 'T': return 'т'
       if a == 'Y': return 'у'
       if a == 'X': return 'х'

   def NumberToRussian(NumberPlateText):
       str0=NumberPlateText[0]
       str1 = NumberPlateText[1]
       str2 = NumberPlateText[2]
       str3 = NumberPlateText[3]
       str4 = NumberPlateText[4]
       str5 = NumberPlateText[5]
       str6 = NumberPlateText[6]
       str7 = NumberPlateText[7]
       if len(NumberPlateText)==8:
            return LatToRus(str0)+str1+str2+str3+LatToRus(str4)+LatToRus(str5)+str6+str7
       if len(NumberPlateText)==9:
            str8 = NumberPlateText[8]
            return LatToRus(str0)+str1+str2+str3+LatToRus(str4)+LatToRus(str5)+str6+str7+str8




   while True:
       global WaitMoovingCarAfterOpeniniBarrier
       global isBarrierOpened
       global CamReadingPermission
       global ReadNumberPlatePermission

       if (WaitMoovingCarAfterOpeniniBarrier == False) and (isBarrierOpened==False) and (CamReadingPermission == False)and (ReadNumberPlatePermission == True):
           try:
               if len(imagesArray) > 8:

                   #print(str(len(imagesArray)))
                   image=imagesArray[len(imagesArray)-1][0]
                   im_ = imagesArray[len(imagesArray) - 1][1]

                   #tensor_image = image.view(image.shape[1], image.shape[2], image.shape[0])

                   #im=cv2.cvtColor(tensor_image.numpy(), cv2.COLOR_RGB2BGR)


                   #im_ = image

                   image = image.to(device2)

                   predict = procRec(model,  device, args, image)
                   if len(predict['masks'].cpu())>0:
                       #print('111111111111111111111111111111111111111111111111111111111111111')
                       scores = predict['scores'].cpu()
                       tmpL=[]


                       for x in range(len(scores)):
                           tmpL.append(float(scores[x]))
                       res=max(tmpL)
                       ind=tmpL.index(res)


                       if float(scores[ind]) >= 0.9997:
                           #print('000000000000000000000')
                           mask = predict['masks'][ind].cpu()
                           box=predict['boxes'][ind].cpu()
                           #print('score ' + str(scores[ind]))
                           splash = color_splash(im_, mask,box,0)
                           if splash is not None:
                               #splash = imutils.resize(splash, width=640)

                               #im1 = cv2.imread('img.jpg')
                               #im_resize = cv2.resize(im1 (500, 500))
                               is_success, im_buf_arr = cv2.imencode(".jpg", splash)
                               byte_im = im_buf_arr.tobytes()


                               #data = open('img.jpg', 'rb').read()
                               #data = splash.tobytes('C')
                               r = requests.post(url, data=byte_im)
                               global NumberPlateText
                               if len(r.text)>7:
                                   if r.text[0].isnumeric()==False:
                                       if r.text[4].isnumeric() == False:
                                           if r.text[5].isnumeric() == False:
                                               if int(r.text[1])>=0:
                                                   if int(r.text[2])>=0:
                                                       if int(r.text[3])>=0:
                                                           if int(r.text[6])>=0:
                                                               if int(r.text[7])>=0:
                                                                    if ((len(r.text)>=8)&(len(r.text)<=9)):
                                                                        print('Номер машины: '+ r.text)
                                                                        NumberPlateText = r.text

                                                                        #http: // 10.57.0.63 / on1
                                                                        #res = requests.get('http://10.57.0.63/on1')
                                                                        #if res.status_code==200:
                                                                        #    print(' Команда но открытие шлакбаума отправлена! ')

                                                                        #if NumberPlateText=='H905CP37':
                                                                        numbRec=False
                                                                        NR=NumberToRussian(NumberPlateText)
                                                                        global trafficlightEntr
                                                                        for xx in range(len(ListCarNumbers)):
                                                                            if NR==ListCarNumbers[xx][2]:
                                                                                print('Въезд: Номер в базе найден, владелец '+str( ListCarNumbers[xx][1]))
                                                                                carPermissionTimeBegin = ListCarNumbers[xx][4][:5]
                                                                                carPermissionTimeEnd = ListCarNumbers[xx][4][6:]
                                                                                datetime_object_begin = datetime.datetime.strptime(carPermissionTimeBegin, '%H:%M')
                                                                                datetime_object_end = datetime.datetime.strptime(carPermissionTimeEnd, '%H:%M')
                                                                                datetime_object_now = datetime.datetime.strptime(datetime.datetime.now().strftime("%H:%M:%S"), '%H:%M:%S')
                                                                                if (datetime_object_now >= datetime_object_begin) & (datetime_object_now <= datetime_object_end):
                                                                                    global ShlakbaumOpenThread
                                                                                    if ShlakbaumOpenThread.is_alive()==False:
                                                                                        #global trafficlightEntr

                                                                                        trafficlightEntr = 'green'
                                                                                        #global ReadNumberPlatePermission
                                                                                        ReadNumberPlatePermission == False
                                                                                        #es = requests.get(IPESP32 + 'onled5')
                                                                                        ShlakbaumOpenThread = Thread(target=ShlakbaumOpen)
                                                                                        ShlakbaumOpenThread.start()
                                                                                        global testFlagCam2
                                                                                        testFlagCam2 = False
                                                                                        #global testFlagCam3
                                                                                        #testFlagCam3 = False
                                                                                        th15 = Thread(target=testCam2andCam3)
                                                                                        th15.start()
                                                                                        sleep(5)
                                                                                        numbRec = True
                                                                                        th_isert_event_sql = Thread(target=sql_insert_evcent_thread(1, NR,str(ListCarNumbers[xx][1])))
                                                                                        th_isert_event_sql.start()
                                                                                else:
                                                                                    print("Нет разрешения на въезд в это время "+str(datetime.datetime.now()))
                                                                                break
                                                                        if numbRec==False:

                                                                            if trafficlightEntr != 'green':
                                                                                print('Въезд: Номера в базе нет')
                                                                                print('Въезд: перемигиваю светофор ')
                                                                                requests.get(IPESP32 + 'offled2')
                                                                                sleep(0.05)
                                                                                requests.get(IPESP32 + 'onled2')
                                                                                #sleep(0.1)
                                                                                #requests.get(IPESP32 + 'offled2')
                                                                                #sleep(0.1)
                                                                                #requests.get(IPESP32 + 'onled2')
                                                                                #sleep(0.1)


                                                                    else:
                                                                        #NumberPlateText = ''
                                                                        pass
                           #if (splash is not None):
                           # cv2.imshow('Finish', splash)
                           # cv2.waitKey(1)

               else:
                   sleep(0.1)
               #global NumberPlateText
               #font = cv2.FONT_HERSHEY_SIMPLEX
               #org = (100, 100)
               #fontScale = 3
               #color = (0, 0, 255)
               #thickness = 8
               #if (len(NumberPlateText)) >= 8:
               #    im_ = cv2.putText(im_, NumberPlateText, org, font,
               #                         fontScale, color, thickness, cv2.LINE_AA)
               #cv2.imshow('cam1', imutils.resize(im_, width=640))
               #cv2.waitKey(1)
               #print(str(len(imagesArray)))
           except:

               #NumberPlateText=''
               print('error16')
       sleep(0.05)

def color_splash_(image, mask):
    #image=io.imread('1.jpg')
    """Apply color splash effect.
    image: RGB image [height, width, 3]\
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    #gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    gray = np.full((image.shape[0], image.shape[1], 3), 0)
    # We're treating all instances as one, so collapse the mask into one layer
    m=mask.numpy()
    mmm=np.empty((image.shape[0], image.shape[1], 3))
    mmm[:, :, 0] = m
    mmm[:, :, 1] = m
    mmm[:, :, 2] = m
    mask=mmm>0

    #mask=mask > 0.5
    #mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[1] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def color_splash(image, mask,box,Mode):
    bminX = int(box[0])
    bmaxX = int(box[2])
    bminY = int(box[1])
    bmaxY = int(box[3])




    #image = cv2.circle(image, (bminX,bminY), radius=4, color=(255, 255, 255), thickness=2)
    #image = cv2.circle(image, (bmaxX,bmaxY), radius=4, color=(255, 255, 255), thickness=2)
    #cv2.imshow('Finish1111111', image)
    #cv2.waitKey(1)




    image = np.roll(image, (-15, -15, -15))
    if Mode==0:
        image=np.roll(image,(-2,-2,-2),axis =0)
    else:
        image = np.roll(image, (-0, -0, -0), axis=0)
    gray = np.full((image.shape[0], image.shape[1], 3), 0)
    gray2 = np.full((image.shape[0], image.shape[1], 3), 255)

    m = mask.numpy()
    mmm = np.empty((image.shape[0], image.shape[1], 3))
    mmm[:, :, 0] = m
    mmm[:, :, 1] = m
    mmm[:, :, 2] = m
    mask = mmm > 0.5
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, gray2, gray).astype(np.uint8)
    else:

        splash = gray
    if mask.shape[0] > 0:
        splash_ = np.where(mask, image, gray).astype(np.uint8)
    else:

        splash_ = gray



    grayNew =cv2.cvtColor(np.float32(splash), cv2.COLOR_BGR2GRAY)

    points = []
    for x in range(bminY,bmaxY,2):
        for y in range(bminX,bmaxX,2):
            if grayNew[x, y] > 0:
                points.append([x, y])

    m = np.array(points)

    targetPoints = minimum_bounding_rectangle(m)

    #splash_ = cv2.line(splash_, (int(targetPoints[0][1]),int(targetPoints[0][0])), (int(targetPoints[1][1]),int(targetPoints[1][0])), (0,0,0), 1)
    #splash_ = cv2.line(splash_, (int(targetPoints[1][1]), int(targetPoints[1][0])),(int(targetPoints[2][1]), int(targetPoints[2][0])), (0, 0, 0), 1)
    #splash_ = cv2.line(splash_, (int(targetPoints[2][1]), int(targetPoints[2][0])),(int(targetPoints[3][1]), int(targetPoints[3][0])), (0, 0, 0), 1)
    #splash_ = cv2.line(splash_, (int(targetPoints[3][1]), int(targetPoints[3][0])),(int(targetPoints[0][1]), int(targetPoints[0][0])), (0, 0, 0),1 )
    minX = int(findMinX(targetPoints))
    maxX = int(findMaxX(targetPoints))

    minY = int(findMinY(targetPoints))
    maxY = int(findMaxY(targetPoints))
    #print('--------------------- '+str(maxX-minX))
    image = splash_[minY:maxY, minX:maxX]

    #print('111   '+str(minY))
    if (maxX-minX)<190:
        #print('слишком мелко')
        return
    if (minY)>700:
         #print('очень близко')
        return
    #cv2.imshow('Finish11', splash)
    #cv2.waitKey(1)
    #cv2.imshow('Finish12221', image)
    #cv2.waitKey(1)



    p1, p2 = detectTwoBottomPoints(targetPoints)
    stat1 = fline(p1, p2)
    from scipy import ndimage
    # rotation angle in degree
    # image = ndimage.rotate(image, int(90-stat1[2]))
    if stat1[2] > 0:
        image = rotate_image(image, int(90 - stat1[2]))
    else:
        image = rotate_image(image, -int(90 - abs(stat1[2])))

    p1, p2 = detectTwoLeftPoints(targetPoints)
    leftSideDistance = findDistances([p1, p2])

    image_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = []
    for y in range(0,image_gr.shape[1],2):
        for x in range(0,image_gr.shape[0],2):
            if image_gr[x, y] > 0:
                points.append([x, y])
    m = np.array(points)

    targetPoints = minimum_bounding_rectangle(m)
    minX = int(findMinX(targetPoints))
    maxX = int(findMaxX(targetPoints))

    minY = int(findMinY(targetPoints))
    maxY = int(findMaxY(targetPoints))

    # image = cv2.line(image, (int(targetPoints[0][1]),int(targetPoints[0][0])), (int(targetPoints[1][1]),int(targetPoints[1][0])), (0,0,0), 1)
    # image = cv2.line(image, (int(targetPoints[1][1]), int(targetPoints[1][0])),(int(targetPoints[2][1]), int(targetPoints[2][0])), (0, 0, 0), 1)
    # image = cv2.line(image, (int(targetPoints[2][1]), int(targetPoints[2][0])),(int(targetPoints[3][1]), int(targetPoints[3][0])), (0, 0, 0), 1)
    # image = cv2.line(image, (int(targetPoints[3][1]), int(targetPoints[3][0])),(int(targetPoints[0][1]), int(targetPoints[0][0])), (0, 0, 0), 1)
    image = image[minY + 5:maxY , minX:maxX]

    ###########
    # ИЗ ТРАПЕЦИИ ПЫТАЕМСЯ ПРЯМОУГ СДЕЛАТЬ
    points_r = []
    angls_r = []
    image_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for y in range(0, image_gr.shape[0] - 25, 10):
        for x in range(image_gr.shape[1] - 1, 0, -1):
            if image_gr[y, x] > 0:
                points_r.append((y, x))
                break
        if y > 0:

            s = fline((y, x), point_pred)
            if (s[2]) != 0:
                if abs(s[2]) < 50:
                    angls_r.append((s[2]))
        point_pred = (y, x)

    points_l = []
    angls_l = []

    for y in range(0, image_gr.shape[0] - 25, 10):
        for x in range(0, image_gr.shape[1] - 1, 1):
            if image_gr[y, x] > 0:
                points_l.append((y, x))
                break
        if y > 0:
            s = fline((y, x), point_pred)
            if (s[2]) != 0:
                if abs(s[2]) < 50:
                    angls_l.append((s[2]))
        point_pred = (y, x)

    # A=fline(p0, p1)
    angls_r_sred = 0
    angls_l_sred = 0
    if len(angls_l) > 0:
        angls_l_sred = int(sum(angls_l) / len(angls_l))
    if len(angls_r) > 0:
        angls_r_sred = int(sum(angls_r) / len(angls_r))
    angls_r_l_sred = (angls_l_sred + angls_r_sred) / 2

    Rad = 1 / 57.29577951308 * angls_r_l_sred
    BC = abs(int(image.shape[0] * math.tan(Rad)))
    if (angls_r_l_sred) > 0:
        p1 = (0, 0)
        p2 = (image.shape[1] - BC, 0)
        p3 = (image.shape[1], image.shape[0])
        p4 = (BC, image.shape[0])

    else:
        p1 = (BC, 0)
        p2 = (image.shape[1], 0)
        p3 = (image.shape[1] - BC, image.shape[0])
        p4 = (0, image.shape[0])

    pts1 = np.float32([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]], [p4[0], p4[1]]])
    pts2 = np.float32([[0, 0], [image.shape[1] - BC, 0], [image.shape[1] - BC, int((image.shape[1] - BC) / 5)],
                       [0, int((image.shape[1] - BC) / 5)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(image, M, (image.shape[1] - BC, int((image.shape[1] - BC) / 5)))
    #plt.subplot(121), plt.imshow(image), plt.title('Input')
    #plt.subplot(122), plt.imshow(dst), plt.title('Output')
    #plt.show()


    image = dst
    image_gr = cv2.circle(image_gr, p1, radius=4, color=(255, 255, 255), thickness=2)
    image_gr = cv2.circle(image_gr, p2, radius=4, color=(255, 255, 255), thickness=2)
    image_gr = cv2.circle(image_gr, p3, radius=4, color=(255, 255, 255), thickness=2)
    image_gr = cv2.circle(image_gr, p4, radius=4, color=(255, 255, 255), thickness=2)

    #cv2.imshow('Points',imutils.resize(image_gr,width=640))
    #cv2.waitKey(1)

    return image
def distance(p0, p1):
    """
    distance between two points p0 and p1
    """
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)
def findDistances(points):
    """
    TODO: describe function
    """
    distanses = []
    cnt = len(points)

    for i in range(cnt):
        p0 = i
        if (i < cnt - 1):
            p1 = i + 1
        else:
            p1 = 0
        distanses.append({"d": distance(points[p0], points[p1])})
    return distanses

def detectTwoLeftPoints(targetPoints):
    lst = [targetPoints[0][1], targetPoints[1][1], targetPoints[2][1], targetPoints[3][1]]
    m = min(lst)
    x1 = lst.index(m)
    lst[x1] = 100000
    m = min(lst)
    x2 = lst.index(m)
    return targetPoints[x1], targetPoints[x2]
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
def detectTwoBottomPoints(targetPoints):
    lst = [targetPoints[0][1], targetPoints[1][1], targetPoints[2][1], targetPoints[3][1]]
    minX = min(lst)
    maxX = max(lst)
    half = minX + (maxX - minX) / 2
    p_left = []
    p_right = []
    for i in range(4):
        if targetPoints[i][1] < half:
            p_left.append(targetPoints[i])
        if targetPoints[i][1] > half:
            p_right.append(targetPoints[i])

    if p_left[0][0] < p_left[1][0]:
        p1 = p_left[0]
    else:
        p1 = p_left[1]
    if p_right[0][0] < p_right[1][0]:
        p2 = p_right[0]
    else:
        p2 = p_right[1]

    if p1[1] < p2[1]:
        return p1, p2
    else:
        return p2, p1

def findMinX(targetPoints):
    lst = [targetPoints[0][1], targetPoints[1][1], targetPoints[2][1], targetPoints[3][1]]
    return min(lst)


def findMaxX(targetPoints):
    lst = [targetPoints[0][1], targetPoints[1][1], targetPoints[2][1], targetPoints[3][1]]
    return max(lst)


def findMinY(targetPoints):
    lst = [targetPoints[0][0], targetPoints[1][0], targetPoints[2][0], targetPoints[3][0]]
    return min(lst)


def findMaxY(targetPoints):
    lst = [targetPoints[0][0], targetPoints[1][0], targetPoints[2][0], targetPoints[3][0]]
    return max(lst)


def fline(p0, p1, debug=False):
    """
    Вычесление угла наклона прямой по 2 точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    if debug:
        print("Уравнение прямой, проходящей через эти точки:")
    if (x1 - x2 == 0):
        k = 1000000000
        b = y2
    else:
        k = (y1 - y2) / (x1 - x2)
        b = y2 - k * x2
    if debug:
        print(" y = %.4f*x + %.4f" % (k, b))
    r = math.atan(k)
    a = math.degrees(r)
    a180 = a
    if (a < 0):
        a180 = 180 + a
    return [k, b, a, a180, r]

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    detail: https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    #     rotations = np.vstack([
    #         np.cos(angles),
    #         -np.sin(angles),
    #         np.sin(angles),
    #         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def SendAlarm(name, f1, f2, DataFolder):
    print('Send Alarm')
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.header import Header
    from email.mime.image import MIMEImage
    from email.utils import make_msgid
    import mimetypes
    smtpObj = smtplib.SMTP('ivn-srv-04.kangaroo.net', 25)
    # msg = MIMEMultipart()
    # bodyStr = 'ПриВЕТ!'
    # msg['Subject'] = Header('Привет', "utf-8")
    # _attach = MIMEText(bodyStr.encode('utf-8'), 'html', 'UTF-8')
    # msg.attach(_attach)
    from email.message import EmailMessage
    msg = EmailMessage()
    msg['Subject'] = 'Шлакбаум КПП ОБ - переход в аварийный режим'
    image_cid = make_msgid(domain='xyz.com')
    image_cid1 = make_msgid(domain='xyz.com')
    msg.add_alternative("""\
    <html>
                <h3> ФИО из БД: """ + name + """</h3>
                <h4> Дата регистрации : """ + DataFolder + """""""""</h4>
                 <body>

                  <TABLE>
                    <TR>
                    <TD>
                        <h3>ФОТО Камера №1 </h3>                        
                        <img src="cid:{image_cid}"  alt="" />
                    </TD>
                    <TD>
                        <h3> ФОТО Камера №3 </h3>
                        <img src="cid:{image_cid1}"  alt="" />
                        </TD>               
                    </TR>
                  </TABLE>             
                </body>
    </html>
    """.format(image_cid=image_cid[1:-1], image_cid1=image_cid1[1:-1]), subtype='html')
    im = Image.open(f1)
    height = im.height
    width = im.width
    new_height = int(400)
    new_width = int(new_height * width / height)
    im_resize = im.resize((new_width, new_height))
    buf = io.BytesIO()
    im_resize.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    #######
    im2 = Image.open(f2)
    new_width2 = int(new_height * im2.width / im2.height)
    im_resize2 = im2.resize((new_width2, new_height))
    buf2 = io.BytesIO()
    im_resize2.save(buf2, format='JPEG')
    byte_im2 = buf2.getvalue()
    # now open the image and attach it to the email
    with open(f1, 'rb') as img:
        # know the Content-Type of the image
        maintype, subtype = mimetypes.guess_type(f1)[0].split('/')
        # attach it
        msg.get_payload()[0].add_related(byte_im, maintype=maintype, subtype=subtype, cid=image_cid)
    with open(f2, 'rb') as img:
        # know the Content-Type of the image
        maintype, subtype = mimetypes.guess_type(img.name)[0].split('/')
        # attach it
        msg.get_payload()[0].add_related(byte_im2, maintype=maintype, subtype=subtype, cid=image_cid1)
    try:
        smtpObj.sendmail("attention@kangaroo.net", "FaceRecognitionForSecurity@kangaroo.net", msg.as_string())
    except:
        print('Не ушло письмо почему то')
    # smtpObj.sendmail("atten

if __name__ == "__main__":
       import argparse

       project_dir = os.path.dirname(os.path.abspath(__file__))
       parser = argparse.ArgumentParser()
       parser.add_argument("--dataset", default="coco")
       parser.add_argument("--data-dir")
       parser.add_argument("--iters", type=int, default=1)
       args = parser.parse_args([]) # for Jupyter Notebook\n",
       args.use_cuda = True
       args.data_dir = "data/coco2017"
       args.ckpt_path = project_dir+"/ckpt/maskrcnn_NumberPlate.pth"
       args.ckpt_path_auto = project_dir+"/ckpt/maskrcnn_auto.pth"
       args.results = os.path.join(os.path.dirname(args.ckpt_path), "results.pth")
       main(args)