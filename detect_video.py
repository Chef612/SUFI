import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
fd=cv2.CascadeClassifier(r'C:\Users\18060\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
fd1=cv2.CascadeClassifier(r'C:\Users\18060\anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml')
fd2=cv2.CascadeClassifier(r'C:\Users\18060\anaconda3\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml')
fd4=cv2.CascadeClassifier(r'C:\Users\18060\anaconda3\Lib\site-packages\cv2\data\haarcascade_profileface.xml')

y = 0
flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/paris.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

count =0
def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    fps = 0.0
    count = 0
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)
        #allowed_classes=['person']


        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        #print(scores[[0],[0]])
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        r,i=vid.read()
        c=scores[0][0]
        #print('Confidence Score'c)
        #print(i)
        gray=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        f=fd.detectMultiScale(gray,1.1,7)
        f1=fd1.detectMultiScale(gray,1.1,7)
        f2=fd2.detectMultiScale(gray,1.1,7)
        #f3=fd3.detectMultiScale(gray,1.1,7)
        f4=fd4.detectMultiScale(gray,1.1,7)
        #print(len(f))
        for x,y,w,h in f:
                cv2.rectangle(i,(x,y),(x+w,y+h),(255,0,0), 2)
        for x,y,w,h in f1:
                cv2.rectangle(i,(x,y),(x+w,y+h),(255,0,0), 2)
        for x,y,w,h in f2:
                cv2.rectangle(i,(x,y),(x+w,y+h),(255,0,0), 2)
        #for x,y,w,h in f3:
        #    cv2.rectangle(i,(x,y),(x+w,y+h),(255,0,0), 2)
        for x,y,w,h in f4:
                cv2.rectangle(i,(x,y),(x+w,y+h),(255,0,0), 2)
        cv2.imshow('face',i)
        cv2.imshow('output', img)
        if(c > 0.80):
                
            if(len(f))==1:
                dirs='cropped_face'
                # Create if there is no cropped face directory
                if not os.path.exists(dirs):
                    os.mkdir(dirs)
                    print("Directory " , dirs ,  " Created ")
                else:    
                    print("Directory " , dirs ,  " has found.")
                for x,y,w,h in f:
                    sub = img[y:y+h, x:x+w]
                    FaceFile = "cropped_face/face_" + str(y+x) + ".jpg" # folder path and random name image
                    cv2.imwrite(FaceFile, sub)
                if FLAGS.output:
                    out.write(img)
            else:
                dirFace='suspected'
                if not os.path.exists(dirFace):
                    os.mkdir(dirFace)
                    print("Directory " , dirFace ,  " Created ")
                else:
                    print("Directory " , dirFace ,  " has found.")
                sub_face = i[100:800, 100:800]
                
                FaceFileName = "suspected/face_" + str(count) + ".jpg" # folder path and random name image
                count = count+1
                cv2.imwrite(FaceFileName, sub_face)
                if FLAGS.output:
                    out.write(img)
       
        
            
    
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass