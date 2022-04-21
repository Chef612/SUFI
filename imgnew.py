import cv2
import detect as dt
#from darknet import Darknet
from PIL import Image

vidcap = cv2.VideoCapture(0)
success, image = vidcap.read()
count = 0

m = Darknet('core/config.py')
m.load_weights('weights/yolov3.weights')
use_cuda = 0
#m.cuda()

while success:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)
    im_pil = im_pil.resize((m.width, m.height))
    boxes = dt.do_detect(m, im_pil, 0.5, 0.4, use_cuda)

    result = open('cropped_face/frame%04d.txt'%(count), 'w')
    for i in range(len(boxes)):
        result.write(boxes[i])
    count = count + 1
    success, image = vidcap.read()
    result.close()
