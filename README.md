##### MAD
It is an object detecion algorithm based on the YOLOv3 and a fast RCNN which prevents adversarial attacks on normal face detection techniques. This is developed with motive of providing better and more accurate surveillance solutions. Scroll down below to see how you can implement this on your own computer.

#### Create the environment for tensorflow CPU on anaconda(3) prompt
conda env create -f conda-cpu.yml
#### Activate the environment
conda activate yolov3-cpu

#### Save the weights file
python load_weights.py
(a .tf file should be created)

#### Run the main python file for activating the webcam
python detect_video.py --video 0
