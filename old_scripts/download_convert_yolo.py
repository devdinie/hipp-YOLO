import os
import subprocess
import time
import sys
import argparse
import requests
import progressbar

FLAGS = None

root_folder = os.path.abspath("/data/chamal/projects/swapna/YOLO_22v1/YOLO/keras_yolo3")
#download_folder = os.path.join(root_folder, "yolo_model")
#if not os.path.exists(download_folder):
        #os.mkdir(download_folder)

if __name__ == "__main__":
        
    """
    url = "https://pjreddie.com/media/files/yolov3.weights"
    r = requests.get(url, stream=True)

    f = open(os.path.join(root_folder, "yolov3.weights"), "wb")
    file_size = int(r.headers.get("content-length"))
    chunk = 100
    num_bars = file_size // chunk
    bar = progressbar.ProgressBar(maxval=num_bars).start()
    i = 0
    for chunk in r.iter_content(chunk):
        f.write(chunk)
        bar.update(i)
        i += 1
    f.close()
    """
    call_string = "python /data/chamal/projects/swapna/YOLO_22v1/YOLO/keras_yolo3/convert.py /data/chamal/projects/swapna/YOLO_22v1/YOLO/keras_yolo3/yolov3.cfg /data/chamal/projects/swapna/YOLO_22v1/YOLO/keras_yolo3/yolov3.weights  /data/chamal/projects/swapna/YOLO_22v1/YOLO/keras_yolo3/yolo.h5"

    subprocess.call(call_string, shell=True, cwd="/data/chamal/projects/swapna/YOLO_22v1/")