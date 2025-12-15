import numpy as np
import os
from rknn.api import RKNN
import cv2
import signal
import sys
from pathlib import Path
import random
import struct
import json
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from queue import Queue
import threading
import time

from tcpserver import TcpServer

RKNN_MODEL_PATH = '/home/cat/project/models/Rknn_3576_20250703.rknn'
# RKNN_MODEL_PATH = '/home/cat/project/models/Rknnv2_3576_20250706FP8.rknn'
REMOTE_IP = '192.168.32.2'
REMOTE_PORT = 12345
IMAGE_PROCESS_SIZE = 224
IMAGE_TEMP_PATH = '/home/cat/project/run_20250706/temp/temp.jpeg'

server = TcpServer()
# RKNN Initialize
# Create RKNN object
rknn = RKNN(verbose=False)

g_stop = False

def load_demo_image(image_size, picture_name, image_root):
    raw_image = Image.open(os.path.join(image_root, picture_name)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).numpy().astype(np.float32)
    return image

frame_queue = Queue(10)

def capture_frame():
    print('--> Opening camera 0')
    cam0 = cv2.VideoCapture(0)
    if not cam0.isOpened():
        print('Camera 0 open failed.')
        exit()
    print('Done.')
    while g_stop == False:
        ret, frame = cam0.read()
        if not ret:
            print('Camera 0 read failed.')
            continue
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)
        time.sleep(0.2)
        pass
    print('Stop capture.')

def sigint_handler(signal, frame):
    g_stop = True
    server.Stop()
    rknn.release()
    sys.exit(0)

if __name__ == '__main__':
    server.Set(REMOTE_IP, REMOTE_PORT)
    server.Start()

    signal.signal(signal.SIGINT, sigint_handler)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_PROCESS_SIZE, IMAGE_PROCESS_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform='rk3576')
    print('done')

    # Load model
    print('--> Loading model')
    ret1 = rknn.load_rknn(RKNN_MODEL_PATH)
    if ret1  != 0:
        print('Load model failed!')
        exit(ret1)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret1 = rknn.init_runtime(target="rk3576")
    if ret1 != 0:
        print('Init runtime environment failed!')
        exit(ret1)
    print('done')

    threading.Thread(target=capture_frame, daemon=True).start()
    
    while g_stop == False:
        frame = frame_queue.get()

        # Scale image and save
        image_save = cv2.resize(frame, (IMAGE_PROCESS_SIZE, IMAGE_PROCESS_SIZE), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(IMAGE_TEMP_PATH, image_save)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_image = Image.fromarray(np.uint8(frame))
        raw_image = transform(raw_image).unsqueeze(0).numpy().astype(np.float16)

        img_file = open('/home/cat/project/run_20250706/temp/temp.jpeg', mode='rb')
        img_data = img_file.read()
        img_file.close()
        img_size = struct.pack('I', os.stat('/home/cat/project/run_20250706/temp/temp.jpeg').st_size)

        outputs = rknn.inference(inputs=[raw_image], data_format=['nchw'])
        print(outputs[0].shape)

        index = np.array(outputs[0], dtype=np.float16)
        x = np.squeeze(np.array(outputs[1], dtype=np.float16))

        x_width = struct.pack('I', x.shape[0])
        x_height = struct.pack('I', x.shape[1])

        index_width = struct.pack('I', index.shape[0])
        index_height = struct.pack('I', index.shape[1])

        packet = img_size + img_data + x_width + x_height + x.tobytes() + index_width + index_height + index.tobytes()
        print('Send packet to server.')
        server.SendPacket(packet)
        pass

    print('Safe exit.')
    server.Stop()
    rknn.release()
    sys.exit(0)
