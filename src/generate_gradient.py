#!/usr/bin/env python3

import numpy as np
import cv2

def make_gradient():
    # gradient = np.random.permutation(np.arange(0, 256, 1))
    # gradient = np.concatenate((np.arange(0, 256, 1), np.arange(255, -1, -1)))
    gradient = np.concatenate((np.arange(255, -1, -1), np.arange(0, 256, 1)))
    # gradient = np.roll(gradient, np.random.randint(0, gradient.size))
    gradient = np.roll(gradient, round(gradient.size/4))
    # gradient = np.arange(255, 0, 2)
    max_ind = np.where(gradient == 255)[0][0]
    print(f"###########################################\nmax is at = {max_ind}\n###########################################")
    im = np.repeat(gradient, 10)
    im = np.repeat([im], 2, axis=0)
    cv2.imwrite("/home/joseph/catkin_ws/src/sensor_fep_light_tracker/textures/bingbong.jpg", im)
    
    with open("/home/joseph/catkin_ws/src/sensor_fep_light_tracker/src/gradient.npy", "wb") as f:
        np.save(f, gradient)
    
if __name__ == "__main__":
    make_gradient()