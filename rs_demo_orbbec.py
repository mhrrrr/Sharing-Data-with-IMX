import numpy as np
import cv2
import math
from lidar_viz import generate_ray_casting_grid_map
import csv
import time
import os


EXTEND_AREA = 5.0
first_row = True


def divide_img_blocks(img, n_blocks):
    # Dividing image into 86 sections
    vertical_split = np.array_split(img,n_blocks,axis = 1)
    return vertical_split




def upper_lower_bands(data):
    # Calculate the central values of the split
    # Currently dividing into 24 parts and taking an average of each parts and considering the minimum of all the parts
    # Return the min distance by filtering out values out of 1 m vertical distance
    h,w = np.shape(data)
    depth_scale = 0.001
    dist = round(np.average(data[int(h/2-5+pixel_change):int(h/2+5+pixel_change):]*depth_scale),2)
    min = dist
    if dist > 0.5 :
        # pixels = int((480*180*np.arctan(0.5/2))/(vfov*math.pi))
        horizontal_split = np.array_split(data,24.0,axis=0)
        for i in range(len(horizontal_split)):
            x = round(np.average(horizontal_split[i])*depth_scale,2)
            min = x if min>x and (np.tan(abs((12-i)*math.pi/180))*x)<0.45 else min
            # if x < 3.4 :
            #     print(x,(12-i))
            #     print(np.tan(abs((12-i)*math.pi/180))*x)
    # print(min)
    return min 

def get_dist_data(depth_colormap,hfov):
    # Getting distance at each degree
    result = divide_img_blocks(depth_colormap, hfov*2)
    dist_data = np.zeros(hfov*2)
    angle_data = np.zeros(hfov*2)
    for i in range(len(result)):
        # Call upper_lower_bands function for getting the values of the upper lower and central band
        # if the distance of the upper/lower bands is lesser, then take that value instead
        dist_data[i] = upper_lower_bands(result[i]) 
        dist_data[i] = dist_data[i] if dist_data[i] < 15.0 else 15.0
        angle_data[i] = math.sin(((i/2)+1-(hfov/2))*math.pi/180)
        # Converting Planar distance from the camera to radial distance for Lidar
        dist_data[i] = dist_data[i]/math.sin(abs((math.pi/2)-angle_data[i]))
    return angle_data, dist_data

def calculate_orientation(accel_data, gyro_data, dt):
    # Convert accelerometer data to roll and pitch angles
    roll = math.atan2(accel_data.get_y(), accel_data.get_z())
    pitch = math.atan2(-accel_data.get_x(), math.sqrt(accel_data.get_y()**2 + accel_data.get_z()**2))

    # Integrate the gyroscope data to update roll and pitch angles
    roll += gyro_data.get_x() * dt
    pitch += gyro_data.get_y() * dt

    # Calculate yaw angle using gyroscope data
    yaw = gyro_data.get_z() * dt

    return roll, pitch, yaw

def pitch_correction(vertical_angular_scale, accel_frame, gyro_frame):
    prev_time = 0.0
    if gyro_frame:
        # Calculate time difference since the last frame
        curr_time = accel_frame.timestamp
        dt = (curr_time - prev_time) / 1000.0 if prev_time else 0.0
        prev_time = curr_time

        # Calculate roll, pitch, and yaw angles
        roll, pitch, yaw = calculate_orientation(accel_frame, gyro_frame, dt)
        pitch_act = math.degrees(roll) + 90.0
        # print(pitch_act)

        #Capping the Pitch Degree to 20 Degree to prevent the Horizon to go out of the image
        if pitch_act < 20 and pitch_act > -20:
            pixel_change = round(pitch_act*vertical_angular_scale)
        else:
            pixel_change = round(20*math.copysign(1,pitch_act)*vertical_angular_scale)
        
        return pixel_change
    else:
        return 0.0

def plot_data(depth_image_2):
    xy_resolution = 0.02 # x-y grid resolution
    ang, dist = get_dist_data(depth_image_2,hfov)
    ox = np.sin(ang) * dist
    oy = np.cos(ang) * dist
    occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = generate_ray_casting_grid_map(ox, oy, xy_resolution, True)
    

# ------ Main ---------------------

# Configure depth and color streams

file_path = "lidar_data.csv"
raw_files = os.path.join(os.getcwd(), "Sharing-Data-with-IMX/depth_images")

times = []
for i in os.listdir(raw_files):
    hfov = 91 # Horizontal Field of View
    vfov = 66 # Vertical Field of View
    
    depth_image = np.fromfile(raw_files + '/' + i, dtype=np.uint16)

    # Assuming you know the width and height of the image
    w = 640
    h = 400

    # Reshape the data
    depth_image = depth_image.reshape((h, w))

    vertical_angular_scale = h/vfov
    
    pixel_change = 0.0

    # Cropping the Depth image to h/64 thickeness Horizontally
    pixels = int(480*12/vfov)
    top = int(h/2 - pixels + pixel_change) if int(h/2-pixels + pixel_change) >= 0 else 0
    bottom = int(h/2 + pixels + pixel_change) if int(h/2 + pixels + pixel_change) >= 0 else 480
    depth_image_2 = depth_image[top:bottom,:]

    start = time.time()
    plot_data(depth_image_2)    
    
    end = time.time()
    times.append(end - start)
print("The time of execution of above program is :", np.average(times) * 10**3, "ms")


