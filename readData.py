from netCDF4 import Dataset
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

import sys, termios, tty, os, time, argparse

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("-f", "--file", required=True, help="path to input netCDF4 file")
    ap.add_argument("-i", "--images", required=True, help="path to input images")
    ap.add_argument("-t", "--time", required=True, help="start time")
    ap.add_argument("-l", "--loading", required=True, help="Loading data files = 'load', Displaying data = 'display', Both = 'both'")
    args = vars(ap.parse_args())

    imagePath = args["images"]
    filePath = args["file"]
    start = (int)(args["time"])
    loading = args["loading"]

    file = Dataset(filePath, "a", format="NETCDF4", chunks={'time'})
    times = file.variables['time'][:]

    start = getTime(start, times)

    if loading == 'load' or loading == 'both':
        projections, depths, profiles = calculate(file)

    if loading == 'display' or loading == 'both':
        projections = np.load('projections_contour_full.npy')
        profiles = np.load('profiles_contour_full.npy')
        depths = np.load('depths_contour_full.npy')
        plt.ion()
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(8, 9))
        fig.suptitle('Cloud Stereography', fontsize=20)

        axs[0, 0].set_title('Area of Cloud')
        axs[0, 1].set_title('Depth of Cloud')
        axs[1, 0].set_title('Height Profile of Cloud')

        axs[0, 0].set_xlabel('X-Coordinate')
        axs[0, 0].set_ylabel('Y-Coordinate')
        # axs[0, 1].set_xlabel('X-Coordinate')
        # axs[0, 1].set_ylabel('Y-Coordinate')
        axs[1, 0].set_xlabel('Z-Coordinate')
        axs[1, 0].set_ylabel('Density of Cloud')

        plt.xticks(file.variables['x'][:])
        plt.yticks(file.variables['y'][:])

        display(projections, depths, profiles, times, axs, start, imagePath)

def contours(proj):
    image_8bit = np.uint8(proj * 255)
    image_8bit = np.pad(image_8bit, ((1, 1), (1, 1)), 'constant')
    threshold_level = 127
    _, binarized = cv2.threshold(image_8bit, threshold_level, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    backtorgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < (len(proj)-1)*(len(proj[0])-1):
            cv2.drawContours(backtorgb, contours, i, (0,255,0), 1)

    return backtorgb

def plot(axs, tm, depths, profiles, times, imagePath):
    axs[0, 0].set_ylim(0, len(depths[tm]))
    axs[1, 0].cla()
    axs[1, 0].plot(profiles[tm])
    axs[0, 0].imshow(depths[tm][::-1,:], cmap=plt.cm.Blues, vmin=0.0000001)
    axs[0, 1].set_title("Time Instance: " + str(parseTime(times[tm])))
    axs[1, 1].set_title("Contours of Cloud")
    axs[1, 1].imshow(contours(depths[tm]),cmap=plt.cm.Greys)
    if os.path.exists(imagePath + 'sgpstereocamaE45.a1.20190715.' + str(parseTime(times[tm])) + '.jpg'):
        # img = mpimg.imread('Images/20190715/sgpstereocamaE45.a1.20190715.163840.jpg')
        img = mpimg.imread(imagePath + 'sgpstereocamaE45.a1.20190715.' + str(parseTime(times[tm])) + '.jpg')
        axs[0, 1].imshow(img)
    else:
        axs[0, 1].cla()

def keyboard():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def display(projections, depths, profiles, times, axs, start, imagePath):

    tm = start
    button_delay = 0.1

    plot(axs, tm, projections, profiles, times, imagePath)

    plt.pause(0.0001)

    while True:

        char = keyboard()
        # print(char)

        if (char == "p"):
            print("Stop!")
            exit(0)

        if (char == "a" and tm > 0):
            # time.sleep(button_delay)
            plot(axs, tm-1, projections, profiles, times, imagePath)
            plt.pause(0.0001)


            # axs[0, 1].imshow(depths[tm-1], cmap=plt.cm.Blues, vmin=0.0000001)

            # stamp = parseTime(times[tm-1])
            # img = mpimg.imread('Images/20190715/sgpstereocamaE45.a1.20190715.' + stamp + '.jpg')
            tm = tm-1

        elif (char == "d" and tm < len(projections)-1):
            # time.sleep(button_delay)
            plot(axs, tm+1, projections, profiles, times, imagePath)
            plt.pause(0.0001)

            # axs[0, 1].imshow(depths[tm], cmap=plt.cm.Blues, vmin=0.0000001)

            # stamp = parseTime(times[tm+1])
            # img = mpimg.imread('Images/20190715/sgpstereocamaE45.a1.20190715.' + stamp + '.jpg')

            tm = tm+1

def getTime(iter, times):
    val = reverseParseTime(iter)
    if val in times:
        return np.where(times==val)[0][0]
    else:
        return 0

def parseTime(ts):
    hours = str(ts//3600)
    minutes = str((ts%3600)//60)
    seconds = str((ts%3600)%60)
    stamp = hours + minutes.zfill(2) + seconds.zfill(2)
    return stamp

def reverseParseTime(ts):
    return ts%100 + ((ts//100)%100)*60 + (ts//10000)*3600

def calculate(file):

    projections = []
    depths = []
    profiles = []

    t = 0

    x_dim = file.variables['x'][:]
    y_dim = file.variables['y'][:]
    z_dim = file.variables['z'][:]
    cloud = file.variables['cloud'][:]

    while True:

        x_len = len(x_dim)
        y_len = len(y_dim)
        z_len = len(z_dim)

        proj = np.zeros([x_len, y_len])
        depth = np.zeros([x_len, y_len])
        profile = np.zeros([z_len])

        z_min, z_max = z_len+1, 0
        for z in range(z_len):
            for y in range(y_len):
                for x in range(x_len):
                    if cloud[t][z][y][x] == 1:
                        z_max = max(z_max, z)
                        z_min = min(z_min, z)
                        depth[y][x] += 1
                        profile[z] += 1
                        if proj[y][x] == 0:
                            proj[y][x] += 1

        projections.append(proj)
        depths.append(depth)
        profiles.append(profile)
        t += 1
        print(t)
        if t >= len(cloud):
            break
    np.save('projections_contour_full.npy', projections)
    np.save('depths_contour_full.npy', depths)
    np.save('profiles_contour_full.npy', profiles)

    # plt.figure("Tracking")
    # plt.imshow(np.subtract(projections[len(projections)-1], projections[0]), cmap=plt.cm.bwr)

    return projections, depths, profiles

if __name__ == '__main__':
    main()
