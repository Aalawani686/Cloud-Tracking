from netCDF4 import Dataset
import numpy as np
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
from random import randint
import sys, termios, tty, os, time, argparse
import threading
import pyautogui

axs = None
projections = None
depths = None
profiles = None
times = None
clouds = []
ids = []
clickedLoc = []
newClick = False

class Cloud:

    def __init__(self, id, area, x, y, contour, innerRadius, outerRadius, parentId, dA, selected, color, taken, visible, Lidar):
        self.id = id
        self.x = x
        self.y = y
        self.area = area
        self.contour = contour
        self.innerRadius = innerRadius
        self.outerRadius = outerRadius
        self.parentId = parentId
        self.dA = dA
        self.selected = selected
        self.color = color
        self.taken = taken
        self.visible = visible
        self.Lidar = Lidar

def setVars(proj, dep, prof, tms, ax):
    global projections, depths, profiles, times, axs
    projections = proj
    depths = dep
    profiles = prof
    times = tms
    axs = ax

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print("Starting " + self.name)
      threadLock = threading.Lock()
      threadLock.acquire()
      runContoursWindow()
      threadLock.release()

def runContoursWindow():

    cv2.setMouseCallback("Contours", mouseClick)


def mouseClick(event,x,y,flags,param):
    global clouds
    if (len(clouds) <= 0):
        return
    clouds.sort(key=lambda c: c.area, reverse=False)

    if event == cv2.EVENT_LBUTTONDOWN:
        chosen = False
        for cloud in clouds:
            cloud.selected = False
            if ((x/3 - cloud.x)**2 + (y/3 - cloud.y)**2) < cloud.outerRadius**2 and chosen != True:
                cloud.selected = True
                chosen = True

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

    if loading == 'load' or loading == 'both':
        projections, depths, profiles = calculate(file)
        setVars(projections, depths, profiles, file.variables['time'][:], None)


    if loading == 'display' or loading == 'both':
        projections = np.load('projections_contour_full.npy')
        profiles = np.load('profiles_contour_full.npy')
        depths = np.load('depths_contour_full.npy')
        plt.ion()
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(8, 9))

        fig.suptitle('Cloud Stereography', fontsize=20)

        setVars(projections, depths, profiles, file.variables['time'][:], axs)
        start = getTime(start, times)


        axs[0, 0].set_title('Area of Cloud')
        axs[0, 1].set_title('Depth of Cloud')
        axs[1, 0].set_title('Height Profile of Cloud')

        axs[0, 0].set_xlabel('X-Coordinate')
        axs[0, 0].set_ylabel('Y-Coordinate')

        axs[1, 0].set_xlabel('Z-Coordinate')
        axs[1, 0].set_ylabel('Density of Cloud')

        plt.xticks(file.variables['x'][:])
        plt.yticks(file.variables['y'][:])

        display(start, imagePath)

def drawContours(contours, backtorgb, proj):
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < (len(proj)-1)*(len(proj[0])-1):
            cv2.drawContours(backtorgb, contours, i, (0,255,0))

    return backtorgb

def findContours(tm):
    input = np.uint8(projections[tm] * 255)
    eroded = cv2.morphologyEx(input, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    dilated = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    padded = np.pad(dilated, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    input = np.pad(input, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

    threshold_level = 127
    _, binarized = cv2.threshold(padded, threshold_level, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(hierarchy[0])-1, -1, -1) :
        if (hierarchy[0][i][3] > 0):
            contours.pop(i)

    backtorgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    if(tm > 0):
        # image_8bitL = np.uint8(projections[tm-1] * 255)
        # image_8bitL = np.pad(image_8bitL, ((1, 1), (1, 1)), 'constant')
        # threshold_level = 127
        # _L, binarizedL = cv2.threshold(image_8bitL, threshold_level, 255, cv2.THRESH_BINARY_INV)
        # contoursL, hierarchyL = cv2.findContours(binarizedL, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        trackContours(contours, contours, hierarchy, 0.2, -1.2, backtorgb)
        drawn = drawContours(contours, backtorgb, projections[tm])
    else:
        trackContours(contours, contours, hierarchy, 0.2, -1.2, backtorgb)
        drawn = drawContours(contours, backtorgb, projections[tm])

    return drawn

def trackContours(lastContours, contours, hierarchy, wX, wY, image):

    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    lastContours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    if(len(contours) <= 1): return

    contours = contours[1:]
    reset = False

    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    if (len(clouds) <= 0):
        reset = True

    if(len(clouds) != 0):
        clouds.sort(key=lambda c: c.area, reverse=True)

    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        area = cv2.contourArea(contours[i])
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if(area > 3):
            (rX,rY),radius = cv2.minEnclosingCircle(contours[i])
            if(reset):
                clouds.append(Cloud(len(ids), area, rX, rY, contours[i], 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, True, False))
                ids.append(len(ids))
            else:
                inNext = False
                viable = []

                for clo in clouds:
                    if(math.fabs(rX-(clo.x + wX)) <= clo.outerRadius and math.fabs(rY-(clo.y + wY)) <= clo.outerRadius
                        and math.fabs(rX-(clo.x + wX)) <= 2*radius and math.fabs(rY-(clo.y + wY)) <= 2*radius): #TODO: fix size comparison
                        if(not clo.taken):
                            viable.append(clo)
                            inNext = True

                while(len(viable) != 0):
                    clo = min(viable, key=lambda clo: 2*(radius**2)*math.fabs((rX-clo.x-wX)**2 + (rY-clo.y-wY)**2))

                    viable.remove(clo)

                    if (area < 40):
                        if (math.fabs((rX-clo.x-wX)**2 > (clo.outerRadius**2)/2 or (rY-clo.y-wY)**2) > (clo.outerRadius**2)/2 or math.fabs(area - (clo.area)) > clo.area):
                            if (len(viable) > 0):
                                continue
                            clouds.append(Cloud(len(ids), area, rX, rY, contours[i], 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, True, False))
                            ids.append(len(ids))
                            break

                    elif (area < 120):
                        if (math.fabs((rX-clo.x-wX)**2 + (rY-clo.y-wY)**2) > clo.outerRadius**2 or math.fabs(area - (clo.area)) > clo.area):
                            if (len(viable) > 0):
                                continue
                            clouds.append(Cloud(len(ids), area, rX, rY, contours[i], 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, True, False))
                            ids.append(len(ids))
                            break

                    else:
                        if ((rX-clo.x-wX)**2 > (clo.outerRadius**2)/2 or ((rY-clo.y-wY)**2) > (clo.outerRadius**2)/2 or math.fabs(area - (clo.area)) > 2*clo.area):
                            if (len(viable) > 0):
                                continue
                            clouds.append(Cloud(len(ids), area, rX, rY, contours[i], 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, True, False))
                            ids.append(len(ids))
                            break

                    clo.area = area
                    clo.x = rX
                    clo.y = rY
                    clo.outerRadius = 2*radius
                    clo.innerRadius = 3*radius/4
                    clo.dA = max(area - clo.area, math.sqrt(clo.area))
                    clo.parentId = -1
                    clo.taken = True
                    break
                if(inNext != True):
                    visible = True
                    clouds.append(Cloud(len(ids), area, rX, rY, contours[i], 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, visible, False))
                    ids.append(len(ids))

    for c in reversed(clouds):
        if c.taken == False:
            clouds.remove(c)
            continue

    clouds.sort(key=lambda c: c.area, reverse=False)

    for i in range(len(clouds)-1):

        for j in range(i+1, len(clouds)):
            if ((clouds[i].x - clouds[j].x)**2 + (clouds[i].y - clouds[j].y)**2 <= (clouds[j].outerRadius)**2) :

                clouds[j].area += clouds[i].area
                clouds[i].parentId = clouds[j].id

    for c in range(len(clouds)):
        clouds[c].taken = False

        if(clouds[c].visible):
            image = cv2.circle(image, (((int)(clouds[c].x + wX), (int)(clouds[c].y + wY))), (int)(clouds[c].outerRadius), tuple(clouds[c].color), 1)

    return image


def plot(tm, imagePath):

    axs[0, 0].set_ylim(0, len(projections[tm]))
    axs[1, 0].cla()
    axs[1, 0].plot(profiles[tm])
    axs[0, 0].imshow(projections[tm][::-1,:], cmap=plt.cm.Blues, vmin=0.0000001)
    axs[0, 1].set_title("Time Instance: " + str(parseTime(times[tm])))
    axs[1, 1].set_title("Depth of Cloud")
    imageCont = findContours(tm)
    # axs[1, 1].imshow(imageCont ,cmap=plt.cm.Greys)
    axs[1, 1].imshow(depths[tm], cmap=plt.cm.Blues, vmin=0.0000001)

    if os.path.exists(imagePath + 'sgpstereocamaE45.a1.20190715.' + str(parseTime(times[tm])) + '.jpg'):
        img = mpimg.imread(imagePath + 'sgpstereocamaE45.a1.20190715.' + str(parseTime(times[tm])) + '.jpg')
        axs[0, 1].imshow(img)
    else:
        axs[0, 1].cla()

    cv2.imshow("Contours", cv2.resize(imageCont, (3*imageCont.shape[0], 3*imageCont.shape[1]), interpolation = cv2.INTER_AREA))


def keyboard():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def display(start, imagePath):

    tm = start
    button_delay = 0.0001

    plot(tm, imagePath)
    char = ''

    plt.pause(0.0001)

    while True:

        lastChar = char
        char = keyboard()

        if (char == "p"):
            exit(0)

        if (char == "m"):
            runContoursWindow()

            # plot(tm, imagePath)
            # plt.pause(0.0001)

        elif (char == "a" and tm > 0):
            plot(tm-1, imagePath)
            plt.pause(0.0001)
            if (lastChar == 'm'):
                plot(tm-1, imagePath)
                plt.pause(0.0001)
            tm = tm-1

        elif (char == "d" and tm < len(projections)-1):
            plot(tm+1, imagePath)
            plt.pause(0.0001)
            if (lastChar == 'm'):
                plot(tm+1, imagePath)
                plt.pause(0.0001)
            tm = tm+1

        elif (char == "f" and tm < len(projections)-10):
            plot(tm+10, imagePath)
            plt.pause(0.0001)
            tm = tm+10

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

        instance = cloud[t]
        instance[instance < 0] = 0
        xy = np.sum(instance, axis=0)
        xy2 = np.sum(instance, axis=0)

        xz = np.sum(instance, axis=2)
        z = np.sum(xz, axis=1)

        depths.append(xy)
        xy2[xy2 > 1] = 1
        projections.append(xy2)

        profiles.append(z)
        t += 1
        if t >= len(cloud):
            break

    np.save('projections_contour_full.npy', projections)
    np.save('depths_contour_full.npy', depths)
    np.save('profiles_contour_full.npy', profiles)

    return projections, depths, profiles

if __name__ == '__main__':
    main()
