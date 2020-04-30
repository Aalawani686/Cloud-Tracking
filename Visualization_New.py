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
      print_time(self.name, self.counter, 3)
      threadLock.release()

def print_time(threadName, delay, counter):
   while counter:
      print(pyautogui.position())
      time.sleep(delay)

def on_mouse_move(event):
    print('Event received:',event.x,event.y)

# def on_click(event):
#     print("hi")
#     global polygon
#     if polygon.contains_point((event.x, event.y)):
#         polygon.set_facecolor(np.random.random(3))
#         print(event.x, event.y)
#         fig.canvas.draw()

def main():

    # mouseThread1 = myThread(1, "Thread-1", 1)
    # mouseThread1.start()

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
        # global polygon
        # polygon = plt.Polygon([[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]])
        # axs[1, 1].add_patch(polygon)

        # fig.canvas.mpl_connect('button_press_event', on_click)

        axs[0, 0].set_xlabel('X-Coordinate')
        axs[0, 0].set_ylabel('Y-Coordinate')
        # axs[0, 1].set_xlabel('X-Coordinate')
        # axs[0, 1].set_ylabel('Y-Coordinate')
        axs[1, 0].set_xlabel('Z-Coordinate')
        axs[1, 0].set_ylabel('Density of Cloud')

        plt.xticks(file.variables['x'][:])
        plt.yticks(file.variables['y'][:])
        # plt.connect('motion_notify_event',on_mouse_move)

        display(start, imagePath)

def drawContours(contours, backtorgb, proj):
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < (len(proj)-1)*(len(proj[0])-1):
            cv2.drawContours(backtorgb, contours, i, (0,255,0))

    return backtorgb

def findContours(tm):
    image_8bit = np.uint8(projections[tm] * 255)
    image_8bit = np.pad(image_8bit, ((1, 1), (1, 1)), 'constant')
    threshold_level = 127
    _, binarized = cv2.threshold(image_8bit, threshold_level, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    backtorgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)



    if(tm > 0):
        image_8bitL = np.uint8(projections[tm-1] * 255)
        image_8bitL = np.pad(image_8bitL, ((1, 1), (1, 1)), 'constant')
        threshold_level = 127
        _L, binarizedL = cv2.threshold(image_8bitL, threshold_level, 255, cv2.THRESH_BINARY_INV)

        contoursL, hierarchyL = cv2.findContours(binarizedL, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        tracking = trackContours(contoursL, contours, 0.2, -1.2, backtorgb)
        drawn = drawContours(contours, tracking, projections[tm])
    else:
        tracking = trackContours(contours, contours, 0.2, -1.2, backtorgb)
        drawn = drawContours(contours, tracking, projections[tm])

    return drawn

def trackContours(lastContours, contours, wX, wY, image):
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    lastContours.sort(key=lambda c: cv2.contourArea(c), reverse=True)


    if(len(contours) <= 1): return

    contours = contours[1:]
    lastContours = lastContours[1:]

    centers = []
    lastCenters = []
    # for c in range(len(clouds)):
    #     clouds[c].taken = False
    #     if(clouds[c].visible):
    #         image = cv2.circle(image, (((int)(clouds[c].x + wX), (int)(clouds[c].y + wY))), (int)(clouds[c].outerRadius), (255, 0, 0), 1)



    for i in range(len(lastContours)):
        M = cv2.moments(lastContours[i])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

    reset = False

    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    if (len(clouds) <= 0):
        print("reset")
        reset = True
    else:
        clouds.sort(key=lambda c: c.area, reverse=True)


    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if(area > 3):
            (rX,rY),radius = cv2.minEnclosingCircle(contours[i])

            if(reset):
                clouds.append(Cloud(len(ids), area, rX, rY, contours[i], 3*radius/4, 2*radius, -1, area, False, [randint(100, 255), randint(100, 255), randint(100, 255)], True, True, False))
                ids.append(len(ids))
            else:
                inNext = False
                viable = []

                for clo in clouds:
                    if(math.fabs(rX-(clo.x + wX)) <= clo.outerRadius and math.fabs(rY-(clo.y + wY)) <= clo.outerRadius): #TODO: fix size comparison
                        if(not clo.taken):
                            viable.append(clo)
                            inNext = True
                        # else:
                        #     clouds.append(Cloud(len(ids), area, rX, rY, contours[i], 3*radius/4, 2*radius, clo.id, area, False, [randint(100, 255), randint(100, 255), randint(100, 255)], True))
                        #     ids.append(len(ids))
                        #     next = True

                if(len(viable) != 0):
                    # clo = min(viable, key=lambda clo: area*(math.fabs(area - (clo.area))))
                    viable.sort(key=lambda clo: (rX-clo.x-wX)**2 + (rY-clo.y-wY)**2, reverse=False)
                    clo = None
                    for c in viable:
                        if(math.fabs(area - (c.area) < area/2)):
                            clo = c

                    ''' + 50*math.fabs((rX-clo.x-wX)**2 + (rY-clo.y-wY)**2))'''
                    if clo == None:
                        continue
                    #TODO if min error is too high, create new cloud
                    clo.area = area
                    clo.x = rX
                    clo.y = rY
                    clo.outerRadius = 2*radius
                    clo.innerRadius = 3*radius/4
                    clo.dA = max(area - clo.area, math.sqrt(clo.area))
                    clo.parentId = -1
                    clo.taken = True
                if(inNext != True):
                    visible = randint(0, 5) > -1
                    clouds.append(Cloud(len(ids), area, rX, rY, contours[i], 3*radius/4, 2*radius, -1, area, False, [randint(100, 255), randint(100, 255), randint(100, 255)], True, visible, False))
                    ids.append(len(ids))


    for c in reversed(clouds):
        if c.taken == False:
            clouds.remove(c)
            continue

    clouds.sort(key=lambda c: c.area, reverse=False)

    for i in range(len(clouds)-1):
        # if (centers[i][5] == False):
        #     continue
        for j in range(i+1, len(clouds)):
            if ((clouds[i].x - clouds[j].x)**2 + (clouds[i].y - clouds[j].y)**2 <= (clouds[j].outerRadius)**2) :
                # (centers[i][3] + centers[j][3])**2
                # centers[i][6] = np.concatenate(centers[i][6], centers[j][6])
                # (rX,rY),radius = cv2.minEnclosingCircle(centers[i][6])
                # centers[j][1] = (centers[i][0] * centers[i][1] + 2 * centers[j][0] * centers[j][1])/(centers[i][0] + 2 * centers[j][0])
                # centers[j][2] = (centers[i][0] * centers[i][2] + 2 * centers[j][0] * centers[j][2])/(centers[i][0] + 2 * centers[j][0])

                clouds[j].area += clouds[i].area
                clouds[i].parentId = clouds[j].id

    # centers.sort(key=lambda c: c[3], reverse=True)
    #
    # for i in range(len(centers)-1):
    #     if (centers[i][5] == False):
    #         continue
    #     for j in range(i+1, len(centers)):
    #         if ((centers[i][1] - centers[j][1])**2 + (centers[i][2] - centers[j][2])**2 <= (2*centers[i][3] + 2*centers[j][3])**2) :
    #
    #             # centers[i][6] = np.concatenate(centers[i][6], centers[j][6])
    #             # (rX,rY),radius = cv2.minEnclosingCircle(centers[i][6])
    #             centers[i][1] = (centers[i][0] * centers[i][1] + centers[j][0] * centers[j][1])/(centers[i][0] + centers[j][0])
    #             centers[j][2] = (centers[i][0] * centers[i][2] + centers[j][0] * centers[j][2])/(centers[i][0] + centers[j][0])
    #
    #             centers[i][0] += centers[j][0]
    #             # centers[j][3] = radius
    #             centers[j][4] = [0, 0, 255]
    #             centers[j][5] = False
                # centers.remove(centers[j])
    # TODO: FIND way to get rid of smaller overlapping circles, since smaller cloud will be part of larger cloud
    # for k in range(len(centers)-1, -1, -1):
    #     if centers[k][5] == False:
            # centers.remove(centers[k])

    for c in range(len(clouds)):
        i = 1

        clouds[c].taken = False
        if(clouds[c].visible):
            # image[(int)(clouds[c].y + wY), (int)(clouds[c].x + wX)] = (0, 255, 0)
            image = cv2.circle(image, (((int)(clouds[c].x + wX), (int)(clouds[c].y + wY))), (int)(clouds[c].outerRadius), tuple(clouds[c].color), 1)

    return image


def plot(tm, imagePath):
    axs[0, 0].set_ylim(0, len(depths[tm]))
    axs[1, 0].cla()
    axs[1, 0].plot(profiles[tm])
    axs[0, 0].imshow(depths[tm][::-1,:], cmap=plt.cm.Blues, vmin=0.0000001)
    axs[0, 1].set_title("Time Instance: " + str(parseTime(times[tm])))
    axs[1, 1].set_title("Contours of Cloud")
    axs[1, 1].imshow(findContours(tm),cmap=plt.cm.Greys)
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

def display(start, imagePath):

    tm = start
    button_delay = 0.0001

    plot(tm, imagePath)

    plt.pause(0.0001)

    while True:

        char = keyboard()

        if (char == "p"):
            print("Stop!")
            exit(0)

        if (char == "a" and tm > 0):
            # time.sleep(button_delay)
            plot(tm-1, imagePath)
            plt.pause(0.0001)
            tm = tm-1

        elif (char == "d" and tm < len(projections)-1):
            # time.sleep(button_delay)
            plot(tm+1, imagePath)
            plt.pause(0.0001)
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

        instance = cloud[t]
        instance[instance < 0] = 0
        xy = np.sum(instance, axis=0)
        xz = np.sum(instance, axis=2)
        z = np.sum(xz, axis=1)

        projections.append(xy)
        xy[xy > 1] = 1
        depths.append(xy)
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
