from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import numpy as np
import PIL
import cv2
import math
from random import randint
import sys, termios, tty, os, time, argparse
import threading
import pyautogui
from matplotlib.patches import Rectangle
import timeit

axs = None
projections = None
depths = None
profiles = None
times = None
radii = {}
distances = {}
clouds = []
ids = []

class Cloud:

    def __init__(self, id, area, x, y, contour, radius, innerRadius, outerRadius, parentId, dA, selected, color, taken, visible, Lidar):
        self.id = id
        self.x = x
        self.y = y
        self.area = area
        self.contour = contour
        self.radius = radius
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
                global radii
                radii = {}
                global distances
                distances = {}
                axs[1, 0].cla()
                axs[0, 1].cla()

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--date", required=True, help="date")
    ap.add_argument("-i", "--images", required=True, help="path to input images")
    ap.add_argument("-ts", "--sonde", required=True, help="start time for sonde")
    ap.add_argument("-tl", "--lidar", required=True, help="start time for lidar")

    ap.add_argument("-l", "--loading", required=True, help="Loading data files = 'load', Displaying data = 'display', Both = 'both'")
    args = vars(ap.parse_args())

    imagePath = args["images"]
    date = args["date"]
    loading = args["loading"]
    time_ld = args["lidar"]
    time_sonde = args["sonde"]

    file = Dataset("SGP_COGS" + str(date) + ".nc", "a", format="NETCDF4", chunks={'time'})

    if loading == 'load' or loading == 'both':
        projections, depths, profiles = calculate(file)
        setVars(projections, depths, profiles, file.variables['time'][:], None)


    if loading == 'display' or loading == 'both':
        projections = np.load('projections_contour_full.npy')
        profiles = np.load('profiles_contour_full.npy')
        depths = np.load('depths_contour_full.npy')
        plt.ion()
        plt.show()

        start, end, draw = plotCloud(date, time_ld, time_sonde)
        #temp = fullCloudDataset(date, time_ld, time_sonde)

        fig, axs = plt.subplots(2, 2, figsize=(8, 9))
        fig.subplots_adjust(hspace=.2)
        fig.subplots_adjust(wspace=.3)

        fig.suptitle('Cloud Stereography', fontsize=20)

        setVars(projections, depths, profiles, file.variables['time'][:], axs)
        start = getClosestTime(start, times)


        axs[0, 0].set_title('Area of Cloud')
        axs[1, 1].set_title('Depth of Cloud')
        axs[1, 0].set_title("Radius of Tracked Cloud")
        axs[0, 1].set_title("Distance to Lidar / Radius")

        axs[0, 0].set_xlabel('X-Coordinate')
        axs[0, 0].set_ylabel('Y-Coordinate')

        axs[1, 0].set_xlabel('Length (m)')
        axs[1, 0].set_ylabel('TimeStamp')

        axs[0, 1].set_xlabel('Ratio')
        axs[0, 1].set_ylabel('TimeStamp')

        axs[1, 1].set_xlabel('X-Coordinate')
        axs[1, 1].set_ylabel('Y-Coordinate')

        plt.xticks(file.variables['x'][:])
        plt.yticks(file.variables['y'][:])

        display(start, imagePath, draw, fig)

def drawContours(contours, backtorgb, proj):
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < (len(proj)-1)*(len(proj[0])-1):
            cv2.drawContours(backtorgb, contours, i, (0,255,0))

    return backtorgb

def findContours(tm, isFirst):
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


    trackContours(tm, contours, contours, hierarchy, 0.2, -1.2, backtorgb, isFirst)
    drawn = drawContours(contours, backtorgb, projections[tm])

    return drawn

def trackContours(tm, lastContours, contours, hierarchy, wX, wY, image, isFirst):

    start = timeit.default_timer()

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

            if M["m00"] != 0:
                rX = int(M["m10"] / M["m00"])
                rY = int(M["m01"] / M["m00"])
            else:
                rX, rY = 0, 0

            if(reset):
                clouds.append(Cloud(len(ids), area, rX, rY, contours[i], radius, 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, True, False))
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
                    clo = min(viable, key=lambda clo: radius*((rX-clo.x-wX)**2 + (rY-clo.y-wY)**2) + math.fabs(clo.area - area))

                    viable.remove(clo)

                    if (area < 40):
                        if (math.fabs((rX-clo.x-wX)**2 > (clo.radius**2)/2 or (rY-clo.y-wY)**2) > (clo.radius**2)/2):
                            if (len(viable) > 0):
                                continue
                            clouds.append(Cloud(len(ids), area, rX, rY, contours[i], radius, 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, True, False))
                            ids.append(len(ids))
                            break

                    elif (area < 200):
                        if (math.fabs((rX-clo.x-wX)**2 + (rY-clo.y-wY)**2) > (clo.radius)**2):
                            if (len(viable) > 0):
                                continue
                            clouds.append(Cloud(len(ids), area, rX, rY, contours[i], radius, 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, True, False))
                            ids.append(len(ids))
                            break

                    else:
                        if ((rX-clo.x-wX)**2 > (clo.outerRadius**2)/2 or ((rY-clo.y-wY)**2) > (clo.outerRadius**2)/2 or math.fabs(area - (clo.area)) > 2*max(area, clo.area)):
                            if (len(viable) > 0):
                                continue
                            clouds.append(Cloud(len(ids), area, rX, rY, contours[i], radius, 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, True, False))
                            ids.append(len(ids))
                            break

                    clo.area = area
                    clo.contour = contours[i]
                    clo.x = rX
                    clo.y = rY
                    clo.radius = radius
                    clo.outerRadius = 2*radius
                    clo.innerRadius = 3*radius/4
                    clo.dA = max(area - clo.area, math.sqrt(clo.area))
                    clo.parentId = -1
                    clo.taken = True
                    break
                if(inNext != True):
                    visible = True
                    clouds.append(Cloud(len(ids), area, rX, rY, contours[i], radius, 3*radius/4, 2*radius, -1, area, False, [randint(0, 200), randint(0, 200), randint(0, 200)], True, visible, False))
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

    radius = -1
    minDistance = -1

    for c in range(len(clouds)):
        dist = cv2.pointPolygonTest(clouds[c].contour,(image.shape[0]/2, image.shape[1]/2),True)
        if ((isFirst and dist > 0) or clouds[c].selected) :
            M = cv2.moments(clouds[c].contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            radius = clouds[c].radius
            mX, mY = image.shape[0]/2, image.shape[1]/2
            minPoint = min(clouds[c].contour, key = lambda p : (mX-p[0][0])**2 + (mY-p[0][1])**2)
            minDistance = math.sqrt(((mX - minPoint[0][0])**2 + (mY - minPoint[0][1])**2))
            clouds[c].selected = True
            image = cv2.line(image, ((int)(mX), (int)(mY)), ((int)(minPoint[0][0]), (int)(minPoint[0][1])), tuple(clouds[c].color), 1)
            image = cv2.circle(image, ((int)(mX), (int)(mY)), 1, tuple(clouds[c].color), 2)
            global radii
            radii[parseTime(times[tm], ':')] = radius * 50
            global distances
            distances[parseTime(times[tm], ':')] = minDistance/radius

        clouds[c].taken = False

        if(clouds[c].visible):
            image = cv2.circle(image, (((int)(clouds[c].x + wX), (int)(clouds[c].y + wY))), (int)(clouds[c].radius), tuple(clouds[c].color), 1)

    stop = timeit.default_timer()

    return image


def plot(tm, imagePath, isFirst, fig):
    fig.suptitle('Cloud Stereography\n' + str(parseTime(times[tm], ":")), fontsize=20)

    axs[0, 0].set_title('Area of Cloud')
    axs[1, 1].set_title('Depth of Cloud')
    axs[1, 0].set_title("Radius of Tracked Cloud")
    axs[0, 1].set_title("Distance to Lidar / Radius")

    axs[0, 0].set_ylim(0, len(projections[tm]))

    axs[0, 0].imshow(projections[tm][::-1,:], cmap=plt.cm.Blues)
    imageCont = findContours(tm, isFirst)

    axs[1, 0].cla()
    axs[1, 0].scatter(dict(sorted(radii.items())).keys(), dict(sorted(radii.items())).values())
    axs[1, 0].xaxis.set_major_locator(plt.MaxNLocator(3))

    axs[0, 1].cla()
    axs[0, 1].scatter(dict(sorted(distances.items())).keys(), dict(sorted(distances.items())).values())
    axs[0, 1].xaxis.set_major_locator(plt.MaxNLocator(3))

    axs[0, 0].set_xlabel('X-Coordinate')
    axs[0, 0].set_ylabel('Y-Coordinate')

    axs[1, 1].set_xlabel('X-Coordinate')
    axs[1, 1].set_ylabel('Y-Coordinate')

    axs[0, 1].set_xlabel('Ratio')
    axs[0, 1].set_ylabel('TimeStamp')

    axs[1, 0].set_xlabel('Length (m)')
    axs[1, 0].set_ylabel('TimeStamp')

    axs[1, 0].set_title("Radius of Tracked Cloud")
    axs[0, 1].set_title("Distance to Lidar / Radius")

    axs[1, 1].imshow(depths[tm], cmap=plt.cm.Blues)

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

def display(start, imagePath, draw, fig):

    tm = start
    button_delay = 0.0001
    plot(tm, imagePath, True, fig)
    char = ''
    plt.pause(0.0001)

    while True:

        if (times[tm] > times[-1]):
            print("Index out of bounds")
            exit(0)

        draw(secondsToHours(times[tm]))
        lastChar = char
        char = keyboard()

        if (char == "q"):
            exit(0)

        if (char == "m"):
            runContoursWindow()

        elif (char == "a" and tm > 0):
            plot(tm-1, imagePath, False, fig)
            plt.pause(0.0001)
            if (lastChar == 'm'):
                plot(tm-1, imagePath, False, fig)
                plt.pause(0.0001)
            tm = tm-1

        elif (char == "d" and tm < len(projections)-1):
            plot(tm+1, imagePath, False, fig)
            plt.pause(0.0001)
            if (lastChar == 'm'):
                plot(tm+1, imagePath, False, fig)
                plt.pause(0.0001)
            tm = tm+1

        elif (char == "f" and tm < len(projections)-10):
            plot(tm+10, imagePath, False, fig)
            plt.pause(0.0001)
            tm = tm+10

def getTime(iter, times):
    val = reverseParseTime(iter)
    if iter in times:
        return np.where(times==val)[0][0]
    else:
        return 0

def getClosestTime(iter, times):
    for time in times:
        if (math.fabs(iter - time) <= 10):
            return np.where(times==time)[0][0]
    return 0

def parseTime(ts, separator):
    hours = str(ts//3600)
    minutes = str((ts%3600)//60)
    seconds = str((ts%3600)%60)
    stamp = hours + separator + minutes.zfill(2) + separator + seconds.zfill(2)
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

def readDLData(date, time):
    """
    :param date: Provide date in the following format as a string: YearDayMonth. Ex: 20180504 -> 5-4-2018
    :param time: Provide time in the following format as a string: HourMinuteSecond Ex: 113000 -> 11:30
    :return: dictionary with the following structure:
    "attenuated_backscatter" : used to identify cloud height
    "range" : height which the DL return belongs to
    "radial_velocity" : 2D array of velocities
    "cloud" : 2D array with tuple data of the height and velocity of the clouds
    "base_rv" :  radial velocity at cloud base of lowest cloud
    """
    filename = "sgpdlfptC1.b1." + str(date) + "." + str(time) + ".cdf"
    f = Dataset(filename, "r", format="NETCDF4")
    return_dict = {}

    time = f.variables['time'][:]
    range_var = list(enumerate(f.variables['range'][:]))
    range_var = [r for r in range_var if 500 < r[1] < 5000]

    min_range_index = range_var[0][0] - 1
    max_range_index = range_var[-1][0]

    attenuated_backscatter = f.variables['attenuated_backscatter'][:]
    attenuated_backscatter = [
        [0 if y[i] <= 6e-5 or (min_range_index >= i or i >= max_range_index) else y[i] for i in range(len(y))]
        for y in attenuated_backscatter]
    radial_velocity = f.variables['radial_velocity'][:]

    cloud, removed_rows = prune_backscatter(attenuated_backscatter, range_var[:], radial_velocity, min_range_index)
    base_height = find_min_cloud(cloud)

    base_index = find_approx_value_index(f.variables['range'][:], base_height, 0)
    if base_index == -1:
        print("failed to find range")

    adjusted_radial_velocity = []
    adjusted_time = []
    for i in range(len(time)):
        if i not in removed_rows:
            adjusted_time.append(time[i])
            adjusted_radial_velocity.append(radial_velocity[i])

    base_rv = [sublst[:base_index] for sublst in adjusted_radial_velocity]

    # attenuated_backscatter = [[(range_var[i][1], y[i]) for i in y if y[i] != 0] for y in attenuated_backscatter]
    # range_var = [r[1] for r in range_var]

    # return_dict['attenuated_backscatter'] = attenuated_backscatter
    return_dict['time'] = adjusted_time
    return_dict['radial_velocity'] = radial_velocity
    return_dict['range'] = [r[1] for r in range_var]
    return_dict['cloud'] = cloud
    return_dict['base_rv'] = base_rv

    f.close()
    return return_dict


def find_min_cloud(lst):
    lst = [item for sublist in lst for item in sublist]
    return min([item[0] for item in lst])


def prune_backscatter(attenuated_backscatter, range_var, radial_velocity, min_range_index):
    cloud = []
    removed_rows = []
    for i in range(len(attenuated_backscatter)):
        cloud_row = []
        for j in range(len(attenuated_backscatter[i])):
            if attenuated_backscatter[i][j] != 0:
                cloud_row.append((range_var[j - min_range_index][1], radial_velocity[i][j - min_range_index]))
        if cloud_row:
            cloud.append(cloud_row)
        else:
            removed_rows.append(i)

    return cloud, removed_rows


def fullCloudDataset(date, time_ld, time_sonde):
    """
    :param date: Provide date in the following format as a string: YearDayMonth. Ex: 20180504 -> 5-4-2018
    :param time_ld: Provide time in the following format as a string: HourMinuteSecond Ex: 113000 -> 11:30
    :param time_sonde: Provide time in the following format as a string: HourMinuteSecond Ex: 113000 -> 11:30
    :return: A dictionary with the following structure:
    "cloud" : an array of tuples with the structure (time (s), altitude (m), radial_velocity (m/s)
    "base_radial_velocity" : 2D array with tuple data of the height and velocity at height less than lowest cloud
    NOTE: "bottom_cloud" is often empty so include checks for this case
    """
    dl_data = readDLData(date, time_ld)
    return_dict = {'cloud': clusterClouds(date, time_ld, time_sonde), 'base_radial_velocity': dl_data['base_rv']}

    return return_dict


def secondsToHours(time):
    return time / 3600


def hoursToSeconds(time):
    return time * 3600


def clusterClouds(date, time_ld, time_sonde):
    """
    :param date: Provide date in the following format as a string: YearDayMonth. Ex: 20180504 -> 5-4-2018
    :param time_ld: Provide time in the following format as a string: HourMinuteSecond Ex: 113000 -> 11:30
    :param time_sonde: Provide time in the following format as a string: HourMinuteSecond Ex: 113000 -> 11:30
    :return: an array of tuples with the structure (time (s), altitude (m), radial_velocity (m/s)
    """

    data_ld = readDLData(date, time_ld)
    data_sonde = readSondeData(date, time_sonde)

    y = [item[0] for sublist in data_ld["cloud"] for item in sublist]
    rv = [item[1] for sublist in data_ld["cloud"] for item in sublist]

    x = []
    for i in range(len(data_ld["time"])):
        for _ in range(len(data_ld["cloud"][i])):
            x.append(data_ld["time"][i])

    u_wind = []
    v_wind = []
    for i in range(len(y)):
        index = find_approx_value_index(data_sonde["altitude"], y[i], 5)
        if index == -1:
            print("failed to find range")
        u_wind.append(data_sonde["u_wind"][index])
        v_wind.append(data_sonde["v_wind"][index])

    velocity = []
    for i in range(len(u_wind)):
        velocity.append(math.sqrt(u_wind[i] ** 2 + v_wind[i] ** 2))

    # 100m separation using speed
    t_s = [200 / v for v in velocity]

    # 300m mininum length
    t_ll = [300 / v for v in velocity]

    # 5000m maximum length
    t_lh = [5000 / v for v in velocity]

    clusters = map_clusters(y, data_sonde["altitude"], x, t_s, t_ll, t_lh)
    groups = condense_common_values(clusters)
    groups = [g for g in groups if g != 0]
    condensed_x = []
    condensed_y = []
    condensed_rv = []
    condensed_clusters = []
    for i in range(len(clusters)):
        if clusters[i] != 0:
            condensed_x.append(x[i])
            condensed_y.append(y[i])
            condensed_rv.append(rv[i])
            condensed_clusters.append(clusters[i])

    temp = zip(condensed_clusters, condensed_x, condensed_y, condensed_rv)
    # print(temp)
    # print(condensed_rv)
    print(groups)
    cloud_data = []
    for i in range(len(groups)):
        cloud_data.append([])
        for t in temp:
            if t[0] != groups[i]:
                break
            cloud_data[i].append(t[1:])

    # clouds provided as separate lists, in time order
    # clouds formatted as tuples with (time (s), altitude (m), radial_velocity (m/s))
    return cloud_data


def condense_common_values(list):
    condensed = []
    for x in list:
        if x not in condensed:
            condensed.append(x)
    return condensed


def plotCloud(date, time_ld, time_sonde):
    clouds = clusterClouds(date, time_ld, time_sonde)
    clusters_plot = [id for c in enumerate(clouds) for id in [c[0]] * len(c[1])]
    plt.ylabel('Height (m)')
    plt.xlabel('Time (h)')
    x_plot = [secondsToHours(c[0]) for cloud in clouds for c in cloud]
    y_plot = [c[1] for cloud in clouds for c in cloud]

    plt.scatter(x_plot, y_plot, c=clusters_plot, cmap='rainbow')
    point = plt.ginput(1)[0]
    x_input = point[0]


    cluster_input = clusters_plot[find_closest_value_index(x_plot, x_input)]
    t1 = hoursToSeconds(x_plot[find_approx_value_index(clusters_plot, cluster_input, 0)])  # function finds the first index with the value
    t2 = hoursToSeconds(x_plot[find_approx_value_index(clusters_plot, cluster_input + 1, 0) - 1])

    plt.close()
    draw = plotCloudAndVelocity(cluster_input, clouds)
    draw(t1)
    return (int)(t1), (int)(t2), draw


def plotCloudAndVelocity(cloud_id, clouds):
    # cloud data should be a tuple as specified as the return value in clusterClouds

    cloud = clouds[cloud_id]
    time_plot = [secondsToHours(c[0]) for c in cloud]
    height_plot = [c[1] for c in cloud]
    rv_plot = [c[2] for c in cloud]

    x_plot = [secondsToHours(c[0]) for cloud in clouds for c in cloud]
    y_plot = [c[1] for cloud in clouds for c in cloud]
    clusters_plot = [id for c in enumerate(clouds) for id in [c[0]] * len(c[1])]
    fig, main = plt.subplots(1)
    plt.scatter(x_plot, y_plot, c=clusters_plot, cmap='rainbow')

    height_heatmap = [100] * len(time_plot)
    max_height = max(height_plot)
    min_height = min(height_plot)
    adjustment_denom = max_height - min_height
    for i in range(len(height_heatmap)):
            adjustment = (height_plot[i] - min_height) / adjustment_denom
            height_heatmap[i] = height_heatmap[i] * adjustment

    plt.subplots(2, 1)
    ax = plt.subplot(211)
    plt.ylabel('Height (m)')
    plt.xlabel('Time (h)')
    plt.scatter(time_plot, height_plot)
    ax2 = plt.subplot(212)
    plt.scatter(time_plot, rv_plot, c=height_heatmap, cmap='bwr')
    plt.ylabel('Radial Velocity (m/s)')
    plt.xlabel('Time (h)')

    def draw(time):

        someY = 2000
        [p.remove() for p in reversed(ax.patches)]
        [p.remove() for p in reversed(main.patches)]

        xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]

        ax.add_patch(Rectangle((time - xrange/40, ax.get_ylim()[0]), xrange/20, yrange, facecolor="green", alpha = 0.5))

        xrange_main = main.get_xlim()[1] - main.get_xlim()[0]
        yrange_main = main.get_ylim()[1] - main.get_ylim()[0]

        main.add_patch(Rectangle((time - xrange_main/80, main.get_ylim()[0]), xrange_main/40, yrange_main, facecolor="black", alpha = 0.5))

    return draw


def map_clusters(cloud_height, altitude, time, t_s, t_ll, t_lh):
    color_map = [1]
    curr_cluster = 1
    t_start = 0

    for i in range(1, len(time)):
        thresh_index = find_approx_value_index(altitude, cloud_height[i], 5)
        if time[i] - time[i - 1] < t_s[thresh_index] and time[i] - time[t_start] < t_lh[thresh_index]:
            color_map.append(curr_cluster)
        else:
            if time[i - 1] - time[t_start] < t_ll[thresh_index]:
                curr_cluster -= 1
                for j in range(t_start, i):
                    color_map[j] = 0
            curr_cluster += 1
            color_map.append(curr_cluster)
            t_start = i
    return color_map


def find_approx_value_index(lst, value, threshold):
    for i in range(len(lst)):
        if value - threshold <= lst[i] <= value + threshold:
            return i
    return -1


def find_closest_value_index(lst, value):
    closest = 0
    closest_dist = abs(value - lst[0])
    for i in range(len(lst)):
        if abs(value - lst[i]) < closest_dist:
            closest = i
            closest_dist = value - lst[i]
    return closest


def optimal_k(points, kmax):
    # Using silhouette score to determine optimal k value for kmeans
    sil = []

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        labels = kmeans.labels_
        sil.append(skmet.silhouette_score(points, labels, metric='euclidean'))

    return sil.index(max(sil)) + 2


def readSondeData(date, time):
    """
    :param date: Provide date in the following format as a string: YearDayMonth. Ex: 20180504 -> 5-4-2018
    :param time: Provide time in the following format as a string: HourMinuteSecond Ex: 113000 -> 11:30
    :return: dictionary with the following structure:
    "u_wind" : wind speed in u direction (x direction)
    "v_wind" : wind speed in v direction (y direction)
    "alt" : altitude of cloud
    """
    filename = "sgpsondewnpnC1.b1." + str(date) + "." + str(time) + ".cdf"
    f = Dataset(filename, "r", format="NETCDF4")
    # print(f.variables)
    return_dict = {}
    temp = f.variables['alt'][:]
    removed_indices = []

    altitude = []
    for i in range(len(temp)):
        a = temp[i]
        if 500 < a < 5000:
            altitude.append(a)
        else:
            removed_indices.append(i)

    u_temp = f.variables['u_wind'][:]
    v_temp = f.variables['v_wind'][:]
    u_wind = []
    v_wind = []
    for i in range(len(u_temp)):
        if i not in removed_indices:
            u_wind.append(u_temp[i])
            v_wind.append(v_temp[i])

    return_dict['u_wind'] = u_wind
    return_dict['v_wind'] = v_wind
    return_dict['altitude'] = altitude
    f.close()

    return return_dict


# Center of COGS data is where the LIDAR is positioned, so we can see the LIDAR data along with the COGS data
def readCOGSData():
    return 0

if __name__ == '__main__':
    main()
