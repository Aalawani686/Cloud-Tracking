from netCDF4 import Dataset
import numpy as np



def main():
    file = Dataset("SGP_COGS20190715.nc", "a", format="NETCDF4", chunks={'time'})
    cloud = file.variables['cloud'][:]
    # print(cloud.shape)
    # print(a)
    # print(np.sum(a, axis=0))
    # print(np.sum(a, axis=1))
    # print(np.sum(a, axis=2))
    instance = cloud[0]

    arr = np.array([[[1,0], [0,0]], [[1, 0], [0, 0]]])
    # arr = arr[arr != 6]
    # print(len(arr))
    instance[instance < 0] = 0
    xy = np.sum(instance, axis=0)
    # xz = np.sum(cloud[0], axis=1)
    # yz = np.sum(cloud[0], axis=2)
    xz = np.sum(instance, axis=2)
    # print(xz)
    z = np.sum(xz, axis=1)
    print(z)
    for i in range(0, len(xy)-15):
        for j in range(0, len(xy[0])):
            print(xy[i][j], end = " ")
        print()
    #
    # print(xy[0][0])
    # print(xz[0][0])
    # print(yz[0][0])


    '''
    zero is z squash
    one is y squash
    two is x squash
    '''



if __name__ == '__main__':
    main()
