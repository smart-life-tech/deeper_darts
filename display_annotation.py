import matplotlib.pyplot as plt


img = plt.imread("dataset/images/d1_02_22_2020/IMG_4595.JPG")
fig, ax = plt.subplots()
ax.imshow(img, extent=[0, 1, 0, 1])

xys= [[0.4405684754521964, 0.12782440284054228], [0.5594315245478035, 0.8721755971594577], [0.12758397932816537, 0.5593931568754034], [0.8724160206718347, 0.4406068431245966], [0.7945736434108527, 0.3867010974822466], [0.9308785529715762, 0.3382827630729503], [0.7493540051679587, 0.3750806972240155]]


i = 0
for xy in xys:
    if i < 4:
        plt.plot(1-xy[0], xy[1], 'bo')
    else:
        plt.plot(xy[0], 1-xy[1], 'ro')
    i+=1

plt.show()