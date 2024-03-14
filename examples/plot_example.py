import matplotlib.pyplot as plt

fig = plt.figure()
for ix in range(2):
    for iy in range(2):
        ax = fig.add_subplot(nx, ny, ny * ix + iy + 1)
        ax.imshow(grid_results[(ix, iy)]["fast_em"], cmap="gray")
plt.show()
