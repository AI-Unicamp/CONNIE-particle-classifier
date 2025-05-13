
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class PlotCanvas(FigureCanvas):
    def __init__(self, parent, file_data, curr_idx):
        self.fig, self.ax = plt.subplots()
        self.cax = self.fig.add_axes([0.84, 0.1, 0.03, 0.8])  
        self.fig.subplots_adjust(right=0.75)

        super(PlotCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.cbar = None
        self.plot(file_data, curr_idx)

    def plot(self, file_data, curr_idx):
        self.ax.clear()
        # current event x and y positions and value
        x_position = file_data["xPix"][curr_idx]
        y_position = file_data["yPix"][curr_idx]
        pixel_values = file_data["ePix"][curr_idx]
        width = np.max(x_position) - np.min(x_position) + 1
        height = np.max(y_position) - np.min(y_position) + 1
        image = np.ones((height, width)) * np.nan

        # Fill image matrix with the pixel value 
        for x, y, value in zip(x_position, y_position, pixel_values):
            image[y - np.min(y_position), x - np.min(x_position)] = value
        
        im = self.ax.imshow(image, cmap="viridis",
                            interpolation="none",
                            origin="upper",
                            vmin=0,
                            vmax=np.nanmax(image),
                            extent=[np.min(x_position), np.max(x_position),
                                    np.max(y_position), np.min(y_position)])

        if self.cbar:
            self.cbar.update_normal(im)  # Efficiently update colorbar limits
        else:
            self.cbar = self.fig.colorbar(im, ax=self.ax, cax=self.cax,
                                          pad=0.1)
            self.cbar.set_label("Electron (e-)")

        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()