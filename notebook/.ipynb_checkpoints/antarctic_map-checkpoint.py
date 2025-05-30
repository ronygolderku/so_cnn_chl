import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import xarray as xr
from matplotlib.lines import Line2D
def antarctic_map(ax):
    def plot_text(p1, p2, ax, ang_d, txt):
        l1 = np.array((p1[0], p1[1]))
        l2 = np.array((p2[0], p2[1]))
        th1 = ax.text(l1[0], l1[1], txt, fontsize=10,
                      transform=nonproj,
                      ha="center",
                      rotation=ang_d, rotation_mode='anchor')

    import cartopy.crs as ccrs
    nonproj = ccrs.PlateCarree()
    
    # Set the extent and add features
    ax.set_extent([-180, 180, -90, -30], nonproj)
    ax.add_feature(cfeature.LAND, color='darkgrey')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.25)
    
    # Add gridlines with labels and custom style
    ax.gridlines(nonproj, draw_labels=False, linewidth=1, xlocs=range(-180, 171, 30), ylocs=[],
                  color='gray', alpha=0.5, linestyle='--', zorder=10)
    ax.gridlines(nonproj, draw_labels=True, linewidth=1, xlocs=[], ylocs=range(-90, -30, 10),
                  color='gray', alpha=0.5, linestyle='--', zorder=10)

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    for lon in range(-180, 180, 30):
        lat = -33  # determined by inspection

        a1, a2 = -29.5, -39  # text anchor for general use ...
        # ... need adjustments in some cases

        if lon >= 90 and lon <= 170:
            plot_text([lon, a1 + 2.35], [lon, a2], ax, -lon - 180, str(lon) + "°E")
            # Special rotation+shift
        elif lon < -90 and lon >= -170:
            # Need a1+2 to move texts in line with others
            plot_text([lon, a1 + 2.5], [lon, a2], ax, -lon + 180, str(-lon) + "°W")
            # Special rotation+shift
        elif lon > 0:
            plot_text([lon, a1], [lon, a2], ax, -lon, str(lon) + "°E")
        elif lon == 0:
            plot_text([lon, a1], [lon, a2], ax, lon, str(lon) + "°")
        elif lon == -180:
            plot_text([lon, a1 + 2.2], [lon, a2], ax, lon + 180, str(-lon) + "°")
        else:
            plot_text([lon, a1], [lon, a2], ax, -lon, str(-lon) + "°W")
            
            
    return ax