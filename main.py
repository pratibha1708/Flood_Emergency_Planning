import json

import geopandas as gpd
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import rasterio.plot
import tkinter as tk
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from rasterio import warp, mask
from rasterio.transform import rowcol
from rasterio.windows import Window
from rtree import index
from shapely.geometry import Point, LineString
from tkinter import *


class MyIndex(object):
    def __init__(self, pose, distance):
        self.localPos = pose
        self.distance = distance


class UserLocation:
    def __init__(self, easting, northing, radius):
        self.easting = easting
        self.northing = northing
        self.radius = radius


'''
 try:
        user_location = UserLocation(float(input("Easting: ")), float(input("Northing: ")),
                                     float(int(input("Searching Radius (km): "))))
        assert 425000 < user_location.easting < 470000 and 75000 < user_location.northing < 100000, \
        'please input a location within the box (430000, 80000)  and (465000, 95000), \
        British National Grid coordinate (easting and northing)'
    except ValueError:
        raise ValueError("Please input integer/float values into both easting and northing entry")
    maximum_height = find_high(saa=int(user_location.radius * 1000 / 5),
                               nearest_easting=5 * round(user_location.easting / 5),
                               nearest_northing=5 * round(user_location.northing / 5), easting=user_location.easting,
                               northing=user_location.northing, 
                               radius=user_location.radius, elevation_raster = elevation_raster)
        # if GUI package fails to work, activate the class and modify classes input as following:
        maximum_height = find_high(saa=int(user_location.radius * 1000 / 5),
                               nearest_easting=5 * round(user_location.easting / 5),
                               nearest_northing=5 * round(user_location.northing / 5), easting=user_location.easting,
                               northing=user_location.northing, 
                               radius=user_location.radius, elevation_raster = elevation_raster)
        netw = Road_Networks([east1,north1],[maximum_height.highest_point.x,maximum_height.highest_point.y],\
        elevation_raster = elevation_raster)
'''


class GUI:

    def __init__(self, w):
        self.lbl0 = Label(w, text='Radius (km)')
        self.lbl1 = Label(w, text='Easting')
        self.lbl2 = Label(w, text='Northing')
        self.ibl3 = Label(w, text='Please, input your current location to provide assistance').pack()
        self.var0 = tk.StringVar("")
        self.var1 = tk.StringVar("")
        self.var2 = tk.StringVar("")
        self.t0 = Entry(bd=3, textvariable=self.var0)
        self.t1 = Entry(bd=3, textvariable=self.var1)
        self.t2 = Entry(bd=3, textvariable=self.var2)
        self.lbl0.place(x=100, y=50)
        self.lbl1.place(x=100, y=75)
        self.t1.place(x=200, y=75)
        self.t0.place(x=200, y=50)
        self.lbl2.place(x=100, y=100)
        self.t2.place(x=200, y=100)
        self.b1 = Button(w, text='Run to higher ground', command=self.add)
        self.b1.place(x=100, y=150)

    def add(self):
        try:
            global east1, north1, radius
            east1 = float(self.var1.get())
            north1 = float(self.var2.get())
            radius = float(self.var0.get())
            assert 425000 < east1 < 470000 and 75000 < north1 < 100000, \
                'please input a location within the box (425000, 75000)  \
            and (470000, 100000), British National Grid coordinate (easting and northing)'

            my_w.t1.delete(0, 'end')
            # remove input everytime after press button
            my_w.t2.delete(0, 'end')
            window.destroy()

        except ValueError:
            raise ValueError("Please input integer/float values into both easting and northing entry")


class Findhigh(UserLocation):
    def __init__(self, saa, nearest_easting, nearest_northing, easting, northing, radius, elevation_raster):
        super().__init__(easting, northing, radius)
        self.saa = saa  # saa = searching area adjust
        self.nearest_easting = nearest_easting
        self.nearest_northing = nearest_northing
        self.elevation_raster = elevation_raster
        self.highest_point = None
        self.highest_elevation = None
        self.df_circle = None

    def find_high(self):
        # turn off the scientific number representation
        pd.set_option('display.float_format', lambda x: '%.5f' % x)
        # the raster pixel has a side length of 5 meters
        # we can find the nearest pixel to the user location
        # and set the pixel location as the central for elevation searching
        east_raster_position = (self.nearest_easting - 425000) / 5
        north_raster_position = (self.nearest_northing - 75000) / 5

        # minimum bounding box for search area (a circle with radius x)
        xmin, ymax = self.elevation_raster.xy(self.elevation_raster.height - north_raster_position - self.saa,
                                              east_raster_position - self.saa)
        xmax, ymin = self.elevation_raster.xy(self.elevation_raster.height - north_raster_position + self.saa,
                                              east_raster_position + self.saa)
        x = np.linspace(xmin, xmax, self.saa * 2)  # saa*2 = diameter of search radius
        y = np.linspace(ymax, ymin, self.saa * 2)

        # making the code adaptable to task 6 (extend region)
        # if row number <0, use first row. if row number > max row, use final row
        # the same rule applies to column
        row_start = np.array([self.elevation_raster.height - north_raster_position - self.saa])
        row_start[row_start[0] < 0] = 0
        row_start = int(row_start)

        row_end = np.array([self.elevation_raster.height - north_raster_position + self.saa])
        row_end[row_end[0] > self.elevation_raster.height] = self.elevation_raster.height
        row_end = int(row_end)

        column_start = np.array([east_raster_position - self.saa])
        column_start[column_start[0] < 0] = 0
        column_start = int(column_start)

        column_end = np.array([east_raster_position + self.saa])
        column_end[column_end[0] > self.elevation_raster.width] = self.elevation_raster.width
        column_end = int(column_end)

        # create x-y 2d arrays for data frame
        xs, ys = np.meshgrid(x, y)
        zs = self.elevation_raster.read(1, window=Window.from_slices(
            (row_start, row_end),
            (column_start, column_end)))
        # some transformation ideas come from user2856 of gis.stackexchange
        # link: https://gis.stackexchange.com/questions/384581/raster-to-geopandas
        # Jan 18 '21 at 6:14
        raster_data = {"X": pd.Series(xs.ravel()),  # X -> easting
                       "Y": pd.Series(ys.ravel()),  # Y -> northing
                       "Elevation": pd.Series(zs.ravel())}

        df = pd.DataFrame(data=raster_data)
        # apply trigonometry to select elevation values within xkm radius and form a dataframe called df_circle
        df_circle = df.drop(df[((df.X - self.easting) ** 2 + (
                df.Y - self.northing) ** 2) ** 0.5 > self.radius * 1000].index)
        self.df_circle = df_circle
        # reset index to properly loop the dataframe
        df_circle = df_circle.reset_index()
        # drop any null values if exists
        df_after_null = df_circle.dropna()

        if df_circle.shape == df_after_null.shape:
            print('No Null Values detected')
        else:
            print('Null Values detected, proceed with caution')

        # Find maximum elevation value and return the pixel's central coordinates
        highest_point_coordinate = Point(round(df_circle.iloc[df_circle['Elevation'].idxmax()][1]),
                                         round(df_circle.iloc[df_circle['Elevation'].idxmax()][2]))

        # print the results
        print('highest point coordinate:' + str(highest_point_coordinate))
        print('highest point elevation:' + str(df_circle.iloc[df_circle['Elevation'].idxmax()][3]))

        self.highest_point = highest_point_coordinate
        self.highest_elevation = df_circle.iloc[df_circle['Elevation'].idxmax()][3]


class Road_Networks():
    def __init__(self, p_location, p_highest, elevation_raster):
        self.p_location = p_location
        self.p_highest = p_highest
        self.elevation_raster = elevation_raster
        self.p_node = None
        self.highest_node = None
        self.s_node_key = None
        self.e_node_key = None
        self.road_links = None
        self.shortest_path_gpd = None

    def xy_transform(self, elevation_array, transformer, x, y):
        ele = elevation_array[rowcol(transformer, x, y)]
        return ele

    def uphill_time(self, coords, transformer, elevation_array):
        # Calculate the value of elevation increase
        uphill_elevation = 0
        i = 0
        x, y = coords[0]
        s_elevation = self.xy_transform(elevation_array, transformer, x, y)
        for point in coords[1:]:
            x, y = point
            elevation = self.xy_transform(elevation_array, transformer, x, y)
            if elevation > s_elevation:
                # Cumulative increase in meters
                ele_increase = elevation - s_elevation
                uphill_elevation = ele_increase + uphill_elevation
            s_elevation = elevation
            i += 1
        # an additional minute is added for every 10 meters of climb
        uph_time = uphill_elevation / 10
        return uph_time

    def find_node(self, roadlist, node,str):
        for i in roadlist:
            for j in self.road_links[i]["coords"]:
                if j == node:
                    key = self.road_links[i][str]
        return key

    def find_nearest(self, idx, loc, list):
        for i in idx.nearest(loc, 1):
            node = list[i]
        return node

    def nearest_node(self):
        # Read itn network information
        itn_path = "Material/itn/solent_itn.json"
        with open(itn_path, "r") as file:
            itn_json = json.load(file)
        # Obtain road node and route information
        road_nodes = itn_json['roadnodes']
        self.road_links = itn_json['roadlinks']
        road_links_keys_list = list(self.road_links.keys())
        road_list = []
        # Get path coordinates
        for i in road_nodes:
            road_list.append(road_nodes[i]['coords'])
        # Use rtree to find the nearest node
        idx = index.Index()
        for i, coords in enumerate(road_list):
            pose = coords + coords
            my_index = MyIndex(pose, 100)
            idx.insert(i, my_index.localPos, obj=my_index)
        # Find the node closest to the user's coordinates
        self.p_node = self.find_nearest(idx, self.p_location, road_list)
        # Find the closest node to the highest point
        self.highest_node = self.find_nearest(idx, self.p_highest, road_list)
        # Find the starting point of the road closest to the user point
        self.s_node_key = self.find_node(road_links_keys_list,self.p_node,"start")
        # Find the ending point of the road closest to the highest point
        self.e_node_key = self.find_node(road_links_keys_list, self.highest_node, "end")

    def shortest_path(self):
        # Create a bidirectional graph
        road_network = nx.DiGraph()
        elevation_array = self.elevation_raster.read(1)
        # Get the affine matrix
        transform = self.elevation_raster.transform
        for link in self.road_links:
            road_length = self.road_links[link]['length']
            road_coordinates = self.road_links[link]['coords']
            # Uphill time calculation
            additional_minute = self.uphill_time(road_coordinates, transform, elevation_array)
            # Time spent walking at a constant speed
            initial_minute = road_length * 0.012
            naismith_time = initial_minute + additional_minute
            road_network.add_edge(self.road_links[link]['start'],
                                  self.road_links[link]['end'],
                                  fid=link, length=self.road_links[link]['length'], weight=naismith_time)
            # Because the length of the back and forth uphill
            # is inconsistent, so calculate the reverse time weight
            additional_minute = self.uphill_time(road_coordinates[-1:], transform, elevation_array)
            naismith_time = initial_minute + additional_minute
            road_network.add_edge(self.road_links[link]['end'],
                                  self.road_links[link]['start'],
                                  fid=link,
                                  length=self.road_links[link]['length'], weight=naismith_time)
        # Use an algorithm to calculate the shortest path
        path = nx.dijkstra_path(road_network, source=self.s_node_key, target=self.e_node_key, weight="weight")
        links = []
        geom = []
        first_node = path[0]
        for node in path[1:]:
            link_fid = road_network.edges[first_node, node]['fid']
            links.append(link_fid)
            geom.append(LineString(self.road_links[link_fid]['coords']))
            first_node = node
        # Get the shortest path
        shortest_path_gpd = gpd.GeoDataFrame({'fid': links, 'geometry': geom})
        self.shortest_path_gpd = shortest_path_gpd


class Plot:
    def __init__(self, radius, ue, un, p_n, h_n, s_p, elevation_raster, hc):
        self.elevation_raster = elevation_raster
        self.radius = radius
        self.ue = ue
        self.un = un
        self.p_n = p_n
        self.h_n = h_n
        self.s_p = s_p
        self.hc = hc

    def plot(self):
        # Defining user point, highest point and shortest path
        s_p = self.s_p
        u_location = Point(self.un, self.ue)
        user_buffer = u_location.buffer(self.radius*1000)
        user_location_c = Point(self.ue, self.un).buffer(150)
        user_e, user_n = user_location_c.exterior.xy
        hc_polygon = self.hc.buffer(150)
        h_e, h_n = hc_polygon.exterior.xy

        # Cropping elevation raster by a circle around user point
        xy = []
        coords = list(user_buffer.exterior.coords)
        for i in coords:
            xy.append([i[1], i[0]])
        features1 = warp.transform_geom(
            {'init': 'EPSG:27700'},
            self.elevation_raster.crs,
            {"type": "Polygon",
             "coordinates": [xy]})

        ele, out_transform = mask.mask(self.elevation_raster,
                                       [features1], filled=False, crop=False, pad=False)

        # Opening background map and defining its colormap and bounds
        background = rasterio.open(str('Material/background/raster-50k_2724246.tif'))
        back_array = background.read(1)
        palette = np.array([value for key, value in background.colormap(1).items()])
        background_image = palette[back_array]
        bounds = background.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plotting background raster
        ax.imshow(background_image,
                  origin='upper',
                  extent=extent,
                  zorder=0)

        # Plotting elevation raster
        rasterio.plot.show(ele,
                           transform=out_transform,
                           ax=ax,
                           zorder=1,
                           alpha=0.8,
                           cmap='terrain',
                           title='Evacuation plan during flooding')

        # Plotting vector data
        self.s_p.plot(ax=ax, edgecolor='blue', linewidth=0.5, zorder=5)
        user_location_df = pd.DataFrame({'name': ['user_point'], 'easting': [self.ue], 'northing': [self.un]})
        user_gdf = gpd.GeoDataFrame(user_location_df, geometry=gpd.points_from_xy(user_location_df.easting,
                                                                                  user_location_df.northing))
        new_df = user_gdf.copy()
        new_df['geometry'] = new_df['geometry'].buffer(self.radius * 1000)

        s_p.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2, label='Evacuation route')
        plt.fill(user_e, user_n, 'y', label='User location', zorder=2)
        plt.fill(h_e, h_n, 'm', label='Highest point location', zorder=2)

        # Providing a plotting framework and setting the extent limits
        ylim_mi = self.un - 10000
        ylim_mx = self.un + 10000
        xlim_mi = self.ue - 10000
        xlim_mx = self.ue + 10000
        plt.ylim(max(0, ylim_mi), min(100000, ylim_mx))
        plt.xlim(max(0, xlim_mi), min(470000, xlim_mx))

        # Adding legend
        image_hidden = ax.imshow(self.elevation_raster.read(1))
        fig.colorbar(image_hidden, orientation='horizontal', label='Elevation, m')

        # Adding scalebar
        fontprops = fm.FontProperties(size=18)
        scalebar = AnchoredSizeBar(ax.transData,
                                   2500, '2500 m', 'lower left',
                                   pad=0.2,
                                   color='black',
                                   frameon=False,
                                   size_vertical=200,
                                   fontproperties=fontprops)

        # Adding north arrow
        x, y, arrow_length = 0.05, 0.95, 0.2
        ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=18,
                    xycoords=ax.transAxes)

        # Plotting all map elements on a single map
        ax = matplotlib.pyplot.gca()
        ax.add_artist(scalebar)
        plt.legend()
        plt.show()


def main():
    elevation_path = "Material/elevation/SZ.asc"
    elevation_raster = rasterio.open(elevation_path)

    maximum_height = Findhigh(saa=int(radius * 1000 / 5),
                              nearest_easting=5 * round(east1 / 5),
                              nearest_northing=5 * round(north1 / 5), easting=east1,
                              northing=north1, radius=radius, elevation_raster=elevation_raster)
    maximum_height.find_high()

    netw = Road_Networks([east1, north1], [maximum_height.highest_point.x, maximum_height.highest_point.y],
                         elevation_raster=elevation_raster)
    netw.nearest_node()
    netw.shortest_path()

    map_plot = Plot(maximum_height.radius, east1, north1, netw.p_node, netw.highest_node, netw.shortest_path_gpd,
                    elevation_raster, maximum_height.highest_point)
    map_plot.plot()


if __name__ == '__main__':
    window = tk.Tk()
    my_w = GUI(window)
    window.title('Flood Emergency Assistance Program')
    window.geometry("400x200")  # GUI dimension
    window.mainloop()
    main()