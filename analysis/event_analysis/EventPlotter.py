"""
Class for plotting hit data for CNN mPMT datasets
"""

import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from watchmal.dataset.cnn_mpmt.cnn_mpmt_dataset import CNNmPMTDataset

class EventPlotter():
    def __init__(self, h5_path, mpmt_positions_file, geo_path):
        # load dataset
        self.dataset = CNNmPMTDataset(h5file=h5_path, mpmt_positions_file=mpmt_positions_file, is_distributed=False)

        # ========================================================================

        # load tubes
        geofile = np.load(geo_path, allow_pickle=True)
        self.tubes = geofile[ 'tube_no' ]
        self.tube_xyz = geofile[ 'position' ]

        tube_x = self.tube_xyz[:,0]
        tube_y = self.tube_xyz[:,1]
        tube_z = self.tube_xyz[:,2]
        
        # define constant attributes for short tank geometry
        self.barrel_half_height = tube_y.max()
        self.R = (tube_x.max() - tube_x.min()) / 2.0

        self.endcap_limit = self.barrel_half_height - 10

        # offset for endcaps due to flattening
        self.y_offset = self.barrel_half_height + self.R

        # offsets when making plot assume positive values only
        self.positive_x_offset = np.pi*self.R
        self.lower_endcap_offset = self.y_offset + self.R
        self.upper_endcap_offset = self.y_offset + self.lower_endcap_offset
        
        # ========================================================================

        # define map for plotting
        self.flat_map_positive = self.PMT_to_flat_cylinder_map_positive(self.tubes, self.tube_xyz)

        # set up dimensions for preimage with short tank data from mapping
        all_tube_mappings = list(self.flat_map_positive.values())
        all_tubes_x = np.array([tube_mapping[0] for tube_mapping in all_tube_mappings])
        all_tubes_y = np.array([tube_mapping[1] for tube_mapping in all_tube_mappings])

        min_pmt_x_value = all_tubes_x.min()
        max_pmt_x_value = all_tubes_x.max()

        min_pmt_y_value = all_tubes_y.min()
        max_pmt_y_value = all_tubes_y.max()

        self.preimage_dimensions = [max_pmt_y_value + min_pmt_y_value + 1, max_pmt_x_value + min_pmt_x_value + 1]
    
    def get_event_data_from_index(self, index):
        """
        Retrieve event data from dataset

        Args:
            index               ... index of event in dataset
        """
        return self.dataset.retrieve_event_data(index)

    # ========================================================================
    # Mapping and plotting functions

    def PMT_to_flat_cylinder_map_positive(self, tubes, tube_xyz):
        """
        Build dictionary of PMT number, to (x,y) on a flat cylinder

        Args:
            tubes                ... tube numbers from geometry file
            tube_xyz             ... tube locations from geometry file
        
        Returns: dict mapping event tube index to flattened cylinder coordinates 
        
        NOTE: Tube numbers in full geometry file go from 1:NPMTs, but it seems like
        the event data number from 0:NPMTs-1, so must subtract 1 from tube number
        """
        mapping = {}
        for idx, tube in enumerate(tubes):
            x = tube_xyz[idx,0]
            y = tube_xyz[idx,1]
            z = tube_xyz[idx,2]
            if ( y > self.endcap_limit ):
                # in top circle of cylinder
                xflat = x + self.positive_x_offset
                yflat = z + self.upper_endcap_offset
                mapping[ int( tube-1 ) ] = [ int(round(xflat)), int(round(yflat)) ]
            elif ( y < -self.endcap_limit):
                # in bottom circle of cylinder
                xflat = x + self.positive_x_offset
                yflat = z + self.R
                mapping[ int( tube-1 ) ] = [ int(round(xflat)), int(round(yflat)) ]
            else:
                # in barrel part of cylinder
                theta = math.atan2( z, x )
                xflat = self.R * theta + np.pi*self.R
                yflat = y + self.lower_endcap_offset
                mapping[ int( tube-1 ) ] = [ int(round(xflat)), int(round(yflat)) ]
        return mapping
    
    def display_event(self, index, data_type='charge', dark=False):
        """
        Plot single event by index

        Args:
            index               ... index of event in dataset
            data_type           ... type of event data to display, one of 'charge' or 'time'
        """
        pmts, charges, times = self.get_event_data_from_index(index)

        if data_type == 'charge':
            self.display_data(pmts, charges, dark=dark)
        elif data_type == 'time':
            self.display_data(pmts, times, dark=dark)

    def display_data(self, tubes, quantities, title="Charge", cutrange=[-1,-1], ax=None, figsize=[30,30], dark=False):
        """
        Plot quantities from an event on a flattened cylinder

        Args:
            tubes               ... np.array of PMTs that were hit
            quantities          ... np.array of PMT quantities (either charge or time)
            title               ... title to add to display
            cutrange            ... minimum and maximum values on plot (or set both same for default)
            figsize             ... figure dimensions
            dark                ... whether to use light or dark mapping
        """
        if dark:
            cmap = copy.copy(plt.cm.viridis)
            cmap.set_bad(color='black')
            plt.set_cmap(cmap)
        else:
            plt.set_cmap('gist_heat_r')
        
        if ax == None:
            fig, ax = plt.subplots(figsize=figsize, facecolor='w')
            fig.suptitle(title, fontsize=80)
        
        im = self.plot_data(tubes, quantities, [-1,-1], ax)

        plt.rc('xtick', labelsize=24) 
        plt.rc('ytick', labelsize=24) 
        plt.xlabel('Distance CCW on perimeter from x-axis (cm)', fontsize=48)
        plt.ylabel('Y (cm)', fontsize=48)

        # Create colourbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=24)

        # Fix title height
        plt.subplots_adjust(top=0.5)
        plt.tight_layout()
        
        return im

    def plot_data(self, tubes, quantities, cutrange, ax):

        PMTFlatMapPositive = self.flat_map_positive
        preimage = np.zeros(self.preimage_dimensions)
        
        imgmin = quantities.min()
        imgmax = quantities.max()

        for idx, tube in enumerate( tubes ):
            if cutrange[0] != cutrange[1]:
                if quantities[idx] < cutrange[0] or quantities[idx] > cutrange[1]:
                    continue
            for dx in range(-3,4):
                for dy in range(-3,4):
                    if abs(dx)==3 and abs(dy)==3:
                        continue
                    #print( "idx=", idx, " len(quantities)=",len(quantities), " tube=", tube, " len(PMTFlatMap)=", len(PMTFlatMapPositive))
                    preimage[ PMTFlatMapPositive[tube][1]+dx, PMTFlatMapPositive[tube][0]+dy ] = quantities[idx]

        if cutrange[0] != cutrange[1]:
            imgmin = cutrange[0]
            imgmax = cutrange[1]
        
        # TODO: check
        preimage[preimage == 0] = np.NaN

        im = ax.imshow(preimage, extent = [-self.positive_x_offset,self.positive_x_offset,-self.lower_endcap_offset,self.lower_endcap_offset], vmin=imgmin, vmax=imgmax)

        return im

        