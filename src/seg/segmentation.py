#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:06:14 2022

@author: nicholas.lusk
"""
import os
import s3fs
import shutil
Import utils

import dask.array as da

from glob import glob
from natsort import natsorted
from imlib.IO.cells import save_cells, get_cells
from cellfinder_core.detect import detect
from argschema.fields import Str , Int, Boolean
from argschema import ArgSchemaParser, ArgSchema

example_input = {"signal_data": "/Users/nicholas.lusk/allen/programs/aind/workgroups/msma/SmartSPIM/ephys/SmartSPIM_625382_2022_09_09_22_10_21_stitched/stitched/OMEZarr/Ex_488_Em_525.zarr/0",
                 "chunk_size": 500,
                 "bkg_subtract": True,
                 "save_dir": None}


class SegSchema(ArgSchema):
    """
    Schema format for Segmentation
    """       
    signal_data = Str(metadata = {'required': True, "description": "Zarr file to be segmented"})
    chunk_size = Int(metadata = {'required': True, "description": "number of planes per chunk (needed to prevent memory crashes)"})
    bkg_subtract = Boolean(metadata = {'required': True, "description": "whether to run background subtraction"})
    signal_data = Str(metadata = {'required': True, "description": "location to save segmentation .xml file"})

class Segment(ArgSchemaParser):
    
    """
    Class for segmenting lightsheet data
    """
    
    default_schema = SegSchema
    
    def run(self):
    
        # create temporary folder for storing chunked data
        self.tmp_path = os.path.join(os.getcwd(), 'tmp')    
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
        
        # load signal data
        if "s3:/" in self.args['input_data']:
            print("S3 data")
            s3 = s3fs.S3FileSystem(anon=True)
            store = s3fs.S3Map(root = self.args['input_data'], s3 = s3, check = False)
            signal_array = da.from_zarr(store).squeeze()
        elif "gs:/" in self.args['input_data']:
            print("GCP data")
            signal_array = da.from_zarr(store).squeeze()
        else:
            print("cannot recognize")
            exit(0)
        
   	
	# check if background sublations will be run
	if self.args['bkg_subtract']:
	    signal_array
	
        # effective parameters found to work with SmartSPIM nuclei signals
        start_plane = 0
        end_plane = -1
        n_free_cpus = 2
        voxel_sizes = [2, 1.8, 1.8] # in microns
        soma_diameter = 9 # in microns
        ball_xy_size = 6
        ball_z_size = 8
        ball_overlap_fraction = 0.6
        log_sigma_size = 0.1
        n_sds_above_mean_thresh = 3
        soma_spread_factor = 1.4
        max_cluster_size = 100000

        detect.main(
                    signal_array,
                    start_plane,
                    end_plane,
                    self.tmp_path,
                    self.arg['chunk_size'],
                    voxel_sizes,
                    soma_diameter,
                    max_cluster_size,
                    ball_xy_size,
                    ball_z_size,
                    ball_overlap_fraction,
                    soma_spread_factor,
                    n_free_cpus,
                    log_sigma_size,
                    n_sds_above_mean_thresh,
                    )
    
    def merge(self):
        
        # load temporary files and save to a single list
        cells = []
        tmp_files = glob(os.path.join(self.tmp_path, '.xml'))     
        for f in natsorted(tmp_files):
            cells.extend(get_cells(f))
        
        # save list of all cells
        save_cells(os.path.join(self.args['save_path'], 'detected_cells.xml'))
        
        # delete tmp folder
        try:
            shutil.rmtree(self.tmp_path)
        except OSError as e:
            print("Error removing temp file %s : %s" % (self.tmp_path, e.strerror))
        
if __name__ == '__main__':
    
    seg = Segment(example_input)
    seg.run()
    seg.merge()
    

    