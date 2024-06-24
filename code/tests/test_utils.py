#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:25:17 2024

@author: nicholas.lusk
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd
import pandas.testing as pd_testing

from pathlib import Path

sys.path.insert(0, '/Users/nicholas.lusk/Documents/Github/aind-smartspim-quantification/code/')

from aind_smartspim_segmentation.utils import utils

class TestSmartspimUtils(unittest.TestCase):
    """Tests utility methods for smartspim quantification capsule"""
    
    def setUp(self):
        """Setting up unit test"""
        current_path = Path(os.path.abspath(__file__)).parent
        self.ccf_files = current_path.joinpath(
            "./resources/"
            )
        self.test_local_json_path = current_path.joinpath(
                "./resources/local_json.json"
            )
        self.test_structureID = 'test'
        self.resolution = 25
        self.CellCounts = utils.CellCounts(self.ccf_files, self.resolution)
        
    def tearDown(self):
        """Tearing down utils unit test"""
        
    def test_delayed_astro(self):
        """Test astropy background subtraction module"""
    
    def test_delayed_detect(self):
        """Test cellfinder delayed detect module"""
    
    def test_find_good_blocks(self):
        """Test module that locates zarr blocks with tissue"""
        
    
    @patch('logging.Logger')
    def test_create_logger(self, mock_log):
        """Test module that creates capsule logging.Logger"""
        
        
    def test_read_json_as_dict(self):
        """Test module that imports JSON files as dictionaries"""
        
        expected_result = {"some_key": "some_value"}
        result = utils.read_json_as_dict(self.test_local_json_path)
        self.assertDictEqual(result, expected_result)
        
    

if __name__ == "__main__":
    unittest.main()