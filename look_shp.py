# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:16:17 2013
script: look_shp.py
@author: gcoombes
"""
import fiona
import os

#base_dir = r"J:\Projects\QLD Projects\Q0139 - Chevron IAG\5. Data\GIS\Smoothed Shapefiles\Entrained Exposure Prob"
#fn = r"IAG_BLOW_SUM_Entrained_Exposure_Prob(576)(0-10m)_sm.shp"
base_dir = r"J:\Tools\Python\arc_smooth\test_process"
#fn = r"IAG_BLOW_SUM_Aromatic_Exposure_Prob(576)(0-10m)_b10km_sm.shp"
#fn = r"IAG_BLOW_SUM_Aromatic_Exposure_Prob(576)(0-10m).shp"
#fn = r"IAG_BLOW_SUM_Surface_Exposure_Prob(1um).shp"
#fn = r"IAG_BLOW_SUM_Entrained_Exposure_Prob(960)(0-10m).shp"
#fn = r"IAG_BLOW_SUM_Aromatic_Exposure_Prob(576)(0-10m)_b10km_sm.shp"
fn = r"Cond_12wk_Win_Surface_Exposure_Time(1um).shp"
#fn = "test_write.shp"
fp = os.path.join(base_dir, fn)
with fiona.collection(fp, "r") as source:
    for rec in source:
        print(rec['properties'])
#        print(rec)
        
#        break
print("Done")