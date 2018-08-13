import snappy
from snappy import  ProductIO
files = ProductIO.readProduct
import numpy as np 
files = ProductIO.readProduct('/Users/yangchao/GitHub/winddir/new/ecmwf/match_S1A_IW_GRDH_1SDV_20160303T222554_20160303T222623_010211_00F13B_D028.dim')

data = np.array([2, 3])