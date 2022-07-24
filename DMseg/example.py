
import pkg_resources
from DMseg.dmseg import *

betafile = pkg_resources.resource_filename('DMseg', 'data/example_beta.csv')
colDatafile = pkg_resources.resource_filename('DMseg', 'data/example_colData.csv')
positionfile = pkg_resources.resource_filename('DMseg', 'data/example_position.csv')

DMseg_res = pipeline(betafile, colDatafile, positionfile)

print(DMseg_res.loc[DMseg_res.FWER<0.05])
