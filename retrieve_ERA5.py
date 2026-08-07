import cdsapi
c = cdsapi.Client()

area='72.75/38.25/72.5/38.5'  # Summit station 
#area='53.25/9.5/53.00/9.75' # Mace Head
#area='36.75/97.25/36.5/97.5'  # Southern great Plains
#area='52/4.75/51.75/5'  # Cabauw  
model_level_request = True

if model_level_request:
    c.retrieve(
        'reanalysis-era5-complete', { # Requests follow MARS syntax
                                      # Keywords 'expver' and 'class' can be dropped. They are obsolete
                                      # since their values are imposed by 'reanalysis-era5-complete'
            'date' : '2021-01-01/to/2021-12-31', # The hyphens can be omitted
            'levelist': '110/to/137', # 1 is top level, 137 the lowest model level in ERA5. Use '/' to separate values.
            'levtype' : 'ml',
            'param' : '130/131/132/133/152', # Full information at https://apps.ecmwf.int/codes/grib/param-db/
                                             # The native representation for temperature is spherical harmonics
            'stream' : 'oper', # Denotes ERA5. Ensemble members are selected by 'enda'
            'time' : '00/to/23/by/1', # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18'
            'type' : 'an',
            'area': area,
            'grid' : '0.25/0.25', # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
            'format' : 'grib', # Output needs to be regular lat-lon, so only works in combination with 'grid'!
        }, 
    'ERA5-model_levels.nc'
    ) # Output file. Adapt as you wish. 

