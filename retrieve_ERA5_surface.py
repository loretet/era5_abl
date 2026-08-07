import cdsapi
c = cdsapi.Client()

area='72.75/38.25/72.5/38.5'  # Summit station
#area='53.25/9.5/53.00/9.75' # Mace Head
#area='36.75/97.25/36.5/97.5'  # Southern great Plains
#area='52/4.75/51.75/5'  # Cabauw
surface_level_request = True

if surface_level_request:
    c.retrieve(
        'reanalysis-era5-single-levels', { # Requests follow MARS syntax
                                           # Keywords 'expver' and 'class' can be dropped. They are obsolete
                                           # since their values are imposed by 'reanalysis-era5-complete'
            "product_type": ["reanalysis"],
            "variable": [
                "sea_surface_temperature",
                "2m_temperature",
                "surface_pressure",
                "cloud_base_height",
                "high_cloud_cover",
                "low_cloud_cover",
                "medium_cloud_cover",
                "boundary_layer_height",
                "geopotential"
            ],
            'date' : '2021-01-01/2021-12-31', # The hyphens can be omitted
            #'stream' : 'oper', # Denotes ERA5. Ensemble members are selected by 'enda'
            'time' : '00/to/23/by/1', # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18'
            #'type' : 'an',
            'area': area,
            #'grid' : '0.25/0.25', # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
            'format' : 'grib', # Output needs to be regular lat-lon, so only works in combination with 'grid'!
        }, 
    'ERA5-surface_levels.nc'
    ) # Output file. Adapt as you wish. 
