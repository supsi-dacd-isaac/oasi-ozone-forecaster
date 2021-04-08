# --------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------#

QUALITY_CODES = {
                     0: 'SUSPECT',
                     2: 'SUSPECT',
                     4: 'SUSPECT',

                     3: 'GOOD',
                     5: 'GOOD',
                     8: 'GOOD',
                     9: 'GOOD',
                    10: 'GOOD',

                     1: 'CORRUPT',
                     6: 'CORRUPT',

                     7: 'EXTRAORDINARY',
                    -1: 'NO_DATA',
                }

ENCODING = 'iso-8859-1'

LOCATIONS = {
                # OASI locations
                'bioggio': 'BIO',
                'brionesm': 'BRI',
                'chiasso': 'CHI',
                'locarno': 'LOC',
                'mendrisio': 'MEN',
                'sagno': 'SAG',
                'tesserete': 'TES',

                # MeteoSvizzera locations
                'meteosvizzera-stabio': 'MS-STB',
                'meteosvizzera-montegeneroso': 'MS-GEN',
                'meteosvizzera-lugano': 'MS-LUG',
                'meteosvizzera-ludianomatro': 'MS-MAT',
                'meteosvizzera-locarnomonti': 'MS-OTL',
                'meteosvizzera-gutsch': 'MS-GUT',
                'meteosvizzera-acquarossacomprovasco': 'MS-COM',

                # ARPA locations
                'Bormio': 'IT-BOR',
                'Castronno': 'IT-CAS',
                'Chiavenna': 'IT-CHI',
                'Como': 'IT-COM',
                'Gallarate': 'IT-GAL',
                'Lecco': 'IT-LEC',
                'Merate': 'IT-MER',
                'Milano': 'IT-MIL',
                'Saronno': 'IT-SAR',
                'Varese': 'IT-VAR',
            }

FILLED_DATA_LOCATIONS = {
                            # OASI locations
                            'Bioggio': 'BIO',
                            'Brione': 'BRI',
                            'Chiasso': 'CHI',
                            'Locarno': 'LOC',
                            'Mendrisio': 'MEN',
                            'Sagno': 'SAG',
                            'Tesserete': 'TES',

                            # ARPA locations
                            'Bormio': 'IT-BOR',
                            'Castronno': 'IT-CAS',
                            'Chiavenna': 'IT-CHI',
                            'Como': 'IT-COM',
                            'Gallarate': 'IT-GAL',
                            'Lecco': 'IT-LEC',
                            'Merate': 'IT-MER',
                            'Milano': 'IT-MIL',
                            'Saronno': 'IT-SAR',
                            'Varese': 'IT-VAR',
                        }


NOT_USED_VARS = ['Tdew']

METEO_FORECAST_STATIONS = ['TICIA', 'P_SAG', 'P_CAP', 'P_BIO', 'MTR', 'COM', 'GEN']

CHUNKS_FORECASTS = {
                        '03': {
                                    'chunk1': {'start': 1, 'end': 8},
                                    'chunk2': {'start': 9, 'end': 17},
                                    'chunk3': {'start': 18, 'end': 25},
                                    'chunk4': {'start': 26, 'end': 33},
                        },

                        '12': {
                                    'chunk1': {'start': 6, 'end': 11},
                                    'chunk2': {'start': 12, 'end': 17},
                                    'chunk3': {'start': 18, 'end': 23},
                                    'chunk4': {'start': 24, 'end': 33},
                        },
                   }

OZONE_INDEXES_LIMITS = ['1: [0-60] μg/m³',   '2: [61-120] μg/m³',   '3: [121-135] μg/m³',
                        '4: [136-180] μg/m³', '5: [181-240] μg/m³', '6: [>240] μg/m³']

SIGNAL_EXCEPTIONS = ['DayWeek', 'IsWeekend', 'IsHolyday', 'RHW__d0', 'RHW__d1', 'RHW__d2', 'RHW__d3', 'RHW__d4',
                     'RHW__d5']


HOLYDAYS = ['06-29', '08-01', '08-15']