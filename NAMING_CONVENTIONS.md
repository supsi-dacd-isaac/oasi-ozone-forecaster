#  Naming conventions:

## Input signals:

There are two types of input: the measures and the weather forecasts; however, the naming convention is similar in both the cases.
Each input has one of the following structures:

* `$LOC__$SIGNAL__$CASE__$FUNCTION`
* `$LOC__$SIGNAL__$CASE`

where:
* `$LOC` is the location to which the input belongs (e.g. the input related Giubiasco have names starting with `GIU` prefix)
* `$SIGNAL` is the OASI or Meteosvizzera signal code related to the input (e.g. `O3` for the ozone) 
* `$CASE` defines if and how the data have to be aggregated
* `$FUNCTION` explains the (eventual) aggregation function related to `$CASE`.

## Locations:

<pre>
# Measures locations
'bioggio': 'BIO',
'brionesm': 'BRI',
'chiasso': 'CHI',
'giubiasco': 'GIU',
'locarno': 'LOC',
'mendrisio': 'MEN',
'sagno': 'SAG',
'tesserete': 'TES',
'nabel-lugano': 'LUG',
'meteosvizzera-acquarossacomprovasco': 'MS-COM',
'meteosvizzera-montegeneroso': 'MS-GEN',
'meteosvizzera-gutsch': 'MS-GUT',
'meteosvizzera-locarnomonti': 'MS-OTL',
'meteosvizzera-magadino': 'MS-MAG',
'meteosvizzera-ludianomatro': 'MS-MAT',
'meteosvizzera-lugano': 'MS-LUG',
'meteosvizzera-stabio': 'MS-STB',

# Weather forecast locations
'comprovasco': 'COM', 
'magadino': 'MAG', 
'matro': 'MTR', 
'locarno-monti': 'OTL', 
'brione-sopra-minusio': 'P_BSM', 
'locarno': 'P_LOC', 
'giubiasco'; 'TIGIU', 
'montegeneroso': 'GEN', 
'bioggio': 'P_BIO', 
'ponte_capriasca': 'P_CAP',
'sagno': 'P_SAG', 
'stabio': 'SBO'
'chiasso': 'TICIA'
</pre>

N.B. Some codes (e.g. `MTR` and `MS-MAT`) correspond to the same reak location (Matro). However, the system considers 
them as two different, and coincident, locations. The former (`MTR`) refers to meteorological forecast while the latter (`MS-MAT`)
to measures.

## Signals:

### Measures:

<pre>
"CN": Cloud coverage
"Gl": Irradiance
"NO": Nitrogen oxide
"NO2": Nitrogen dioxide 
"NOx": Nitrogen oxides
"O3": Ozone
"P": Pressure
"Prec": Rain
"RH": Relative humidity
"T": Temperature
"Tdew": Dew point temperature
"WD": Wind direction
"WDvect": Wind direction (vectorial)
"WS": Wind speed
"WSvect": Wind speed (vectorial)
"WSgust": Wind gust
</pre>

### Weather forecast:

<pre>
GLOB            W m-2           Average downward shortwave radiation flux at surface within last 3h
PS              Pa              Surface pressure (not reduced)
TOT_PREC        kg m-2          Total precipitation within last 3h
RELHUM_2M       %               2m relative humidity (with respect to water)
TD_2M           K               2m dew point temperature
T_2M            K               2m air temperature
DD_10M          degrees         10m wind direction
FF_10M          m s-1           10m wind speed
VMAX_10M        m s-1           10m wind gust
CLCT            %               Total cloud area fraction
PMSL            Pa              Surface pressure reduced to msl
</pre>

## Cases:

### Measures:

The following cases are currently available:

* Hourly values: the value acquired in an hours is aggregated performing an average
* Aggregated values: the aforementioned hourly values are aggregated considering the following cases:
  * `24h`: aggregation of the values acquired in the period `[now-24h:now]` (MOR: now=07:00; EVE: now=19:00) 
  * `48h`: aggregation of the values acquired in the period `[now-48h:now-24h]` 
  * `72h`: aggregation of the values acquired in the period `[now-72h:now-48h]` 

### Weather forecast:

The following cases are currently available:

* `step`: where teh forecast of a single hourly forecast is taken into account
* `chunk`: where an aggregation is performed considering a set of hourly values. The chunk is defined below:

<pre>
'chunk1': {'start': 1, 'end': 8},
'chunk2': {'start': 9, 'end': 17},
'chunk3': {'start': 18, 'end': 25},
'chunk4': {'start': 26, 'end': 33},
</pre>

`chunk4` means that it considers the hourly steps between 26 and 33.

## Aggregation functions:

Currently, `$FUNCTION` can have one of the following values: `mean`, `max`, `min` 


## Examples:

### Measures:

* Hourly values (`CHI__O3__m10`): the average of O3 acquired values in Chiasso in the period 20:00-20:59 of the day before 
  * `MOR`: (`m0` refers to period 20:00-20:59 UTC (06:00 - 10 h = 20:00 day before))
  * `EVE`: (`m0` refers to period 08:00-08:59 UTC (18:00 - 10 h = 08:00 of the same day)
* Aggregation (`MEN__NO2__24h__mean`): the NO2 average of the hourly values of the last 24 hours (e.g. in a MOR case [`'2022-05-31T07:00:00Z':'2022-05-31T07:00:00Z'`])

### Weather forecast:

* `step` case (`GEN__T_2M__step10`):
  * `MOR`: the forecast of `GEN__T_2M` signal related to 13:00 (`MOR` models consider the last Meteosvizzera running at 03:00 -> 3 + 10 = 13)
  * `EVE`: the forecast of `GEN__T_2M` signal related to 22:00 (`EVE` models consider the last Meteosvizzera running at 12:00 -> 12 + 10 = 22)

* `chunk` case (`P_CAP__T_2M__chunk2__mean`):
  * `MOR`: the average forecast of `P_CAP__T_2M` signals related to period 12:00-17:59
  


