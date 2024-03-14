# Utsav Anantbhat


import os
import pathlib
import sys
import numpy as np
import pandas as pd
from xml.etree import ElementTree as eltree


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    xmlns = 'http://www.topografix.com/GPX/1/0'

    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', f'{pt["lat"]:.10f}')
        trkpt.setAttribute('lon', f'{pt["lon"]:.10f}')
        time = doc.createElement('time')
        time.appendChild(doc.createTextNode(pt['timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")))
        trkpt.appendChild(time)
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    doc.documentElement.setAttribute('xmlns', xmlns)

    with open(output_filename, 'w') as fh:
        fh.write(doc.toprettyxml(indent='  '))


# Implement the haversine function; sources/references: https://devpress.csdn.net/python/62fdab7fc677032930804307.html, 
# https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows,
# https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas/29546836#29546836
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) #convert angles from degrees to radians for the 4 values
    diff_lat = lat2 - lat1
    diff_lon = lon2 - lon1 #latitude and longitude difference: seconds point - first point
    hsine = np.sin(diff_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(diff_lon / 2) ** 2 #haversine formula
    c = 2 * np.arcsin(np.sqrt(hsine))
    meters = 6371000 * c #radius of Earth in meters: 6,371,000
    return meters


def distance(data):

    # Calculate the total distance traveled given lat and lon coordinates
    data['lat1'] = data['lat'].shift(-1)
    data['lon1'] = data['lon'].shift(-1)
    data['ds'] = data.apply(lambda x: haversine(x['lat'], x['lon'], x['lat1'], x['lon1']), axis=1)
    total_distance = np.sum(data['ds'][:-1]) #total distance traveled in meters
    return total_distance 



def get_data(input_gpx):

    # Parse the GPX data from the input file and return a DataFrame with timestamp, latitude, and longitude.
    data_tree = eltree.parse(input_gpx)
    data_frame = pd.DataFrame(columns=['timestamp', 'lat', 'lon'])

    for pos, time in zip(data_tree.iter('{http://www.topografix.com/GPX/1/0}trkpt'),
                         data_tree.iter('{http://www.topografix.com/GPX/1/0}time')):
        data_in = {'lat': pd.to_numeric(pos.attrib['lat']), 
                     'lon': pd.to_numeric(pos.attrib['lon']),
                     'timestamp': pd.to_datetime(time.text, utc=True, format='mixed')}
        data_entry = pd.DataFrame(data_in, index=[0])
        data_frame = pd.concat([data_frame, data_entry], ignore_index=True)
    return data_frame


def correlate_data(accl_data, gps_data, phone_data):
   
    # Process accelerometer, GPS, and phone data to find the best time offset and create a DataFrame with correlated data.
    offset_vals = np.linspace(-5.0, 5.0, 101) #initialize offset value linespace
    cross_correlation_vals = [] #initialize the cross correlation values list/array

    first_time = accl_data['timestamp'].min() #accelerometer data timestamp

    for offset in offset_vals:
    
        accl = accl_data.copy() #rename and create copies of the lists for convenience
        gps = gps_data.copy()
        phone = phone_data.copy()

        phone['timestamp'] = first_time + pd.to_timedelta(phone['time'] + offset, unit='s') #phone data starts at exactly the same time as the accelerometer data

        """
        # Grouping using pandas Grouper method -> not effective here; wrong output
        accl = accl.groupby(pd.Grouper(key='timestamp', freq='4S')).mean()
        phone = phone.groupby(pd.Grouper(key='timestamp', freq='4S')).mean()
        gps = gps.groupby(pd.Grouper(key='timestamp', freq='4S')).mean()
        """

        accl['timestamp'] = accl['timestamp'].dt.round('4S') #round the data to 4 seconds...
        accl = accl.groupby(['timestamp']).mean() #..., group the 4 second bins and take the mean of each

        gps['timestamp'] = gps['timestamp'].dt.round('4S')
        gps = gps.groupby(['timestamp']).mean()
        
        phone['timestamp'] = phone['timestamp'].dt.round('4S')
        phone = phone.groupby(['timestamp']).mean()

        # Merging the data
        merge_data = pd.merge(phone, gps, "inner", left_on=phone.index, right_on=gps.index) 
        merge_data = merge_data.rename(columns={'key_0':'timestamp'}).set_index(['timestamp'])
        merge_data = pd.merge(merge_data, accl, "inner", left_on=merge_data.index, right_on=accl.index)
        merge_data = merge_data.rename(columns={'key_0':'timestamp'}).set_index(['timestamp'])
        
        dot_product = merge_data['gFx'] * merge_data['x'] #calculate the dot product
        cross_correlation = dot_product.sum() #sum the results
        cross_correlation_vals.append(cross_correlation) #append the values to the list

    best_offset = offset_vals[np.argmax(cross_correlation_vals)] #get the best time offset
    
    return merge_data, best_offset



def main():
    input_directory = pathlib.Path(sys.argv[1])
    output_directory = pathlib.Path(sys.argv[2])

    accl = pd.read_json(input_directory / 'accl.ndjson.gz', lines=True, convert_dates=['timestamp'])[['timestamp', 'x', 'y']]
    gps = get_data(input_directory / 'gopro.gpx')
    phone = pd.read_csv(input_directory / 'phone.csv.gz')[['time', 'gFx', 'Bx', 'By']]

    combined, best_offset = correlate_data(accl, gps, phone)
    combined.reset_index(inplace=True)

    print(f'Best time offset: {best_offset:.1f}')
    os.makedirs(output_directory, exist_ok=True)
    output_gpx(combined[['timestamp', 'lat', 'lon']], output_directory / 'walk.gpx')
    combined[['timestamp', 'Bx', 'By']].to_csv(output_directory / 'walk.csv', index=False)


main()