# Utsav Anantbhat


import sys
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from xml.dom.minidom import parse
from xml.etree import ElementTree as ET


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.7f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.7f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')


"""
# Earlier implementation of reading the files

def in_gpx(input_gpx, input_csv):
    data = parse(input_gpx)
    trkpt = data.getElementsByTagName("trkpt")
    lat = []
    lon = []
    for i in trkpt:
        lat.append(i.getAttribute('lat'))
        lon.append(i.getAttribute('lon'))
    points = pd.DataFrame()
    points['lat'] = lat
    points['lat'] = points['lat'].astype(float)
    points['lon'] = lon
    points['lon'] = points['lon'].astype(float)

    input_csv = pd.read_csv(input_csv, parse_dates=['datetime'])
    points['Bx'] = input_csv['Bx']
    points['By'] = input_csv['By']
    points = points.set_index('datetime')
    return points

"""

# Implement the haversine function; sources/references: https://devpress.csdn.net/python/62fdab7fc677032930804307.html, 
# https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2]) #convert angles from degrees to radians for the 4 values
    diff_lat = lat2 - lat1
    diff_lon = lon2 - lon1 #latitude and longitude difference: seconds point - first point
    hsine = np.sin(diff_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(diff_lon / 2) ** 2 #haversine formula
    c = 2 * np.arcsin(np.sqrt(hsine))
    meters = 6371000 * c #radius of Earth in meters: 6,371,000
    return meters

def get_data(input_gpx, input_csv):

    tree = ET.parse(input_gpx) #parse GPX data
    data = pd.DataFrame(columns=['lat', 'lon', 'datetime'])

    for position, time in zip(tree.iter('{http://www.topografix.com/GPX/1/0}trkpt'), #append position and time from GPX file
                              tree.iter('{http://www.topografix.com/GPX/1/0}time')):
        data_entry = {'lat': position.attrib['lat'], 'lon': position.attrib['lon'], 'datetime': time.text}
        entry = pd.DataFrame(data_entry, index=[0])
        data = pd.concat([data, entry], ignore_index=True)

    # Convert from object dtype to numeric and datetime
    data['lat'] = pd.to_numeric(data['lat'])
    data['lon'] = pd.to_numeric(data['lon'])
    data['datetime'] = pd.to_datetime(data['datetime'])

    data_csv = pd.read_csv(input_csv, parse_dates=['datetime']) #read CSV file

    # Combine GPX and CSV data
    data['Bx'] = data_csv['Bx']
    data['By'] = data_csv['By']

    data = data.set_index('datetime')

    return data


def distance(data):

    # Calculate the total distance traveled given lat and lon coordinates
    data['lat1'] = data['lat'].shift(-1)
    data['lon1'] = data['lon'].shift(-1)
    data['ds'] = data.apply(lambda x: haversine(x['lat'], x['lon'], x['lat1'], x['lon1']), axis=1)
    total_distance = np.sum(data['ds'][:-1]) #total distance traveled in meters
    return total_distance 


def smooth(input_data):
    data = input_data

    initial_state = data.iloc[0][['lat', 'lon', 'Bx', 'By']]
    observation_covariance = np.diag([0.5, 0.5, 0.79, 0.79]) ** 2
    transition_covariance = np.diag([0.53, 0.53, 0.79, 0.79]) ** 2
    transition = np.array([[1, 0, 5*10**-7, 34*10**-7],
                           [0, 1, -49*10**-7, 9*10**-7],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )

    smoothed_data = kf.smooth(data[['lat', 'lon', 'Bx', 'By']])[0]
    smoothed_data = pd.DataFrame(smoothed_data, columns=['lat', 'lon', 'Bx', 'By'])
    smoothed_data['datetime'] = data.reset_index()['datetime']
    smoothed_data.set_index('datetime', inplace=True)

    return smoothed_data


def main():
    input_gpx = sys.argv[1]
    input_csv = sys.argv[2]

    points = get_data(input_gpx, input_csv)

    dist = distance(points)
    print(f'Unfiltered distance: {dist:.2f}')

    smoothed_points = smooth(points)
    smooth_dist = distance(smoothed_points)
    print(f'Filtered distance: {smooth_dist:.2f}')

    output_gpx(smoothed_points, 'out.gpx')


if __name__ == '__main__':
    main()
