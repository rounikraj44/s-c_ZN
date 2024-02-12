import pandas as pd
import numpy as np
import datetime
import re
import json
import os
import folium
from folium.plugins import PolyLineTextPath
import geopy
import time

from datetime import datetime, timedelta
from geographiclib.geodesic import Geodesic
from webdriver_manager.chrome import ChromeDriverManager
from geopy.distance import geodesic

import math
import warnings

warnings.filterwarnings("ignore")
from shapely.geometry import Point, LineString, MultiLineString, MultiPoint
from pyproj import Transformer
from shapely.ops import transform

import requests

import matplotlib.pyplot as plt

import matplotlib.font_manager as font_manager
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Arial', size=12)
plt.rcParams["figure.figsize"] = [16, 11]

import PIL
from PIL import Image, ImageFont, ImageDraw


from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService

from matplotlib import pyplot as plt, dates as mdates

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as rImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, LongTable, TableStyle, Spacer, PageBreak,ListFlowable, ListItem, Frame, KeepInFrame

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import psycopg2

DATABASE_NAME = "speed_consumption_instance1"
USER_NAME = "postgres"
PASSWORD = "54321Post"
DB_URL = "speed-consumption-instance1.cxstuhubx9sn.us-east-1.rds.amazonaws.com"
DB_PORT = '5432'


def trigger_pdf(filename_inp, from_port_inp,to_port_inp,prepared_basis_inp,\
                          voyage_phase_inp,fuel_type_used_inp,waranted_weather_yes_no_inp,bf_limit_inp,\
                          windwave_limit_inp,swell_height_inp,swh_limit_inp,gwx_type_inp,not_sure_L78_inp,\
                          gwx_hours_inp,performance_calculation_inp,current_tolerance_inp,tolerance_inp,mistolerance_inp,\
                          About_Cons_MaxTolerence_inp,extrapolation_Allowed_inp,report_type_inp):
    print("is this working")
    ENVIRONMENT="LOCAL"
    if("ENVIRONMENT" in os.environ):
        ENVIRONMENT = os.environ["ENVIRONMENT"]

    def connect_to_db():
        # Establishing the connection
        conn = psycopg2.connect(
            database=DATABASE_NAME,
            user=USER_NAME, password=PASSWORD,
            host=DB_URL,
            port=DB_PORT
        )
        # Setting auto commit false
        conn.autocommit = True

        # Creating a cursor object using the cursor() method
        cursor = conn.cursor()
        return cursor


    '''READING THE INPUT FILE'''

    #filename = '5-Voutakos Performance Report.xlsx'  # single, working, executve all code from library
    filename=filename_inp
    raw_df = pd.read_excel(filename)  # ,keep_default_na=False)
    # print(raw_df.T)

    numeric_series_40 = pd.to_numeric(raw_df.iloc[40, 2:], errors='coerce')
    numeric_series_41 = pd.to_numeric(raw_df.iloc[41, 2:], errors='coerce')
    numeric_series_42 = pd.to_numeric(raw_df.iloc[42, 2:], errors='coerce')
    numeric_series_43 = pd.to_numeric(raw_df.iloc[43, 2:], errors='coerce')
    numeric_series_44 = pd.to_numeric(raw_df.iloc[44, 2:], errors='coerce')
    numeric_series_48 = pd.to_numeric(raw_df.iloc[48, 2:], errors='coerce')
    numeric_series_51 = pd.to_numeric(raw_df.iloc[51, 2:], errors='coerce')
    numeric_series_52 = pd.to_numeric(raw_df.iloc[52, 2:], errors='coerce')

    raw_df.iloc[40, 2:] = numeric_series_40
    raw_df.iloc[41, 2:] = numeric_series_41
    raw_df.iloc[42, 2:] = numeric_series_42
    raw_df.iloc[43, 2:] = numeric_series_43
    raw_df.iloc[44, 2:] = numeric_series_44
    raw_df.iloc[48, 2:] = numeric_series_48
    raw_df.iloc[51, 2:] = numeric_series_51
    raw_df.iloc[52, 2:] = numeric_series_52

    # print(raw_df)
    class Preprocess:
        '''preprocessing the input file'''

        def __init__(self, raw_df):
            self.df = raw_df
            # print(self.df)

        def process_file(self):
            self.df.dropna(axis=1, how='all', inplace=True)
            # print(self.df)
            filter_from_row = raw_df[raw_df['Unnamed: 0'] == 'VESSEL'].index[0]
            self.df = self.df.loc[filter_from_row:, :].drop(['Unnamed: 0'], axis=1).T
            self.df = self.df.rename(columns=self.df.iloc[0]).drop(self.df.index[0])
            self.df.reset_index(drop=True, inplace=True)
            self.df.fillna(0, inplace=True)

            # print("first process",self.df)
            # print(self.df.info())

            print('Before to_datetime')
            # print(self.df["Date"],self.df["Ship's Time (UTC)"])

            print('After to_datetime')
            self.df["Date"] = pd.to_datetime(self.df["Date"], errors='coerce')
            self.df["Ship's Time (UTC)"] = pd.to_datetime(self.df["Ship's Time (UTC)"], errors='coerce')

            # print(type(self.df['Date']),self.df['Date'])
            # print(type(self.df["Ship's Time (UTC)"]),self.df["Ship's Time (UTC)"])

            print('before dt accessor')

            # print(self.df["Date"],self.df["Ship's Time (UTC)"])

            # print('??????')
            # print(self.df.info())

            # print('what data type')
            # print(self.df["Ship's Time (UTC)"].dtype.name)
            # print("is this the one",self.df)

            # This is orginal code
            #         if self.df["Ship's Time (UTC)"].dtype.name=='datetime64[ns]':

            #             self.df["Steaming Time (Hrs)"]=self.df["Ship's Time (UTC)"].diff()
            #             self.df['Steaming Time (Hrs)']=[res.total_seconds()/86400*24 for res in self.df["Steaming Time (Hrs)"]]
            #             self.df["Ship's Time (UTC)"]=self.df["Ship's Time (UTC)"]

            #         else:
            #             print('not the expected datatype')

            # This is original code

            print("is this the 4")

            # new altered code

            # ---original start

            #         for i in range(len(self.df)-1):

            #             if self.df['Steaming Time (Hrs)'][i]!=0:
            #                 self.df['Steaming Time (Hrs)'][i]=self.df["Ship's Time (UTC)"][i]- self.df["Ship's Time (UTC)"][i-1]
            #                 print("not zero")
            #                 print(self.df["Ship's Time (UTC)"][i],self.df["Ship's Time (UTC)"][i-1])

            #                 self.df['Steaming Time (Hrs)'][i]=self.df['Steaming Time (Hrs)'][i].total_seconds()/86400*24
            #                 print(self.df['Steaming Time (Hrs)'][i])

            #             else:
            #                 self.df['Steaming Time (Hrs)'][i]=self.df["Ship's Time (UTC)"][i]-self.df["Ship's Time (UTC)"][i]
            #                 print("zero")
            #                 print(self.df["Ship's Time (UTC)"][i])
            #                 self.df['Steaming Time (Hrs)'][i]=self.df['Steaming Time (Hrs)'][i].total_seconds()/86400*24
            #                 print(self.df['Steaming Time (Hrs)'][i])
            # ----original-end

            if self.df['Steaming Time (Hrs)'].sum() == 0:

                self.df['Steaming Time (Hrs)'][0] = 0

                for i in range(1, len(self.df)):
                    self.df['Steaming Time (Hrs)'][i] = self.df["Ship's Time (UTC)"][i] - self.df["Ship's Time (UTC)"][
                        i - 1]
                    print("All zero")
                    # print(self.df["Ship's Time (UTC)"][i],self.df["Ship's Time (UTC)"][i-1])

                    self.df['Steaming Time (Hrs)'][i] = self.df['Steaming Time (Hrs)'][i].total_seconds() / 86400 * 24
                    print(self.df['Steaming Time (Hrs)'][i])

            else:

                for i in range(len(self.df) - 1):

                    if self.df['Steaming Time (Hrs)'][i] != 0:
                        # self.df['Steaming Time (Hrs)'][i]=self.df["Ship's Time (UTC)"][i]-self.df["Ship's Time (UTC)"][i-1]
                        self.df['Steaming Time (Hrs)'][i] = self.df["Steaming Time (Hrs)"][i]
                        print("not zero")
                        # print(self.df["Ship's Time (UTC)"][i],self.df["Ship's Time (UTC)"][i-1])

                        # self.df['Steaming Time (Hrs)'][i]=self.df['Steaming Time (Hrs)'][i].total_seconds()/86400*24
                        print(self.df['Steaming Time (Hrs)'][i])

                    else:
                        self.df['Steaming Time (Hrs)'][i] = self.df["Ship's Time (UTC)"][i] - self.df["Ship's Time (UTC)"][
                            i]
                        print("zero")
                        print(self.df["Ship's Time (UTC)"][i])
                        self.df['Steaming Time (Hrs)'][i] = self.df['Steaming Time (Hrs)'][i].total_seconds() / 86400 * 24
                        print(self.df['Steaming Time (Hrs)'][i])

            # print(self.df['Steaming Time (Hrs)'])

            # new altered code

            self.df["Date"] = self.df["Date"].dt.date
            self.df["Ship's Time (UTC)"] = self.df["Ship's Time (UTC)"].dt.time

            print('after dt accessor')
            # print(self.df["Date"],self.df["Ship's Time (UTC)"])

            #         try:
            #             self.df['Steaming Time (Hrs)']=self.df['Steaming Time (Hrs)'].diff()
            #             self.df['Steaming Time (Hrs)']=[res.total_seconds()/86400*24 for res in self.df['Steaming Time (Hrs)']]
            #         except Exception as e:
            #             self.df['Steaming Time (Hrs)']=self.df['Steaming Time (Hrs)'].diff()

            self.df['Latitude'] = self.df['Latitude'].apply(str)
            self.df['Longitude'] = self.df['Longitude'].apply(str)

            self.df['Latitude'] = '"""' + self.df['Latitude'] + '"""'
            self.df['Longitude'] = '"""' + self.df['Longitude'] + '"""'

            # print(self.df.iloc[:,36:56])

            # print(self.df.info())
            # print(self.df)
            self.df.iloc[:, 36:56] = self.df.iloc[:, 36:56].replace('-', '0')
            self.df.iloc[:, 36:56] = self.df.iloc[:, 36:56].astype(float)

            self.dtn_columns = ['Date', "Ship's Time (UTC)", 'Latitude', 'Longitude',
                                'Observed distance (NM)', 'Steaming Time (Hrs)', 'Average speed (Kts)', 'Course']

            # print(self.df['Longitude'])

            #         pattern_check=r"\d+\.\d+"
            #         pattern={"""(?P<deg>\d+)°(?P<min>\d+)\'(?P<sec>\d+)\"(?P<dir>[a-zA-Z])""",
            #                  """(?P<deg>\d+)\s+(?P<min>\d+)\'\s+(?P<sec>\d+)\"\s+(?P<dir>[a-zA-Z])""",
            #                 """(?P<deg>\d+)\s+(?P<min>\d+)\s+(?P<dir>[a-zA-Z])"""}

            return self.df
        def obs_distance_calculation(self, processed_file):

            self.distance_to_go = processed_file['Distance to go (NM)']
            # print(type(processed_file['Distance to go (NM)']))
            self.Observed_distance = processed_file['Observed distance (NM)']
            # print(type(processed_file['Observed distance (NM)']))

            observed = []
            self.Observed_distance = self.Observed_distance
            for i in np.arange(1, len(self.Observed_distance)):
                if math.isnan(self.Observed_distance[i]):
                    self.Observed_distance[i] = (self.distance_to_go[i - 1] - self.distance_to_go[i])
                else:
                    self.Observed_distance[i] = self.Observed_distance[i]
            return processed_file

        def distance_to_go_calculation(self, processed_file):

            self.distance_to_go = processed_file['Distance to go (NM)']
            self.Observed_distance = processed_file['Observed distance (NM)']
            self.total_distance = self.Observed_distance.sum()

            if self.distance_to_go[0] == 0:
                self.distance_to_go[0] = self.total_distance

            for i in np.arange(1, len(self.distance_to_go)):
                self.distance = self.distance_to_go
                if self.distance[i] == 0:
                    self.distance[i] = (self.distance[i - 1] - self.Observed_distance[i])
                else:
                    self.distance[i] = self.distance[i]
            return processed_file

        def average_speed_calculation(self, processed_file):

            self.steaming_hours = processed_file['Steaming Time (Hrs)']
            self.obs_distance = processed_file['Observed distance (NM)']

            self.avg_speed = processed_file['Average speed (Kts)']
            for i in np.arange(len(self.avg_speed)):
                if math.isnan(self.avg_speed[i]):
                    self.avg_speed[i] = self.obs_distance[i] / self.steaming_hours[i]

            return processed_file

        def fuel_calculation(self, processed_file):

            self.processed_file = processed_file
            for i in range(1, len(processed_file)):
                # print(i)
                if self.processed_file.iloc[:, 41][i] == 0:
                    self.processed_file.iloc[:, 41][i] = self.processed_file.iloc[:, 40][i - 1] - \
                                                         self.processed_file.iloc[:, 40][i]

                if self.processed_file.iloc[:, 46][i] == 0:
                    self.processed_file.iloc[:, 46][i] = self.processed_file.iloc[:, 45][i - 1] - \
                                                         self.processed_file.iloc[:, 45][i]

                if self.processed_file.iloc[:, 51][i] == 0:
                    self.processed_file.iloc[:, 51][i] = self.processed_file.iloc[:, 50][i - 1] - \
                                                         self.processed_file.iloc[:, 50][i]

                if self.processed_file.iloc[:, 56][i] == 0:
                    self.processed_file.iloc[:, 56][i] = self.processed_file.iloc[:, 55][i - 1] - \
                                                         self.processed_file.iloc[:, 55][i]

            return self.processed_file

        def dms_decimal_calculation(self, processed_file):

            pattern_check = '^"""(-\d+|\d+)\.\d+"""$'
            pattern = {"""(?P<deg>\d+)°(?P<min>\d+)\(?P<sec>\d+)\(?P<dir>[a-zA-Z])""": [],
                       """(?P<deg>\d+)\s+(?P<min>\d+)\s+(?P<sec>\d+)\s+(?P<dir>[a-zA-Z])""": [],
                       """(?P<deg>\d+)\s+(?P<min>\d+)\s+(?P<dir>[a-zA-Z])""": [],
                       """(?P<deg>\d+)\s+(?P<min>\d+)(?P<dir>[a-zA-Z])""": [],
                       """(?P<deg>\d+)\.(?P<min>\d+)\s+(?P<dir>[a-zA-Z])""": [],
                       """(?P<deg>\d+)\.(?P<min>\d+)(?P<dir>[a-zA-Z])""": [],
                       """(?P<deg>\d+)\s+(?P<min>\d+)\'\s+(?P<dir>[a-zA-Z])""": []
                       }

            self.validation = pattern_check
            self.check_pattern = pattern

            self.lat_lng = processed_file[['Latitude', 'Longitude']]
            self.no_of_columns = len(self.lat_lng.columns)

            for k in range(self.no_of_columns):

                # print("...."*50)
                # print(self.lat_lng.iloc[:,k])
                for j in range(len(self.lat_lng.iloc[:, k])):
                    i = self.lat_lng.iloc[:, k][j]

                    match = re.search(self.validation, str(i))

                    if match:
                        print("match found, dont process,keep orginal", i)

                        print(self.check_pattern)
                        print(match.group())
                        print(match.group().strip('"'))

                        # self.lat_lng.iloc[:,k][j]=self.lat_lng.iloc[:,k][j].strip()
                        self.lat_lng.iloc[:, k][j] = match.group().strip('"')


                    else:

                        for key in self.check_pattern:

                            print(key)

                            dms_deg = 0
                            dms_min = 0
                            dms_sec = 0
                            dms_dir = ''

                            try:
                                match = re.search(key, i)
                            except Exception as e:
                                match = re.search(key, str(i))

                            if match:

                                print("match found for decimal conversion", i)
                                pattern = key  # matching key found from dictionary of patterns
                                # print(pattern)

                                try:
                                    dms_deg = match.group('deg')
                                    dms_min = match.group('min')
                                    dms_sec = match.group('sec')
                                    dms_dir = match.group('dir')
                                except Exception as e:
                                    dms_deg = match.group('deg')
                                    dms_min = match.group('min')
                                    dms_sec = 0
                                    dms_dir = match.group('dir')

                                decimal_coordinate = (float(dms_deg) + float(dms_min) / 60 + float(dms_sec) / (60 * 60)) * (
                                    -1 if dms_dir in ['W', 'S'] else 1)

                                i = decimal_coordinate
                                self.lat_lng.iloc[:, k][j] = i
                                # print(self.lat_lng.iloc[:,k][j])

                            else:
                                print('No existing key matches, impute new pattern key', i)

            processed_file['Latitude'] = self.lat_lng['Latitude']
            processed_file['Longitude'] = self.lat_lng['Longitude']

            # print("second file",processed_file)

            return processed_file

        def add_missing_columns(self, processed_file):

            self.processed_file = processed_file
            missing_columns_list = ['True Wind Force (BF)_forc_wx', 'True Wind Force (KNOTS)_forc_wx',
                                    'True Wind Force Direction_forc_wx',
                                    'Significant Wave Height_forc_wx', 'Swell height (m)_forc_wx',
                                    'Swell  Direction_forc_wx',
                                    'Wind Sea Height (m)_forc_wx', 'Wind Sea Height Direction_forc_wx',
                                    'Sea Currents (kts)_forc_wx',
                                    'Sea Currents Direction_forc_wx']

            for i in missing_columns_list:
                if i not in self.processed_file:
                    self.processed_file[i] = 0

            print('immediately after adding missing columns')

            # print(self.processed_file)

            return self.processed_file

        def DTN_calculation(self, processed_file):
            self.dtn = processed_file
            # print(self.dtn)
            print("--" * 100)
            column_names = ["Date", "Ship's Time (UTC)", "Latitude", "Longitude", "Course",
                            "Distance to go (NM)", "Observed distance (NM)", "Steaming Time (Hrs)", "Average speed (Kts)"]
            # additional_columns=['A','B','C']
            additional_columns = [""
                                  "swellDirection",
                                  "swellHeight",
                                  "waveDirection",
                                  "waveHeight",
                                  "windDirection",
                                  "Wind Force(BF Scale)",
                                  "windSpeed",
                                  "airTemperature",
                                  "waterTemperature",
                                  "pressure",
                                  "precipitation",
                                  "visibility",
                                  "windWaveDirection",
                                  "windWaveHeight",
                                  "currentSpeed",
                                  "currentDirection"]

            dtn_input = self.dtn[column_names]

            dtn_input.rename(columns={"Date": "Date", "Course": "Average Ships Course since Last Report",
                                      "Observed distance (NM)": "Steaming Distance Since Last Noon Report",
                                      "Steaming Time (Hrs)": "Steaming Hours Since Last noon Report",
                                      "Average speed (Kts)": "Average Speed Since Last noon Report",
                                      "Ship's Time (UTC)": 'Time ( UTC)'}, inplace=True)
            # print(dtn_input.info())

            # dtn_input['Date']=dtn_input['Date'].dt.date
            # dtn_input['Time (UTC)']=dtn_input["Time (UTC)"].astype('datetime64')
            # dtn_input['Time (UTC)']=dtn_input['Time (UTC)'].dt.time

            # print("$$$$"*50)
            # print(dtn_input.info())

            re_order_list = ['Date', 'Time ( UTC)', 'Latitude', 'Longitude', 'Steaming Distance Since Last Noon Report',
                             'Steaming Hours Since Last noon Report', 'Average Speed Since Last noon Report',
                             'Average Ships Course since Last Report']

            dtn_input = dtn_input[re_order_list]

            dtn_input[[additional_columns]] = ''  # appending aditional columns

            dtn_input = dtn_input.T
            print('9999' * 10, dtn_input)

            length_of_columns = len(dtn_input.columns)
            print(length_of_columns)

            dict = {}
            for i in range(length_of_columns):

                print('iterate', i)

                if i == length_of_columns - length_of_columns:
                    dict[i] = 'Arrival (EOSP)'

                elif i == length_of_columns - 1:
                    dict[i] = 'Departure (COSP)'
                else:
                    dict[i] = 'Noon'

            dtn_input.rename(columns=dict, inplace=True)

            # print(dtn_input)
            temp = dtn_input.iloc[:, -1]
            dtn_input.drop(dtn_input.columns[len(dtn_input.columns) - 1], axis=1, inplace=True)

            # swapping departure column from last position to 2nd position
            new_col = temp

            dtn_input.insert(loc=1, column='Departure (COSP)', value=temp)

            print("dtn_input.columns", dtn_input.columns)
            # data_input.iloc[5,:]=''

            return dtn_input


    ### code from existing program


    preprocess = Preprocess(raw_df)
    processed_file = preprocess.process_file()

    # print(processed_file)

    processed_file = preprocess.obs_distance_calculation(processed_file)
    processed_file = preprocess.distance_to_go_calculation(processed_file)
    # print(processed_file)


    processed_file = preprocess.average_speed_calculation(processed_file)
    processed_file = preprocess.fuel_calculation(processed_file)
    processed_file = preprocess.dms_decimal_calculation(processed_file)  # uptill this point we have processed file

    print(
        'The below is the processed file where i am adding the rows below remarks coloumn which is not available in input file')
    processed_file_add_missing_columns = preprocess.add_missing_columns(
        processed_file)  # here i am adding the missing columsn
    print("is tihs the processed file")
    # print(processed_file_add_missing_columns)
    # print(processed_file.T)
    # processed_file.T.to_csv('output.csv')


    DTN = preprocess.DTN_calculation(
        processed_file_add_missing_columns)  # This is a standalone dtn file and will not affect proceessed_file
    # print(filename)
    # DTN.to_excel(filename.split(".")[0]+".xlsx")
    DTN.to_excel("./dtn_output/" + filename)


    def cut(line, distance, lines):
        # Cuts a line in several segments at a distance from its starting point
        if distance <= 0.0 or distance >= line.length:
            return [LineString(line)]
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))
            if pd == distance:
                return [
                    LineString(coords[:i + 1]),
                    LineString(coords[i:])
                ]
            if pd > distance:
                cp = line.interpolate(distance)
                lines.append(LineString(coords[:i] + [(cp.x, cp.y)]))
                line = LineString([(cp.x, cp.y)] + coords[i:])
                if line.length > distance:
                    cut(line, distance, lines)
                else:
                    lines.append(LineString([(cp.x, cp.y)] + coords[i:]))
                return lines


    def generate_route_array(df2):
        # print(df2)
        len(df2)
        route_arr = []
        speed_arr = []
        co_xy_arr = []

        for i in range(len(df2) - 1):
            coordinates = str(df2.iloc[i, 3]) + "," + str(df2.iloc[i, 2]) + ";" + str(df2.iloc[i + 1, 3]) + "," + str(
                df2.iloc[i + 1, 2])
            # coordinates=format(df2.iloc[i,3],'.10f')+","+format(df2.iloc[i,2],'.10f')+";"+format(df2.iloc[i+1,3],'.10f')+","+format(df2.iloc[i+1,2],'.10f')
            co_xy_arr.append(coordinates)
            print(co_xy_arr)
            speed_arr.append(df2.iloc[i + 1, 6])
            # print(route_arr)

        for i in range(len(co_xy_arr)):
            query_string = co_xy_arr[i]
            cursor = connect_to_db()
            print("connect to db successfull")
            cursor.execute("SELECT * from speed_consumption1.sea_routes where route_id=" + "'" + str(query_string) + "'")
            myresult = cursor.fetchall()

            # print(myresult)

            # Need to convert to some other format ..

            if myresult != []:
                route_arr.append(myresult[0][1])
            else:
                url = "https://api.searoutes.com/route/v2/sea/"
                item = query_string
                speed = speed_arr[i]
                print(item, speed)
                final_url = url + item + "/plan?continuousCoordinates=false&allowIceAreas=false&avoidHRA=false&avoidSeca=false&speed=" + str(
                    speed)
                headers = {
                    "accept": "application/json",
                    "x-api-key": api_key
                }
                response = requests.get(final_url, headers=headers)
                # print(response.text)
                route_json = json.loads(response.text)
                # print(route_json)
                route_arr.append(route_json)

                def log_details():

                    lst = []
                    values = [filename, datetime.utcnow(), 'SR', 1]
                    lst.append(values)

                    df = pd.DataFrame(lst)
                    df = df.to_csv(header=None, index=False)
                    f = open("/home/ubuntu/speed_consumption/notebooks/Forecast/results/demofile3.txt", "a")
                    f.write(df)
                    f.close()

                #log_details()

                start_lat = route_json['features'][0]['geometry']['coordinates'][0][0][0]
                start_lng = route_json['features'][0]['geometry']['coordinates'][0][0][1]
                end_lat = route_json['features'][0]['geometry']['coordinates'][0][-1][0]
                end_lng = route_json['features'][0]['geometry']['coordinates'][0][-1][1]

                temp_route_id = str(start_lat) + "," + str(start_lng) + ";" + str(end_lat) + "," + str(end_lng)

                route_id = temp_route_id
                route = json.dumps(route_json)

                voyage_query_insert_str = '''INSERT INTO speed_consumption1.sea_routes(route_id,route) VALUES ('''
                voyage_query = voyage_query_insert_str + "'" + route_id + "',"
                voyage_query += "'" + route + "')"

                cursor.execute(voyage_query)

        print('Route Array Below')
        # print(route_arr)

        return route_arr, speed_arr


    def generate_sub_segments(route_arr, speed_arr, df2):
        multi_line_arr = []
        line_str_arr = []
        orig_time_arr = []

        #     print("***"*100)
        #     print(route_arr)

        #     print("^"*100)
        #     print(speed_arr)

        for i in range(len(route_arr)):
            route_json = route_arr[i]
            speed_item = speed_arr[i]
            route_json_geo = route_json
            print("<>" * 100)
            # print(route_json_geo)
            # route_json_geo=json.loads(route_json_geo)
            # print(type(route_json_geo)) # this is a string
            try:
                mls_temp = MultiLineString(route_json_geo["features"][0]["geometry"]["coordinates"])

            except Exception as e:
                route_json_geo = json.loads(route_json_geo)
                mls_temp = MultiLineString(route_json_geo["features"][0]["geometry"]["coordinates"])

            multi_line_arr.append(mls_temp)
            for geom in mls_temp.geoms:
                line_str_arr.append(geom)

        # transformer = Transformer.from_crs("epsg:4326", "epsg:32633",always_xy=True)
        # transformer1 = Transformer.from_crs("epsg:32633","epsg:4326",always_xy=True)
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        transformer1 = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
        timedelta_hours = 3
        i = 0
        lat_long_arr = []
        date_arr = []
        time_arr = []
        lat_arr = []
        lon_arr = []
        date_time_arr = []
        s_distance_arr = []
        s_time_arr = []
        speed_arr_res = []
        course_arr = []
        waypoint_type_arr = []
        for segment in line_str_arr:
            dtobj = datetime.strptime(df2["DateTime"].values[i], "%Y-%m-%dT%H:%M:%S")
            dtobj_future = datetime.strptime(df2["DateTime"].values[i + 1], "%Y-%m-%dT%H:%M:%S")
            time_diff = dtobj_future - dtobj
            time_diff_hours = time_diff.total_seconds() / (60 * 60)
            num_segments = round(time_diff_hours / 3)
            orig_time_arr.append(df2["DateTime"].values[i])
            # date_arr.append(datetime.strftime(dtobj,"%Y-%m-%d"))
            # time_arr.append(datetime.strftime(dtobj,"%H:%M:%S"))
            # date_str =
            line2 = transform(transformer.transform, segment)

            print("<<" * 50)

            # print(line2)

            if line2.length / 1852 == 0:
                continue

            print(">>" * 50)

            speed = speed_arr[i]

            #     if(speed==0):
            #         #print(round(line2.length/1852,2))
            #         segment_length_in_nm = round(line2.length/1852,2)/8
            #         print("Speed is 0. Hence subdiving it into 8 segments."+str(segment_length_in_nm))
            #     else:
            #         segment_length_in_nm=speed*3

            print("##" * 50)
            print("Segment:" + str(i + 1))
            print("Start Time:" + df2["DateTime"].values[i])
            print("Time Difference:" + str(time_diff_hours) + " hours")
            print("Number of Segments:" + str(num_segments))
            print("Original Segment Length:" + str(round(line2.length / 1852, 2)) + " NM.")
            print("Speed:" + str(speed) + " knots.")
            if (num_segments <= 1):
                segment_length_in_nm = line2.length / 1852
            else:
                segment_length_in_nm = (line2.length / 1852) / (num_segments)
            print("Sub Dividing Line into " + str(segment_length_in_nm) + " NM segments.")
            print("Segment Length NM:" + str(segment_length_in_nm))
            #     p = gpd.GeoSeries(segment)
            #     p.plot()
            #     plt.show()
            result = cut(line2, math.ceil(segment_length_in_nm * 1852), list())
            print("Total Number of sub segments:" + str(len(result)))
            #     for item in result:
            #         print("Item Length:"+str(item.length/1852)+" NM")
            #         if(item.length/1852<0.001):
            #             print("Removing nano segment:")
            #             result.remove(item)

            print("New Total Number of sub segments:" + str(len(result)))
            print(dtobj)
            print("##" * 50)
            j = 0
            time_added = 3
            prev_point_time = None
            temp_date = dtobj
            for item in result:
                # if(temp_date==None):
                #    temp_date = dtobj
                print("--" * 50)
                print("Sub Segment:" + str(j + 1))
                print("Time:" + datetime.strftime(temp_date, "%Y-%m-%dT%H:%M:%S"))
                print("Item Length:" + str(round(item.length / 1852, 2)) + " NM")
                line3 = transform(transformer1.transform, item)
                temp = (line3.boundary)
                start_point = list(temp.geoms)[0]  # LONG, LAT FORMAT
                end_point = list(temp.geoms)[1]
                print("Start Point:" + str(start_point.y) + "," + str(start_point.x))
                print("End Point:" + str(end_point.y) + "," + str(end_point.x))
                lat_long_arr.append((start_point.y, start_point.x))
                date_arr.append(datetime.strftime(temp_date, "%Y-%m-%d"))
                time_arr.append(datetime.strftime(temp_date, "%H:%M:%S"))
                date_time_arr.append(
                    datetime.strftime(temp_date, "%Y-%m-%d") + "T" + datetime.strftime(temp_date, "%H:%M:%S"))
                temp_geodesic = Geodesic.WGS84.Inverse(start_point.y, start_point.x, end_point.y, end_point.x)
                print(temp_geodesic)
                print(start_point.x, start_point.y, end_point.x, end_point.y)
                segment_distance = round(item.length / 1852, 2)
                segment_time = 3
                lat_arr.append(start_point.y)
                lon_arr.append(start_point.x)
                if (i == 0 and j == 0):
                    s_distance_arr.append(0)
                    s_time_arr.append(0)
                    speed_arr_res.append(0)
                    course_arr.append(0)
                else:
                    s_distance_arr.append(prev_segment_distance)
                    time_diff = temp_date - prev_segment_time
                    time_diff_hours = time_diff.total_seconds() / (60 * 60)
                    s_time_arr.append(time_diff_hours)
                    speed_arr_res.append(prev_segment_distance / time_diff_hours)
                    course_arr.append(prev_course)

                if (j == 0):
                    waypoint_type_arr.append(ORIGINAL)
                else:
                    waypoint_type_arr.append(DERIVED)
                # lat_long_arr.append((end_point.y,end_point.x))
                #         p = gpd.GeoSeries(line3)
                #         p.plot()
                #         plt.show()
                j += 1
                prev_point_time = temp_date
                prev_segment_distance = segment_distance
                prev_segment_time = temp_date
                prev_course = temp_geodesic["azi1"] + 360 if float(temp_geodesic["azi1"]) < 0 else float(
                    temp_geodesic["azi1"])
                print("Course:" + str(prev_course))
                print("--" * 50)
                temp_date = dtobj + timedelta(hours=time_added)
                time_added += timedelta_hours
                # if(temp_date>dtobj_future):
                #    temp_date = dtobj_future
            print("==" * 50)
            i += 1
        lat_arr.append(end_point.y)
        lon_arr.append(end_point.x)
        s_distance_arr.append(segment_distance)
        waypoint_type_arr.append(ORIGINAL)
        course_temp1 = temp_geodesic["azi1"] + 360 if float(temp_geodesic["azi1"]) < 0 else float(temp_geodesic["azi1"])
        course_arr.append(course_temp1)
        lat_long_arr.append((end_point.y, end_point.x))
        temp_date_final = datetime.strptime(df2["DateTime"].values[len(df2) - 1], "%Y-%m-%dT%H:%M:%S")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(temp_date_final)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        date_arr.append(datetime.strftime(temp_date_final, "%Y-%m-%d"))
        time_arr.append(datetime.strftime(temp_date_final, "%H:%M:%S"))
        date_time_arr.append(
            datetime.strftime(temp_date_final, "%Y-%m-%d") + "T" + datetime.strftime(temp_date_final, "%H:%M:%S"))
        time_diff = temp_date_final - prev_segment_time
        time_diff_hours = time_diff.total_seconds() / (60 * 60)
        s_time_arr.append(time_diff_hours)
        speed_arr_res.append(prev_segment_distance / time_diff_hours)
        return date_arr, time_arr, lat_arr, lon_arr, s_distance_arr, s_time_arr, speed_arr_res, course_arr, waypoint_type_arr


    def N_generate_sub_segments(route_arr, speed_arr, df2):
        multi_line_arr = []
        line_str_arr = []
        orig_time_arr = []
        for i in range(len(route_arr)):
            route_json = route_arr[i]
            speed_item = speed_arr[i]
            route_json_geo = route_json
            print(route_json_geo)
            # route_json_geo=json.loads(route_json_geo)
            try:
                mls_temp = MultiLineString(route_json_geo["features"][0]["geometry"]["coordinates"])

            except Exception as e:
                route_json_geo = json.loads(route_json_geo)
                mls_temp = MultiLineString(route_json_geo["features"][0]["geometry"]["coordinates"])

            multi_line_arr.append(mls_temp)
            for geom in mls_temp.geoms:
                line_str_arr.append(geom)

        # transformer = Transformer.from_crs("epsg:4326", "epsg:32633",always_xy=True)
        # transformer1 = Transformer.from_crs("epsg:32633","epsg:4326",always_xy=True)
        transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
        transformer1 = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)
        timedelta_hours = 1
        i = 0
        lat_long_arr = []
        date_arr = []
        time_arr = []
        lat_arr = []
        lon_arr = []
        date_time_arr = []
        s_distance_arr = []
        s_time_arr = []
        speed_arr_res = []
        course_arr = []
        waypoint_type_arr = []
        for segment in line_str_arr:
            dtobj = datetime.strptime(df2["DateTime"].values[i], "%Y-%m-%dT%H:%M:%S")
            dtobj_future = datetime.strptime(df2["DateTime"].values[i + 1], "%Y-%m-%dT%H:%M:%S")
            time_diff = dtobj_future - dtobj
            time_diff_hours = time_diff.total_seconds() / (60 * 60)
            num_segments = round(time_diff_hours / 1)
            orig_time_arr.append(df2["DateTime"].values[i])
            # date_arr.append(datetime.strftime(dtobj,"%Y-%m-%d"))
            # time_arr.append(datetime.strftime(dtobj,"%H:%M:%S"))
            # date_str =
            line2 = transform(transformer.transform, segment)

            print("<<" * 50)

            print(line2)

            if line2.length / 1852 == 0:
                continue

            print(">>" * 50)

            speed = speed_arr[i]

            #     if(speed==0):
            #         #print(round(line2.length/1852,2))
            #         segment_length_in_nm = round(line2.length/1852,2)/8
            #         print("Speed is 0. Hence subdiving it into 8 segments."+str(segment_length_in_nm))
            #     else:
            #         segment_length_in_nm=speed*3

            print("##" * 50)
            print("Segment:" + str(i + 1))
            print("Start Time:" + df2["DateTime"].values[i])
            print("Time Difference:" + str(time_diff_hours) + " hours")
            print("Number of Segments:" + str(num_segments))
            print("Original Segment Length:" + str(round(line2.length / 5556, 2)) + " NM.")
            print("Speed:" + str(speed) + " knots.")
            if (num_segments <= 1):
                segment_length_in_nm = line2.length / 5556
            else:
                segment_length_in_nm = (line2.length / 5556) / (num_segments)
            print("Sub Dividing Line into " + str(segment_length_in_nm) + " NM segments.")
            print("Segment Length NM:" + str(segment_length_in_nm))
            #     p = gpd.GeoSeries(segment)
            #     p.plot()
            #     plt.show()
            result = cut(line2, math.ceil(segment_length_in_nm * 5556), list())
            print("Total Number of sub segments:" + str(len(result)))
            #     for item in result:
            #         print("Item Length:"+str(item.length/1852)+" NM")
            #         if(item.length/1852<0.001):
            #             print("Removing nano segment:")
            #             result.remove(item)

            print("New Total Number of sub segments:" + str(len(result)))
            print(dtobj)
            print("##" * 50)
            j = 0
            time_added = 1
            prev_point_time = None
            temp_date = dtobj
            for item in result:
                # if(temp_date==None):
                #    temp_date = dtobj
                print("--" * 50)
                print("Sub Segment:" + str(j + 1))
                print("Time:" + datetime.strftime(temp_date, "%Y-%m-%dT%H:%M:%S"))
                print("Item Length:" + str(round(item.length / 1852, 2)) + " NM")
                line3 = transform(transformer1.transform, item)
                temp = (line3.boundary)
                start_point = list(temp.geoms)[0]  # LONG, LAT FORMAT
                end_point = list(temp.geoms)[1]
                print("Start Point:" + str(start_point.y) + "," + str(start_point.x))
                print("End Point:" + str(end_point.y) + "," + str(end_point.x))
                lat_long_arr.append((start_point.y, start_point.x))
                date_arr.append(datetime.strftime(temp_date, "%Y-%m-%d"))
                time_arr.append(datetime.strftime(temp_date, "%H:%M:%S"))
                date_time_arr.append(
                    datetime.strftime(temp_date, "%Y-%m-%d") + "T" + datetime.strftime(temp_date, "%H:%M:%S"))
                temp_geodesic = Geodesic.WGS84.Inverse(start_point.y, start_point.x, end_point.y, end_point.x)
                print(temp_geodesic)
                print(start_point.x, start_point.y, end_point.x, end_point.y)
                segment_distance = round(item.length / 5556, 2)
                segment_time = 1
                lat_arr.append(start_point.y)
                lon_arr.append(start_point.x)
                if (i == 0 and j == 0):
                    s_distance_arr.append(0)
                    s_time_arr.append(0)
                    speed_arr_res.append(0)
                    course_arr.append(0)
                else:
                    s_distance_arr.append(prev_segment_distance)
                    time_diff = temp_date - prev_segment_time
                    time_diff_hours = time_diff.total_seconds() / (60 * 60)
                    s_time_arr.append(time_diff_hours)
                    speed_arr_res.append(prev_segment_distance / time_diff_hours)
                    course_arr.append(prev_course)

                if (j == 0):
                    waypoint_type_arr.append(ORIGINAL)
                else:
                    waypoint_type_arr.append(DERIVED)
                # lat_long_arr.append((end_point.y,end_point.x))
                #         p = gpd.GeoSeries(line3)
                #         p.plot()
                #         plt.show()
                j += 1
                prev_point_time = temp_date
                prev_segment_distance = segment_distance
                prev_segment_time = temp_date
                prev_course = temp_geodesic["azi1"] + 360 if float(temp_geodesic["azi1"]) < 0 else float(
                    temp_geodesic["azi1"])
                print("Course:" + str(prev_course))
                print("--" * 50)
                temp_date = dtobj + timedelta(hours=time_added)
                time_added += timedelta_hours
                # if(temp_date>dtobj_future):
                #    temp_date = dtobj_future
            print("==" * 50)
            i += 1
        lat_arr.append(end_point.y)
        lon_arr.append(end_point.x)
        s_distance_arr.append(segment_distance)
        waypoint_type_arr.append(ORIGINAL)
        course_temp1 = temp_geodesic["azi1"] + 360 if float(temp_geodesic["azi1"]) < 0 else float(temp_geodesic["azi1"])
        course_arr.append(course_temp1)
        lat_long_arr.append((end_point.y, end_point.x))
        temp_date_final = datetime.strptime(df2["DateTime"].values[len(df2) - 1], "%Y-%m-%dT%H:%M:%S")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(temp_date_final)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$")
        date_arr.append(datetime.strftime(temp_date_final, "%Y-%m-%d"))
        time_arr.append(datetime.strftime(temp_date_final, "%H:%M:%S"))
        date_time_arr.append(
            datetime.strftime(temp_date_final, "%Y-%m-%d") + "T" + datetime.strftime(temp_date_final, "%H:%M:%S"))
        time_diff = temp_date_final - prev_segment_time
        time_diff_hours = time_diff.total_seconds() / (60 * 60)
        s_time_arr.append(time_diff_hours)
        speed_arr_res.append(prev_segment_distance / time_diff_hours)
        return date_arr, time_arr, lat_arr, lon_arr, s_distance_arr, s_time_arr, speed_arr_res, course_arr, waypoint_type_arr


    def generate_intermediate_values_new(df1):
        df2 = df1.T
        df2 = df2.dropna(how="all")
        # print('checking df2 datatime',df2)
        dates_str = [datetime.strftime(pd.Timestamp(dt), "%Y-%m-%d") for dt in df2["Date"].values]
        times_str = [str(dtobj) for dtobj in df2["Time ( UTC)"].values]
        combined_time_stamp = [a + "T" + b for a, b in zip(dates_str, times_str)]
        df2["DateTime"] = combined_time_stamp
        df2 = df2.sort_values("DateTime")
        route_arr, speed_arr = generate_route_array(df2)
        #     print('The BELOW CODE IS FOR DICTIONARY')
        #     print(d)
        date_arr, time_arr, lat_arr, lon_arr, s_distance_arr, s_time_arr, speed_arr_res, course_arr, waypoint_type_arr = generate_sub_segments(
            route_arr, speed_arr, df2)
        N_date_arr, N_time_arr, N_lat_arr, N_lon_arr, N_s_distance_arr, N_s_time_arr, N_speed_arr_res, N_course_arr, N_waypoint_type_arr = N_generate_sub_segments(
            route_arr, speed_arr, df2)

        # print(df2)

        data = df1
        # print(data)
        # data.to_csv('data_df1.csv')

        data = data.iloc[2:4, :].dropna(axis=1)
        data.insert(len(data.columns) - 1, 'LAST', data.pop('Arrival (EOSP)'))
        # print(data)
        # ------
        org_lat_array = data.iloc[0, 0:].values
        org_lng_array = data.iloc[1, 0:].values

        print(org_lat_array, org_lng_array)

        data_point = list(zip(org_lat_array, org_lng_array))
        print(data_point)

        new_df2 = data.T.reset_index(drop=True)

        # print(new_df2)

        # ---------

        print(N_lat_arr)
        print(N_lon_arr)

        lst = list(zip(N_lat_arr, N_lon_arr))

        print(lst)

        df10 = pd.DataFrame(lst, columns=['Lat', 'lng'])
        df10 = df10.round(2)
        print(df10)

        if no_of_legs == 2:

            df10['leg'] = 'leg1'

            # print(df10)

            mask = df10['Lat'].isin([round(lat, 2)]) & df10['lng'].isin([round(lng, 2)])

            print(lat, lng)

            print(mask)

            leg1 = df10[mask]
            print(leg1)

            split_index = df10[mask].index
            split_index

            leg2 = df10.iloc[split_index[0]:, :]
            leg2['leg'] = 'leg2'

            final_df = pd.concat([df10, leg2], join='inner', axis=0, ignore_index=False)

            # print(final_df)

            final_plot_df = final_df[~final_df.index.duplicated(keep='last')]
            # print(final_plot_df)

            leg1 = final_plot_df[final_plot_df.loc[:, 'leg'] == 'leg1']
            print(leg1)
            leg2 = final_plot_df[final_plot_df.loc[:, 'leg'] == 'leg2']
            print(leg2)

            join = final_plot_df.iloc[leg1.count().unique()[0] - 1:leg1.count().unique()[0] + 1, :]
            print(join)
            generate_map1(leg1, join, leg2, new_df2)



        else:
            df10['leg'] = 'leg1'
            final_plot_df = df10
            leg = final_plot_df[final_plot_df.loc[:, 'leg'] == 'leg1']
            print(leg)
            generate_map2(leg, new_df2)

        final_df = generate_final_df(date_arr, time_arr, lat_arr, lon_arr, s_distance_arr, s_time_arr, speed_arr_res,
                                     course_arr, waypoint_type_arr)
        return final_df


    def convert_degree_to_radians(degree):
        return degree / radians_to_degree


    def calculate_good_weather_periods(df1, filter_type, current_excluded, current_limit, bf_limit, swell_height_limit,
                                       sig_wave_height_limit, wind_wave_height_limit):
        print("printing all", current_excluded, current_limit, bf_limit, swell_height_limit, sig_wave_height_limit,
              wind_wave_height_limit)

        format1 = "%d %b %YT%H:%M:%S"  # %Y-%m-%dT%H:%M:%S
        weather_eval_arr = []
        for i in range(len(df1.columns)):
            col_name = df1.columns[i]
            wind_bf = df1[col_name]["Wind Force(BF Scale)"]
            swell_height = df1[col_name]["swellHeight"]
            sig_wave_height = df1[col_name]["waveHeight"]
            avg_speed_since_last_noon_report = df1[col_name]["Average Speed Since Last noon Report"]
            avg_course_since_last_noon_report = df1[col_name]["Average Ships Course since Last Report"]
            current_speed = df1[col_name]["currentSpeed"]
            current_direction = df1[col_name]["currentDirection"]
            wind_wave_height = df1[col_name]["windWaveHeight"]
            datapoint_type = df1[col_name]["Point Type"]
            steaming_hours = df1[col_name]["Steaming Hours Since Last noon Report"]
            # print(col_name)
            if (col_name > 0):
                date_string_prev = df1.loc['Date'][col_name - 1]
                time_string_prev = df1.loc['Time ( UTC)'][col_name - 1]
                date_time_prev = str(date_string_prev) + "T" + str(time_string_prev)
                date_string = df1.loc['Date'][col_name]
                time_string = df1.loc['Time ( UTC)'][col_name]
                date_time = str(date_string) + "T" + str(time_string)
                # print(date_time_prev)
                timestamp_start = datetime.strptime(date_time_prev, format1)
                timestamp_end = datetime.strptime(date_time, format1)

                time_diff = (timestamp_end - timestamp_start).total_seconds() / 3600
            else:
                time_diff = 0

            current_factor = avg_speed_since_last_noon_report - math.sqrt((np.power(current_speed, 2) + np.power(
                avg_speed_since_last_noon_report, 2) - 2 * avg_speed_since_last_noon_report * current_speed * math.cos(
                convert_degree_to_radians(current_direction - avg_course_since_last_noon_report + 180))))
            weather_eval = ""
            weather_type = "Yes"
            if (current_excluded == "Yes"):
                if (filter_type == "BF Only"):
                    print("BF filter type selected")
                    # if(current_factor<current_limit or wind_bf>bf_limit):
                    #  weather_eval="Bad"
                    if (wind_bf > bf_limit) and (current_factor < current_limit):
                        weather_eval = "WI,CU"
                    elif (wind_bf > bf_limit):
                        weather_eval = "WI"
                    elif (current_factor < current_limit):
                        weather_eval = "CU"

                elif (filter_type == "SWH"):
                    print("SWH filter type selected")
                    # if(current_factor<current_limit or wind_bf>bf_limit or sig_wave_height>sig_wave_height_limit):
                    #  weather_eval="Bad"
                    print(type(current_factor),type(current_limit))
                    print(type(sig_wave_height), type(sig_wave_height_limit))
                    print(type(wind_bf), type(bf_limit))
                    if (wind_bf > bf_limit) and (sig_wave_height > sig_wave_height_limit) and (current_factor < current_limit):
                        weather_eval = "WI,WA,CU"
                    elif (wind_bf > bf_limit) and (sig_wave_height > sig_wave_height_limit):
                        weather_eval = "WI,WA"
                    elif (sig_wave_height > sig_wave_height_limit) and (current_factor < current_limit):
                        weather_eval = "WA,CU"
                    elif (wind_bf > bf_limit) and (current_factor < current_limit):
                        weather_eval = "WI,CU"
                    elif (wind_bf > bf_limit):
                        weather_eval = "WI"
                    elif (sig_wave_height > sig_wave_height_limit):
                        weather_eval = "WA"
                    elif (current_factor < current_limit):
                        weather_eval = "CU"

                elif (filter_type == "DSS"):
                    # if(current_factor<current_limit or wind_bf>bf_limit or wind_wave_height>wind_wave_height_limit or swell_height>swell_height_limit):
                    #  weather_eval="Bad"
                    if (wind_bf > bf_limit) and (wind_wave_height > wind_wave_height_limit or swell_height > swell_height_limit) and (current_factor < current_limit):
                        weather_eval = "WI,WA,CU"
                    elif (wind_bf > bf_limit) and (wind_wave_height > wind_wave_height_limit or swell_height > swell_height_limit):
                        weather_eval = "WI,WA"
                    elif (wind_wave_height > wind_wave_height_limit or swell_height > swell_height_limit) and (current_factor < current_limit):
                        weather_eval = "WA,CU"
                    elif (wind_bf > bf_limit) and (current_factor < current_limit):
                        weather_eval = "WI,CU"
                    elif (wind_bf > bf_limit):
                        weather_eval = "WI"
                    elif (wind_wave_height > wind_wave_height_limit or swell_height > swell_height_limit):
                        weather_eval = "WA"
                    elif (current_factor < current_limit):
                        weather_eval = "CU"

                elif (filter_type == "SWH+DSS"):
                    # if(current_factor<current_limit or wind_bf>bf_limit or sig_wave_height>sig_wave_height_limit or swell_height>swell_height_limit or wind_wave_height>wind_wave_height_limit):
                    #  weather_eval="Bad"
                    if (wind_bf > bf_limit) and (sig_wave_height > sig_wave_height_limit or swell_height > swell_height_limit or wind_wave_height > wind_wave_height_limit) and (current_factor < current_limit):
                        weather_eval = "WI,WA,CU"
                    elif (wind_bf > bf_limit) and (sig_wave_height > sig_wave_height_limit or swell_height > swell_height_limit or wind_wave_height > wind_wave_height_limit):
                        weather_eval = "WI,WA"
                    elif (sig_wave_height > sig_wave_height_limit or swell_height > swell_height_limit or wind_wave_height > wind_wave_height_limit) and (current_factor < current_limit):
                        weather_eval = "WA,CU"
                    elif (wind_bf > bf_limit) and (current_factor < current_limit):
                        weather_eval = "WI,CU"
                    elif (wind_bf > bf_limit):
                        weather_eval = "WI"
                    elif (sig_wave_height > sig_wave_height_limit or swell_height > swell_height_limit or wind_wave_height > wind_wave_height_limit):
                        weather_eval = "WA"
                    elif (current_factor < current_limit):
                        weather_eval = "CU"

                # print("Current Factor:"+str(current_factor)+",Eval:"+weather_eval)

            elif (current_excluded == "No"):
                # if(wind_bf>bf_limit or sig_wave_height>sig_wave_height_limit):
                #  weather_eval="Bad"
                if (filter_type == "BF Only"):
                    # if(wind_bf>bf_limit):
                    #  weather_eval="Bad"
                    if (wind_bf > bf_limit):
                        weather_eval = "WI"

                elif (filter_type == "SWH"):
                    # if(wind_bf>bf_limit or sig_wave_height>sig_wave_height_limit):
                    #  weather_eval="Bad"

                    if (wind_bf > bf_limit) and (sig_wave_height > sig_wave_height_limit):
                        weather_eval = "WI,WA"
                    elif (wind_bf > bf_limit):
                        weather_eval = "WI"
                    elif (sig_wave_height > sig_wave_height_limit):
                        weather_eval = "WA"

                elif (filter_type == "DSS"):
                    # if(wind_bf>bf_limit or wind_wave_height>wind_wave_height_limit or swell_height>swell_height_limit):
                    #  weather_eval="Bad"

                    if (wind_bf>bf_limit) and (wind_wave_height>wind_wave_height_limit or swell_height>swell_height_limit):
                        weather_eval = "WI,WA"
                    elif (wind_bf > bf_limit):
                        weather_eval = "WI"
                    elif (wind_wave_height>wind_wave_height_limit or swell_height>swell_height_limit):
                        weather_eval = "WA"

                elif (filter_type == "SWH+DSS"):
                    # if(wind_bf>bf_limit or sig_wave_height>sig_wave_height_limit or swell_height>swell_height_limit or wind_wave_height>wind_wave_height_limit):
                    #  weather_eval="Bad"

                    if (wind_bf>bf_limit) and (sig_wave_height>sig_wave_height_limit or swell_height>swell_height_limit or wind_wave_height>wind_wave_height_limit):
                        weather_eval = "WI,WA"
                    elif (wind_bf>bf_limit):
                        weather_eval = "WI"
                    elif (sig_wave_height>sig_wave_height_limit or swell_height>swell_height_limit or wind_wave_height>wind_wave_height_limit):
                        weather_eval = "WA"

            if (weather_eval != ""):
                weather_type = "No"
            weather_eval_arr.append(weather_eval)
            df1.loc["Current factor", col_name] = current_factor
            df1.loc["Current Distance", col_name] = steaming_hours * current_factor
            df1.loc["Time Period", col_name] = time_diff
            # df1.loc["Good Weather Period",col_name] = current_factor
            if (datapoint_type == "Original"):
                df1.loc["Good Weather", col_name] = weather_type  # "NA"
            elif (datapoint_type == "Derived"):
                df1.loc["Good Weather", col_name] = weather_type  # weather_eval
                # df1.loc["Daily Current Factor",col_name] = ""

        day_start = False
        bad_weather_time = 0
        good_weather_time = 0
        current_factor_sum = 0
        temp_current_factor_sum = None
        current_factor_sum_count = 0
        temp_current_factor_sum_count = 0
        for i in range(len(df1.columns)):
            # print(i)
            col_name = df1.columns[i]
            datapoint_type = df1[col_name]["Point Type"]
            time_period = df1[col_name]["Time Period"]
            weather_type = df1.loc["Good Weather", col_name]
            current_factor = df1.loc["Current factor", col_name]
            # print(np.isnan(current_factor))
            if (np.isnan(current_factor)):
                current_factor = 0
            df1.loc["Total Good Weather Period", col_name] = weather_eval_arr[i]  # "Yes"
            # print("Sum:"+str(current_factor_sum)+",Factor:"+str(current_factor))
            if (day_start == False and datapoint_type == "Original"):
                day_start = True
                df1.loc["Total Good Weather Period", col_name] = "No"

            elif (day_start and datapoint_type == "Derived"):
                if (weather_type == "No"):
                    bad_weather_time += time_period
                else:
                    good_weather_time += time_period
                current_factor_sum += current_factor
                current_factor_sum_count += 1

            elif (day_start and datapoint_type == "Original"):
                if (good_weather_time >= good_weather_hours_per_day_limit):
                    # df1.loc["Good Weather Period",col_name] = good_weather_time
                    df1.loc["Total Good Weather Period", col_name] = "Yes"
                else:
                    df1.loc["Total Good Weather Period", col_name] = "No"
                current_factor_sum += current_factor
                current_factor_sum_count += 1
                temp_current_factor_sum = current_factor_sum
                temp_current_factor_sum_count = current_factor_sum_count
                current_factor_sum_count = 0
                current_factor_sum = 0
                good_weather_time = 0
                bad_weather_time = 0

            if (datapoint_type == "Original"):
                if (temp_current_factor_sum is not None):
                    df1.loc["Daily Current Factor", col_name] = temp_current_factor_sum / temp_current_factor_sum_count
                    print("daily_current_factor", temp_current_factor_sum, temp_current_factor_sum_count,
                            temp_current_factor_sum / temp_current_factor_sum_count)
                else:
                    df1.loc["Daily Current Factor", col_name] = 0
            elif (datapoint_type == "Derived"):
                df1.loc["Daily Current Factor", col_name] = ""
        return df1


    def calculate_new_position_given_distance_and_bearing(lat1, lon1, initial_bearing, distance_in_meters):
        origin = geopy.Point(lat1, lon1)
        destination = VincentyDistance(kilometers=(distance_in_meters / 1000)).destination(origin, initial_bearing)
        lat2, lon2 = destination.latitude, destination.longitude
        return (lat2, lon2)


    def generate_final_df(date_arr, time_arr, lat_arr, lon_arr, s_distance_arr, s_time_arr, speed_arr_res, course_arr,
                          waypoint_type_arr):
        final_data_df = pd.DataFrame()
        final_data_df["Date"] = date_arr
        final_data_df["Time ( UTC)"] = time_arr
        final_data_df["Latitude"] = lat_arr
        final_data_df["Longitude"] = lon_arr
        final_data_df["Steaming Distance Since Last Noon Report"] = s_distance_arr
        final_data_df["Steaming Hours Since Last noon Report"] = s_time_arr
        final_data_df["Average Speed Since Last noon Report"] = speed_arr_res
        final_data_df["Average Ships Course since Last Report"] = course_arr
        final_data_df["Point Type"] = waypoint_type_arr
        final_data_df["swellDirection"] = np.nan
        final_data_df["swellHeight"] = np.nan
        final_data_df["waveDirection"] = np.nan
        final_data_df["waveHeight"] = np.nan
        final_data_df["windDirection"] = np.nan
        final_data_df["Wind Force(BF Scale)"] = np.nan
        final_data_df["windSpeed"] = np.nan
        final_data_df["airTemperature"] = np.nan
        final_data_df["waterTemperature"] = np.nan
        final_data_df["pressure"] = np.nan
        final_data_df["precipitation"] = np.nan
        final_data_df["visibility"] = np.nan
        final_data_df["windWaveDirection"] = np.nan
        final_data_df["windWaveHeight"] = np.nan
        final_data_df["currentSpeed"] = np.nan
        final_data_df["currentDirection"] = np.nan
        return final_data_df.T


    def generate_intermediate_values(df1):
        df2 = df1.T
        # del df2[np.nan]
        df2 = df2.dropna(how="all")
        # print(df2)
        # print(df2["Date"].values)
        dates_str = [datetime.strftime(pd.Timestamp(dt), "%Y-%m-%d") for dt in df2["Date"].values]
        # print("Dates:")
        # print(dates_str)
        times_str = [str(dtobj) for dtobj in df2["Time ( UTC)"].values]
        # print(times_str)
        # combined_time_stamp = [datetime.strptime(a+"T"+b,'%Y-%m-%dT%H:%M:%S') for a,b in zip(dates_str,times_str)]
        combined_time_stamp = [a + "T" + b for a, b in zip(dates_str, times_str)]
        # print(combined_time_stamp)
        df2["DateTime"] = combined_time_stamp
        df2 = df2.sort_values("DateTime")

        timedelta_hours = 3
        col_len = len(df2["DateTime"].values)
        date_arr = []
        time_arr = []
        lat_arr = []
        lon_arr = []
        s_distance_arr = []
        s_time_arr = []
        speed_arr = []
        waypoint_type_arr = []
        course_arr = []

        # print(df2.columns)
        for i in range(col_len):
            dtobj = datetime.strptime(df2["DateTime"].values[i], "%Y-%m-%dT%H:%M:%S")
            latitude = df2["Latitude"].values[i]
            longitude = df2["Longitude"].values[i]
            if (i == 0):
                s_distance_arr.append(0)
                s_time_arr.append(0)
                speed_arr.append(0)
                waypoint_type_arr.append(ORIGINAL)
                course_arr.append(df2["Average Ships Course since Last Report"].values[i])
            elif (i >= col_len - 1):
                # print("----------------------------")
                # print(str(dtobj)+"-"+str(latitude)+","+str(longitude))
                # print("----------------------------")
                date_arr.append(datetime.strftime(dtobj, "%Y-%m-%d"))
                time_arr.append(datetime.strftime(dtobj, "%H:%M:%S"))
                lat_arr.append(latitude)
                lon_arr.append(longitude)
                temp_geodesic = Geodesic.WGS84.Inverse(new_lat, new_lon, latitude, longitude)
                # print("Old Lat,Long:"+str(new_lat)+","+str(new_lon)+"-New Lat,Long:"+str(lat_long_tuple[0])+","+str(lat_long_tuple[1])+",Distance:"+str(temp_geodesic["s12"]/1852))
                # print(temp_geodesic)
                s_distance_arr.append(temp_geodesic["s12"] / 1852)
                s_time_arr.append((dtobj - temp_date).seconds / 3600)
                print(dtobj)
                print(temp_date)
                speed_arr.append((temp_geodesic["s12"] / 1852) / ((dtobj - temp_date).seconds / 3600))
                waypoint_type_arr.append(ORIGINAL)
                # print("$$$$$$")
                # print(str(dtobj)+"-"+str(temp_date))
                # print("$$$$$$")
                continue
            else:
                temp_geodesic = Geodesic.WGS84.Inverse(new_lat, new_lon, latitude, longitude)
                # print("Old Lat,Long:"+str(new_lat)+","+str(new_lon)+"-New Lat,Long:"+str(lat_long_tuple[0])+","+str(lat_long_tuple[1])+",Distance:"+str(temp_geodesic["s12"]/1852))
                s_distance_arr.append(temp_geodesic["s12"] / 1852)
                s_time_arr.append((dtobj - temp_date).seconds / 3600)
                # print(dtobj)
                # print(temp_date)
                if ((dtobj - temp_date).seconds == 0):
                    speed_arr.append(0)
                else:
                    speed_arr.append((temp_geodesic["s12"] / 1852) / ((dtobj - temp_date).seconds / 3600))
                waypoint_type_arr.append(ORIGINAL)
            # print("----------------------------")
            # print(str(dtobj)+"-"+str(latitude)+","+str(longitude))
            # print("----------------------------")

            date_arr.append(datetime.strftime(dtobj, "%Y-%m-%d"))
            time_arr.append(datetime.strftime(dtobj, "%H:%M:%S"))
            lat_arr.append(latitude)
            lon_arr.append(longitude)

            next_dtobj = datetime.strptime(df2["DateTime"].values[i + 1], "%Y-%m-%dT%H:%M:%S")
            course_heading = df2["Average Ships Course since Last Report"].values[i + 1]
            course_speed = df2["Average Speed Since Last noon Report"].values[i + 1]
            course_arr.append(course_heading)
            time_added = 3
            j = 0
            while True:
                temp_date1 = dtobj + timedelta(hours=time_added)
                time_added += timedelta_hours
                distance = course_speed * timedelta_hours
                if (temp_date1 >= next_dtobj):
                    if (j == 0):
                        temp_date = dtobj
                        new_lat = latitude
                        new_lon = longitude
                    break
                temp_date = temp_date1
                # print(temp_date)

                # print("Distance:"+str(distance))
                if (j == 0):
                    lat_long_tuple = calculate_new_position_given_distance_and_bearing(latitude, longitude, course_heading,
                                                                                       distance * 1852)
                    temp_geodesic = Geodesic.WGS84.Inverse(latitude, longitude, lat_long_tuple[0], lat_long_tuple[1])
                    # print("Old Lat,Long:"+str(latitude)+","+str(longitude)+"-New Lat,Long:"+str(lat_long_tuple[0])+","+str(lat_long_tuple[1])+",Distance:"+str(temp_geodesic["s12"]/1852))
                    s_distance_arr.append(temp_geodesic["s12"] / 1852)
                else:
                    lat_long_tuple = calculate_new_position_given_distance_and_bearing(new_lat, new_lon, course_heading,
                                                                                       distance * 1852)
                    # print("Old Lat,Long:"+str(new_lat)+","+str(new_lon)+"-New Lat,Long:"+str(lat_long_tuple[0])+","+str(lat_long_tuple[1])+",Distance:"+str(temp_geodesic["s12"]/1852))
                    temp_geodesic = Geodesic.WGS84.Inverse(new_lat, new_lon, lat_long_tuple[0], lat_long_tuple[1])
                    s_distance_arr.append(temp_geodesic["s12"] / 1852)
                s_time_arr.append(timedelta_hours)
                speed_arr.append((temp_geodesic["s12"] / 1852) / (timedelta_hours))
                j += 1
                new_lat = lat_long_tuple[0]
                new_lon = lat_long_tuple[1]
                # print(str(temp_date)+"-"+str(new_lat)+","+str(new_lon))
                date_arr.append(datetime.strftime(temp_date, "%Y-%m-%d"))
                time_arr.append(datetime.strftime(temp_date, "%H:%M:%S"))
                lat_arr.append(new_lat)
                lon_arr.append(new_lon)
                waypoint_type_arr.append(DERIVED)
                course_arr.append(course_heading)
        final_df = generate_final_df(date_arr, time_arr, lat_arr, lon_arr, s_distance_arr, s_time_arr, speed_arr,
                                     course_arr, waypoint_type_arr)
        return final_df


    access_key = 'adcc8986-97d6-11ec-8fd4-0242ac130002-adcc8a12-97d6-11ec-8fd4-0242ac130002'
    formater = "%Y-%m-%d%H:%M:%S"
    formater1 = "%Y-%m-%dT%H:%M:%S"

    def save_screen_shot_of_map(filename):
        # print("did it work")
        # pass
        options = webdriver.ChromeOptions()
        #if(ENVIRONMENT=="LOCAL"):
        wd = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        options.add_argument('window-size=1280x1024')
        desired_dpi = 1.0
        options.add_argument(f"--force-device-scale-factor={desired_dpi}")
        dx, dy = wd.execute_script("var w=window; return [w.outerWidth - w.innerWidth, w.outerHeight - w.innerHeight];")
        print("dx and dy",dx,dy)
        wd.set_window_size(1280 +dx, 1024 + dy)
        # print("filename before maps",filename)
        filename_html=filename.replace("xlsx", "html")
        filename_png=filename.replace("xlsx","png")
        # print("./maps/"+filename)
        # wd.get("http://localhost:63342/performance_form/maps/"+filename+"?_ijt=iqo1kak408fr73po3e56pj3qja&_ij_reload=RELOAD_ON_SAVE")
        # time.sleep(10)
        # wd.save_screenshot("./images/"+filename.replace("xlsx", "png"))
        filename="C:/Users/DELL/PycharmProjects/performance_form/performance_form/maps/"+filename_html
        wd.get("file://"+filename)
        time.sleep(10)
        filename = "C:/Users/DELL/PycharmProjects/performance_form/performance_form/images/"+filename_png
        wd.save_screenshot(filename)

        print(filename)


    def add_arrow_line(latitudes, longitudes, m, line_obj, color):
        points = list(zip([latitudes, longitudes]))
        text = " ⟶ "
        arrow_color = color
        arrow_size = 0
        text_positions = [i for i in range(1, len(points) - 1)]
        PolyLineTextPath(
            line_obj,
            positions=text_positions,
            text=text,
            repeat=True,
            offset=arrow_size,
            attributes={"fill-opacity": 0.8, "font-size": "15", "font-weight": "bold", "fill": arrow_color}
        ).add_to(m)
        return m


    def plot_map(df, filename="map.html"):
        # print(df)
        if ("Ship Name" in df.columns):
            latitudes = df[df['Ship Name'] == "Latitude"].values
            longitudes = df[df['Ship Name'] == "Longitude"].values
        else:
            latitudes = df[df['param'] == "Latitude"].values
            longitudes = df[df['param'] == "Longitude"].values

        df1 = pd.DataFrame()
        df1["latitudes"] = np.array(latitudes[0])
        df1["longitudes"] = np.array(longitudes[0])
        df1 = df1.dropna()
        df1.rename(index={2: len(df1) + 1}, inplace=True)
        df1 = df1.sort_index()
        df1.reset_index(inplace=True)
        del df1["index"]
        df1 = df1[1:]
        # df1
        df1["waypoint_id"] = np.arange(1, len(df1) + 1)

        new_latitudes = [i if str(i).isnumeric() else float(str(i).replace(" ", "")) for i in df1["latitudes"].values]
        new_longitudes = [i if str(i).isnumeric() else float(str(i).replace(" ", "")) for i in df1["longitudes"].values]
        df1["latitudes"] = pd.to_numeric(new_latitudes)
        df1["longitudes"] = pd.to_numeric(new_longitudes)
        # Create the map and add the line
        m = folium.Map(zoom_start=4)
        # m.PolyLine(df1.values, line_color='#FF0000', line_weight=5)
        # print(df1)
        # print(df1.values)
        # folium.PolyLine(df1[["latitudes","longitudes"]].values,color='red', weight=5, opacity=0.8).add_to(m)
        if (two_legs):
            try:
                break_down_index = df1[(df1["latitudes"] == first_leg_end_position[0]) & (
                            df1["longitudes"] == first_leg_end_position[1])].index.values[0]
                df1_1 = df1[:break_down_index]
                df1_2 = df1[break_down_index - 1:]
                if (connect_lines):
                    line_obj_1 = folium.PolyLine(df1_1[["latitudes", "longitudes"]].values, color='green', weight=5,
                                                 opacity=0.8).add_to(m)
                    line_obj_2 = folium.PolyLine(df1_2[["latitudes", "longitudes"]].values, color='red', weight=5,
                                                 opacity=0.8).add_to(m)
                for i in range(0, len(df1_1)):
                    folium.CircleMarker(
                        location=[df1_1.iloc[i]['latitudes'], df1_1.iloc[i]['longitudes']],
                        popup=str(df1_1.iloc[i]['waypoint_id']),
                        # icon=folium.Icon(color='blue',icon_color='green')
                        radius=3.5,
                        color="green"
                    ).add_to(m)

                for i in range(0, len(df1_2)):
                    folium.CircleMarker(
                        location=[df1_2.iloc[i]['latitudes'], df1_2.iloc[i]['longitudes']],
                        popup=str(df1_1.iloc[i]['waypoint_id']),
                        # icon=folium.Icon(color='blue',icon_color='red')
                        radius=3.5,
                        color="red"
                    ).add_to(m)
                if (connect_lines):
                    m = add_arrow_line(df1_1.latitudes, df1_1.longitudes, m, line_obj_1)
                    m = add_arrow_line(df1_2.latitudes, df1_2.longitudes, m, line_obj_2)

            except Exception as e:
                print("First Leg End Lat Long Does Not Match")
                if (connect_lines):
                    line_obj = folium.PolyLine(df1[["latitudes", "longitudes"]].values, color='red', weight=5,
                                               opacity=0.8).add_to(m)
                for i in range(0, len(df1)):
                    folium.Marker(
                        location=[df1.iloc[i]['latitudes'], df1.iloc[i]['longitudes']],
                        popup=str(df1.iloc[i]['waypoint_id']),
                    ).add_to(m)
                if (connect_lines):
                    m = add_arrow_line(df1.latitudes, df1.longitudes, m, line_obj)
        else:
            if (connect_lines):
                line_obj = folium.PolyLine(df1[["latitudes", "longitudes"]].values, color='red', weight=5,
                                           opacity=0.8).add_to(m)

            for i in range(0, len(df1)):
                folium.Marker(
                    location=[df1.iloc[i]['latitudes'], df1.iloc[i]['longitudes']], popup=str(df1.iloc[i]['waypoint_id']),
                ).add_to(m)
            if (connect_lines):
                m = add_arrow_line(df1.latitudes, df1.longitudes, m, line_obj)

        #   for i in range(0,len(df1)):
        #     folium.Marker(
        #         location=[df1.iloc[i]['latitudes'], df1.iloc[i]['longitudes']],popup=str(df1.iloc[i]['waypoint_id']),
        #     ).add_to(m)

        m.fit_bounds(m.get_bounds())
        # m.save('../content/drive/My Drive/Colab Notebooks/maps/'+filename)
        m.save('./maps/' + filename)
        # m.save(filename)
        try:
            save_screen_shot_of_map(filename)

        except Exception as e:
            print(e)
            print("Unable to save screenshot. Proceeding with the execution")
        return m


    def generate_map1(leg1, join, leg2, new_df2):
        m = folium.Map(zoom_start=4)

        points_leg1 = []
        for i in np.arange(leg1.index[0], leg1.index[-1] + 1):
            points_leg1.append([leg1['Lat'][i], leg1['lng'][i]])
            line_obj = folium.PolyLine(points_leg1, color='red', weight=0.5).add_to(m)
            m = add_arrow_line(leg1['Lat'][i], leg1['lng'][i], m, line_obj, color='red')

        points_leg2 = []
        for i in np.arange(leg2.index[0], leg2.index[-1] + 1):
            points_leg2.append([leg2['Lat'][i], leg2['lng'][i]])
            line_obj = folium.PolyLine(points_leg2, color='green', weight=0.5).add_to(m)
            m = add_arrow_line(leg2['Lat'][i], leg2['lng'][i], m, line_obj, color='green')

        points_join = []

        for i in np.arange(join.index[0], join.index[-1] + 1):
            points_join.append([join['Lat'][i], join['lng'][i]])
            line_obj = folium.PolyLine(points_join, color='red', weight=0.5).add_to(m)
            m = add_arrow_line(join['Lat'][i], join['lng'][i], m, line_obj, color='red')

        for i in np.arange(new_df2.index[0], new_df2.index[-1] + 1):
            folium.Marker(
                location=[new_df2.iloc[i][0], new_df2.iloc[i][1]],
                icon=folium.DivIcon(html=f"""
                         <div><svg>                    
                             <circle cx="1" cy="1" r="2" stroke="orange" stroke-width="3" fill="red" />
                        </svg></div>""")
            ).add_to(m)

        m.fit_bounds(m.get_bounds())
        m.save('./maps/' + filename.replace("xlsx", "html"))

        try:
            save_screen_shot_of_map(filename)

        except Exception as e:
            print(e)
            print("Unable to save screenshot. Proceeding with the execution")
        return m


    def generate_map2(leg, new_df2):
        m = folium.Map(zoom_start=4)

        points = []

        for i in np.arange(leg.index[0], leg.index[-1] + 1):
            points.append([leg['Lat'][i], leg['lng'][i]])

            line_obj = folium.PolyLine(points, color='red', weight=0.5).add_to(m)
            m = add_arrow_line(leg['Lat'][i], leg['lng'][i], m, line_obj, color='blue')

        for i in np.arange(new_df2.index[0], new_df2.index[-1] + 1):
            folium.Marker(
                location=[new_df2.iloc[i][0], new_df2.iloc[i][1]],
                icon=folium.DivIcon(html=f"""
                         <div><svg>                    
                             <circle cx="1" cy="1" r="2" stroke="red" stroke-width="3" fill="red" />
                        </svg></div>""")
            ).add_to(m)

        #         folium.Marker(
        #               location=[new_df2.iloc[i][0],new_df2.iloc[i][1]],
        #               #popup=data.iloc[i]['name'],
        #            ).add_to(m)

        m.fit_bounds(m.get_bounds())
        # m.save('../content/drive/My Drive/Colab Notebooks/maps/'+filename.replace("xlsx","html"))
        m.save('./maps/' + filename.replace("xlsx", "html"))

        try:
            save_screen_shot_of_map(filename)

        except Exception as e:
            print(e)
            print("Unable to save screenshot. Proceeding with the execution")
        return m

    def get_beaufort_scale_from_wind_speed(windspeed):
        # WIND SPEED IS IN METERS PER SECOND
        if (windspeed < 1):
            return 0
        elif (windspeed < 4.0):
            return 1
        elif (windspeed <= 6.0):
            return 2
        elif (windspeed <= 10.0):
            return 3
        elif (windspeed <= 16.0):
            return 4
        elif (windspeed <= 21):
            return 5
        elif (windspeed <= 27.0):
            return 6
        elif (windspeed <= 33.0):
            return 7
        elif (windspeed <= 40.0):
            return 8
        elif (windspeed <= 47.0):
            return 9
        elif (windspeed <= 55.0):
            return 10
        elif (windspeed <= 63.0):
            return 11
        else:
            return 12


    def make_sg_call(lat, lng, timestamp_start, source="noaa"):
        timestamp_start_str = timestamp_start.isoformat('T') + 'Z'
        timestamp_end = timestamp_start + timedelta(hours=1)
        timestamp_end_str = timestamp_end.isoformat('T') + 'Z'
        parameters = ['swellDirection', 'swellHeight', 'waveDirection', 'waveHeight', 'windDirection', 'windSpeed',
                      'airTemperature', 'waterTemperature', 'pressure', 'precipitation', 'visibility', 'windWaveDirection',
                      'windWaveHeight', 'currentSpeed', 'currentDirection']
        response = requests.get(
            'https://api.stormglass.io/v2/weather/point',
            params={
                'lat': lat,
                'lng': lng,
                'params': 'windSpeed,windDirection,waveHeight,waveDirection,currentSpeed,currentDirection,swellHeight,swellDirection,airTemperature,waterTemperature,pressure,precipitation,visibility,windWaveDirection,windWaveHeight',
                'start': timestamp_start_str,
                'end': timestamp_end_str,
                # 'source':source
            },
            headers={
                'Authorization': access_key
            }
        )
        # print(response.text)
        # Do something with response data.
        json_data = response.json()

        print(json_data)

        def log_details():

            lst = []
            values = [filename, datetime.utcnow(), 'SG', 1]
            lst.append(values)

            df = pd.DataFrame(lst)
            df = df.to_csv(header=None, index=False)
            f = open("/home/ubuntu/speed_consumption/notebooks/Forecast/results/demofile3.txt", "a")
            f.write(df)
            f.close()

        #log_details()

        # print(json_data)
        weather_params_dict = {}
        for weather_data in json_data['hours']:

            print("=**=" * 50)
            print(json_data['hours'])

            time_stamp = weather_data['time']
            # if(pd.to_datetime(time_stamp).strftime(formater1)==timestamp_start.strftime(formater1)):
            if (True):
                for param in parameters:
                    if (param in weather_data):
                        if (param in ['currentSpeed', 'currentDirection']):
                            if ("meto" in weather_data[param]):
                                weather_params_dict[param] = weather_data[param]["meto"]
                            elif ("sg" in weather_data[param]):
                                weather_params_dict[param] = weather_data[param]["sg"]
                        else:
                            if (source in weather_data[param]):
                                weather_params_dict[param] = weather_data[param][source]
                            # -----test code block
                            elif ('sg' in weather_data[param]):
                                weather_params_dict[param] = weather_data[param]['sg']

                                # -----test code block


                            else:
                                weather_params_dict[param] = np.nan
                    else:
                        weather_params_dict[param] = np.nan
                break

        return weather_params_dict


    def populate_weather_data(filename, filter_type, current_excluded, current_limit, bf_limit, swell_height_limit,
                              sig_wave_height_limit, wind_wave_height_limit):
        df = pd.read_excel('./dtn_output/' + filename)
        # print(df.columns)
        df1 = df.rename(columns={'Unnamed: 0': 'param'})
        # plot_map(df1,filename.split(".")[0]+".html")
        df1 = df1.set_index('param')
        if ("With" in df1.index):
            df1 = df1.drop(['With'])
        if ("Against" in df1.index):
            df1 = df1.drop(['Against'])
        df1 = df1[df1.index.notnull()]
        df1 = generate_intermediate_values_new(df1)

        parameters = df1.index[4:].tolist()
        indices = ['Date', 'Time ( UTC)', 'Latitude', 'Longitude']
        # df1 = df1.loc[indices]
        # df1 = df1.dropna(axis=1, how='all')
        for i in range(len(df1.columns)):
            col_name = df1.columns[i]
            # print("Column:"+str(col_name))
            if col_name == 'param':
                pass
            else:
                date_string = df1.loc['Date'][col_name]
                time_string = df1.loc['Time ( UTC)'][col_name]
                # print(date_string)
                # print(time_string)
                # print("Reached here 0")
                date = pd.to_datetime(date_string).date()
                date_ = date.strftime("%d %b %Y")
                date_time = str(date) + "T" + str(time_string)
                # print(date_time)
                timestamp_start = datetime.strptime(date_time, formater1)
                lat = str(df1.loc['Latitude'][col_name])
                long = df1.loc['Longitude'][col_name]
                # print("Reached here 1")
                source = "noaa"
                try:
                    weather_params_dict = make_sg_call(lat, long, timestamp_start, source)
                except:
                    time.sleep(20)
                    weather_params_dict = make_sg_call(lat, long, timestamp_start, source)
                # print(weather_params_dict.keys())
                for key in weather_params_dict.keys():
                    name = key
                    value = weather_params_dict[key]
                    param_name = name
                    df1.loc['Date'][col_name] = date_
                    # print("Key:"+str(key)+",Value:"+str(value))
                    if name == 'windSpeed':
                        new_value = float(value) * 1.944
                        bf = get_beaufort_scale_from_wind_speed(new_value)
                        df1.loc['Wind Force(BF Scale)', col_name] = bf
                        df1.loc[param_name, col_name] = new_value
                    elif name == 'currentSpeed':
                        new_value = float(value) * 1.944
                        df1.loc[param_name, col_name] = new_value
                    elif name == 'airTemperature' or name == 'waterTemperature':
                        new_value = float(value)
                        df1.loc[param_name, col_name] = new_value
                    else:
                        df1.loc[param_name, col_name] = value
                # print(df1[col_name])
        count = 1
        col_list = []
        for col in df1.columns:
            if ("nan" in str(col) or "Unnamed" in str(col)):
                col_name = "Noon " + str(count)
                count += 1
            else:
                col_name = col
            col_list.append(col_name)
        df1.columns = col_list
        try:
            print('calculate_good_weather_working')
            df1 = calculate_good_weather_periods(df1.copy(), filter_type, current_excluded, current_limit, bf_limit,
                                                 swell_height_limit, sig_wave_height_limit, wind_wave_height_limit)
        except Exception as e:
            print("Exception:" + str(e))
        # df1.to_csv("../content/drive/My Drive/Colab Notebooks/results/"+filename.split(".")[0]+".csv")
        df1.to_csv('./results_output/' + filename.split(".")[0] + ".csv")
    # -------------new input------
    api_key = "ZxdjtJ6eyxmJsvh3Z2m3xCPVAZv8fNVy9MQH3PKv"

    radians_to_degree = 180 / 3.141592653589793238462
    bf_limit = bf_limit_inp
    swell_height = swell_height_limit = swell_height_inp
    swh_limit = sig_wave_height_limit = swh_limit_inp
    windwave_limit = wind_wave_height_limit = windwave_limit_inp


    bf_limit_dss = 3  # manual entry , not from sheet
    current_limit = 0  # nothing to b e done here

    voyage_phase = voyage_phase_inp  # -- "MID" or "END" --> need to choosed as per need

    # Yes or No
    gwx_type = gwx_type_inp  # adverse current "inc or x" - adverse current
    gwx_hours = good_weather_hours_per_day_limit = gwx_hours_inp
    gwx_method = 'consecutive'  # -- choose between 'consecutive' or 'all' as per need

    connect_lines = False

    # current_excluded="Yes"


    report_type = report_type_inp  # BF/DSS/SWH_DSS/SWH

    if report_type == "BF":
        filter_type = "BF Only"
    elif report_type == "SWH":
        filter_type = "SWH"
    elif report_type == "DSS":
        filter_type = "DSS"
    else:
        filter_type = "SWH+DSS"

    if gwx_type == "x":
        adverse_current = "excluded"
        current_excluded = "Yes"
    else:
        adverse_current = "not_excluded"  # G78
        current_excluded = "No"

    no_of_legs = 1  #
    lat = 56.1513888888889
    lng = 10.2361111111111

    imo_id = 9422495  # default vessel draft = 9422495

    filename = filename
    ORIGINAL = "Original"
    DERIVED = "Derived"
    #validataion(filename)  # voyage date check
    # dms_dec=coordinate_conversion(filename)# convert dms to decimal
    # print(dms_dec)
    #coordinate_point_validation(filename)  # current data availablity check
    populate_weather_data(filename, filter_type, current_excluded, current_limit, bf_limit, swell_height_limit,
                          sig_wave_height_limit, wind_wave_height_limit)

    steaming_hr_1 = 0.0
    current_distance_1 = 0.0

    waranted_weather_yes_no = waranted_weather_yes_no_inp #'YES'  # j78    #about_s_and_c = "YES"/"NO"
    current_tolerance = current_tolerance_inp #  # G79             #About Speed Min Tolerence    #-0.58
    tolerance = tolerance_inp #  # I79                     #About Speed Max Tolerence
    mistolerance = mistolerance_inp  # K79                  #About Cons MinTolerence
    About_Cons_MaxTolerence = About_Cons_MaxTolerence_inp  # M79      #About Cons MaxTolerence
    not_sure_L78 = not_sure_L78_inp#  # L78               #Current Factor
    performance_calculation = performance_calculation_inp  # Voyage,Daily #p78
    extrapolation_Allowed = extrapolation_Allowed_inp  # O79 YES/NO
    adverse_current = adverse_current  # excluded/not_excluded       - adverse current G78
    cp_ordered_cons_go = 0  # this is user input, not from sheet
    GW_consumption_GO_added = "excluded"  # excluded/included   # This is done maually when needed. user input, not from sheet
    bad_weather_period_definition_as_per_CP_in_hrs = good_weather_hours_per_day_limit

    # phase 3 inputs
    prepared_basis = prepared_basis_inp  # manual , user input Optimal Speed/CP Speed
    constant_speed = True  # user input
    from_port = from_port_inp  # user input
    to_port = to_port_inp # user input

    co2_factor_hsfo = 3.114  # constant
    co2_factor_mdo = 3.206  # constant
    co2_factor_lng = 2.755  # constant

    fuel_type_used = fuel_type_used_inp  # "VULSFO"HSFO  # user input
    co2_factor = 3.114
    if (fuel_type_used == "MDO"):
        co2_factor = 3.206
    elif (fuel_type_used == "LNG"):
        co2_factor = 2.755

    import dataframe_image as dfi

    print(filename)
    filename = filename.split(".")[0] + ".csv"
    print(filename)

    wx = pd.read_csv('./results_output/' + filename)

    # ---new input --------

    for i in range(2, len(wx.columns)):

        steaming_hr = float(wx.iloc[5, i])
        print("steaming_hr", steaming_hr)
        current_distance = float(wx.iloc[26, i])
        print("current_distance", current_distance)

        steaming_hr_1 += steaming_hr
        current_distance_1 += current_distance
        print(steaming_hr, current_distance)

        if wx.iloc[8, i] == 'Original':
            # print("sum",steaming_hr,current_distance)
            wx.iloc[30, i] = current_distance_1 / steaming_hr_1
            print("original_value", wx.iloc[30, i])
            # print("sum",current_distance,steaming_hr,current_distance/steaming_hr)
            steaming_hr_1 = 0
            current_distance_1 = 0


    # print(wx)

    class gwx:
        '''Gwx calculation iterations'''

        def __init__(self, wx):
            '''Filter the required columns before processing'''
            self.df = wx
            print("7777" * 50)
            # print(self.df)
            # self.df=self.df.rename(columns=self.df.iloc[0]).drop(self.df.index[0]).T
            self.df = self.df.rename(columns=self.df.iloc[0]).T
            self.df = self.df.reset_index(drop=True)

            # print(self.df)

            # setting row 1 as header

            header_row = self.df.iloc[0]
            table_values = self.df.iloc[1:, ]
            # print(type(table_values))
            self.new_table = pd.DataFrame(table_values.values, columns=header_row)
            # print('new_table',self.new_table)

            # swapping arrival to last location

            #         last_row=self.new_table.iloc[1]

            #         print('printing last row',last_row)
            #         self.new_table.iloc[-1]=last_row
            # self.new_table.drop(1,axis=0,inplace=True)
            self.new_table.reset_index(drop=True, inplace=True)

            print('^^^' * 50)
            # self.new_table.drop()

            # print(self.new_table.info())

            self.new_table["Wind Force(BF Scale)"] = self.new_table["Wind Force(BF Scale)"].astype(int)
            self.new_table["Current factor"] = self.new_table["Current factor"].astype(float)
            self.new_table['windWaveHeight'] = self.new_table['windWaveHeight'].astype(float)
            self.new_table['swellHeight'] = self.new_table['swellHeight'].astype(float)
            self.new_table['waveHeight'] = self.new_table['waveHeight'].astype(float)

            df = pd.DataFrame()
            # print(self.new_table)

        def BF(self, gwx_hours, bf_limit):

            self.current_fac = gwx_type

            self.new_table = self.new_table.assign(
                Good_Wx_flag=np.where(self.new_table['Wind Force(BF Scale)'] <= bf_limit, 3, 0))

            # print("self.new_table",self.new_table['Good_Wx_flag'])
            return self.new_table

        def SWH(self, gwx_hours, bf_limit, swh_limit):

            self.current_fac = gwx_type

            self.new_table = self.new_table.assign(
                Good_Wx_flag=np.where(((self.new_table['Wind Force(BF Scale)'] <= bf_limit)
                                       & (self.new_table['waveHeight'] <= swh_limit)), 3, 0))

            # print("self.new_table_SWH",self.new_table)
            return self.new_table

        def DSS(self, gwx_hours, bf_limit, swh_limit, windwave_limit):

            self.current_fac = gwx_type

            self.new_table = self.new_table.assign(
                Good_Wx_flag=np.where(((self.new_table['Wind Force(BF Scale)'] <= bf_limit)
                                       & (self.new_table['windWaveHeight'] <= windwave_limit)
                                       & (self.new_table['swellHeight'] <= swell_height)), 3, 0))

            # print("self.new_table_DSS",self.new_table)
            return self.new_table

        def SWH_DSS(self, gwx_hours, bf_limit, swh_limit, windwave_limit):

            self.current_fac = gwx_type

            self.new_table = self.new_table.assign(
                Good_Wx_flag=np.where(((self.new_table['Wind Force(BF Scale)'] <= bf_limit)
                                       & (self.new_table['windWaveHeight'] <= windwave_limit)
                                       & (self.new_table['swellHeight'] <= swell_height)
                                       & (self.new_table['waveHeight'] <= swh_limit)), 3, 0))

            # print("self.new_table_SWH_DSS",self.new_table)
            return self.new_table

        def splitting_df(self):
            '''Splitting dataframe into individual units'''
            self.weather_table = self.new_table

            # print("self.weather_table_splitting_df",self.weather_table)

            # computer the bin size

            self.weather_table['Point Type'] == 'Original'
            bins = self.weather_table.index.values[self.weather_table['Point Type'] == 'Original']
            # print(self.weather_table)

            bins = bins.tolist()
            bins = [values + 1 for values in bins]
            # bins[0]=bins[0]
            bins.insert(0, 0)

            print('binsize', bins)
            # print('binslen',len(bins))

            # for i in np.arange(0,len(bins)):

            df_list = []
            report_type = ['BF', 'SWH', 'DSS', 'SWH_DSS']
            # report_type=['SWH']
            for j in report_type:

                self.weather_table['Report Type'] = j
                # print("confused",self.weather_table['Report Type'])
                for i in np.arange(len(bins) - 1):
                    print(",,," * 50)
                    print(i)
                    print('>>>' * 50)
                    # print('splitting_df',i,i+1)
                    # self.weather_table['df_current_factor_positive']=(all([i > 0 for i in self.weather_table['Current factor']]))
                    self.weather_table['df_current_factor_positive'] = (
                        all([i > 0 for i in self.weather_table['Current factor'].iloc[bins[i]:bins[i + 1]]]))
                    # print(self.weather_table['df_current_factor_positive'])
                    self.weather_table['df_current_factor_average'] = self.weather_table['Current factor'].iloc[
                                                                      bins[i]:bins[i + 1]].mean()
                    self.weather_table['df_current_factor_average_positive'] = True if self.weather_table[
                                                                                           'Current factor'].iloc[
                                                                                       bins[i]:bins[
                                                                                           i + 1]].mean() > 0 else False

                    current_df = self.weather_table.iloc[bins[i]:bins[i + 1]]
                    # print("current_df",current_df)

                    current_df['12_BF_inc'] = current_df['Good_Wx_flag']
                    current_df['15_BF_inc'] = current_df['Good_Wx_flag']
                    current_df['18_BF_inc'] = current_df['Good_Wx_flag']
                    current_df['24_BF_inc'] = current_df['Good_Wx_flag']

                    current_df['12_BF_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['15_BF_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['18_BF_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['24_BF_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_positive'] * \
                                            current_df['df_current_factor_average_positive']

                    current_df['12_SWH_inc'] = current_df['Good_Wx_flag']
                    current_df['15_SWH_inc'] = current_df['Good_Wx_flag']
                    current_df['18_SWH_inc'] = current_df['Good_Wx_flag']
                    current_df['24_SWH_inc'] = current_df['Good_Wx_flag']

                    current_df['12_SWH_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['15_SWH_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['18_SWH_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['24_SWH_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_positive'] * \
                                             current_df['df_current_factor_average_positive']

                    current_df['12_DSS_inc'] = current_df['Good_Wx_flag']
                    current_df['15_DSS_inc'] = current_df['Good_Wx_flag']
                    current_df['18_DSS_inc'] = current_df['Good_Wx_flag']
                    current_df['24_DSS_inc'] = current_df['Good_Wx_flag']

                    current_df['12_DSS_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['15_DSS_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['18_DSS_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_average_positive']
                    current_df['24_DSS_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_positive'] * \
                                             current_df['df_current_factor_average_positive']

                    current_df['12_SWH_DSS_inc'] = current_df['Good_Wx_flag']
                    current_df['15_SWH_DSS_inc'] = current_df['Good_Wx_flag']
                    current_df['18_SWH_DSS_inc'] = current_df['Good_Wx_flag']
                    current_df['24_SWH_DSS_inc'] = current_df['Good_Wx_flag']

                    current_df['12_SWH_DSS_x'] = current_df['Good_Wx_flag'] * current_df[
                        'df_current_factor_average_positive']
                    current_df['15_SWH_DSS_x'] = current_df['Good_Wx_flag'] * current_df[
                        'df_current_factor_average_positive']
                    current_df['18_SWH_DSS_x'] = current_df['Good_Wx_flag'] * current_df[
                        'df_current_factor_average_positive']
                    current_df['24_SWH_DSS_x'] = current_df['Good_Wx_flag'] * current_df['df_current_factor_positive'] * \
                                                 current_df['df_current_factor_average_positive']

                    # print(type(current_df))
                    # print("confused1",current_df)
                    df_list.append(current_df)

            return df_list

        def Gwx_pairing(self, weather_table, gwx_hours, report_type, gwx_type, gwx_method):

            self.gwx_hours = gwx_hours
            self.report_type = report_type
            self.gwx_type = gwx_type
            self.gwx_method = gwx_method

            print(self.gwx_hours, self.report_type, self.gwx_type)

            self.filter = str(gwx_hours) + "_" + str(report_type) + "_" + str(gwx_type)
            # print(self.filter)

            print('GWX Pairing')
            self.df = weather_table

            if self.gwx_hours == 12:

                for k in range(len(self.df)):

                    # print('printing k,filtering only the required columns-12',k)
                    i_df = self.df[k]
                    # print('length of id-12',len(i_df),i_df)

                    for m in range(len(i_df)):

                        if m + 4 <= len(i_df):

                            if self.gwx_method == 'all':
                                if i_df[self.filter].sum() >= 12:
                                    self.df[k]['GWD'] = 'GWD'
                                else:
                                    self.df[k]['GWD'] = 'NO'

                            elif self.gwx_method == 'consecutive':

                                print('Individual rows printing')
                                s1 = i_df.iloc[m][self.filter]
                                s2 = i_df.iloc[m + 1][self.filter]
                                s3 = i_df.iloc[m + 2][self.filter]
                                s4 = i_df.iloc[m + 3][self.filter]

                                print(s1, s2, s3, s4)

                                if (s1 + s2 + s3 + s4) == 12:
                                    self.df[k]['GWD'] = 'YES'
                                    print("GWD")
                                    break
                                else:
                                    self.df[k]['GWD'] = 'NO'
                                    print("NO_GWD")

                    # print(i_df)

            if self.gwx_hours == 15:

                for k in range(len(self.df)):
                    # print('printing k,filtering only the required columns',k)
                    i_df = self.df[k]
                    # print('length of id',len(i_df),i_df)

                    for m in range(len(i_df)):

                        if m + 5 <= len(i_df):

                            if self.gwx_method == 'all':

                                if i_df[self.filter].sum() >= 15:
                                    self.df[k]['GWD'] = 'GWD'
                                else:
                                    self.df[k]['GWD'] = 'NO'

                            elif self.gwx_method == 'consecutive':

                                print('Individual rows printing')
                                s1 = i_df.iloc[m][self.filter]
                                s2 = i_df.iloc[m + 1][self.filter]
                                s3 = i_df.iloc[m + 2][self.filter]
                                s4 = i_df.iloc[m + 3][self.filter]
                                s5 = i_df.iloc[m + 4][self.filter]

                                print(s1, s2, s3, s4, s5)

                                if (s1 + s2 + s3 + s4 + s5) == 15:
                                    self.df[k]['GWD'] = 'YES'
                                    print("GWD")
                                    break
                                else:
                                    self.df[k]['GWD'] = 'NO'
                                    print("NO_GWD")

                                #                 print(i_df)

            if self.gwx_hours == 18:

                for k in range(len(self.df)):
                    # print('printing k,filtering only the required columns',k)
                    i_df = self.df[k]
                    # print('length of id',len(i_df),i_df)

                    for m in range(len(i_df)):

                        if m + 6 <= len(i_df):

                            if self.gwx_method == 'all':

                                if i_df[self.filter].sum() >= 18:
                                    self.df[k]['GWD'] = 'GWD'
                                else:
                                    self.df[k]['GWD'] = 'NO'

                            elif self.gwx_method == 'consecutive':

                                print('Individual rows printing')
                                s1 = i_df.iloc[m][self.filter]
                                s2 = i_df.iloc[m + 1][self.filter]
                                s3 = i_df.iloc[m + 2][self.filter]
                                s4 = i_df.iloc[m + 3][self.filter]
                                s5 = i_df.iloc[m + 4][self.filter]
                                s6 = i_df.iloc[m + 5][self.filter]

                                print(s1, s2, s3, s4, s5, s6)

                                if (s1 + s2 + s3 + s4 + s5 + s6) == 18:
                                    self.df[k]['GWD'] = 'YES'
                                    print("GWD")
                                    break
                                else:
                                    self.df[k]['GWD'] = 'NO'
                                    print("NO_GWD")

            if self.gwx_hours == 24:

                for k in range(len(self.df)):
                    # print('printing k,filtering only the required columns',k)
                    i_df = self.df[k]
                    # print('length of id',len(i_df),i_df)

                    for m in range(len(i_df)):

                        if m + 8 <= len(i_df):

                            if self.gwx_method == 'all':

                                if i_df[self.filter].sum() >= 24:
                                    self.df[k]['GWD'] = 'GWD'
                                else:
                                    self.df[k]['GWD'] = 'NO'

                            elif self.gwx_method == 'consecutive':

                                print('Individual rows printing')
                                s1 = i_df.iloc[m][self.filter]
                                s2 = i_df.iloc[m + 1][self.filter]
                                s3 = i_df.iloc[m + 2][self.filter]
                                s4 = i_df.iloc[m + 3][self.filter]
                                s5 = i_df.iloc[m + 4][self.filter]
                                s6 = i_df.iloc[m + 5][self.filter]
                                s7 = i_df.iloc[m + 6][self.filter]
                                s8 = i_df.iloc[m + 7][self.filter]

                                print(s1, s2, s3, s4, s5, s6, s7, s8)

                                if (s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8) == 24:
                                    self.df[k]['GWD'] = 'YES'
                                    print("GWD")
                                    break
                                else:
                                    self.df[k]['GWD'] = 'NO'
                                    print("NO_GWD")

            #                         if m+8<=len(i_df):

            #                             print('Individual rows printing')
            #                             s1=i_df.iloc[m][self.filter]
            #                             s2=i_df.iloc[m+1][self.filter]
            #                             s3=i_df.iloc[m+2][self.filter]
            #                             s4=i_df.iloc[m+3][self.filter]
            #                             s5=i_df.iloc[m+4][self.filter]
            #                             s6=i_df.iloc[m+5][self.filter]
            #                             s7=i_df.iloc[m+6][self.filter]
            #                             s8=i_df.iloc[m+7][self.filter]

            #                             print(s1,s2,s3,s4,s5,s6,s7,s8)

            #                             if (s1+s2+s3+s4+s5+s6+s7+s8)==24:
            #                                 self.df[k]['GWD']='GWD'
            #                                 print("GWD")
            #                                 break
            #                             else:
            #                                 self.df[k]['GWD']='NO'
            #                                 print("NO_GWD")

            #                 #print(i_df)

            return i_df

        def report_type_df(self, weather_table):

            self.weather_table = weather_table

            dfs = []
            for name in self.weather_table:
                dfs.append(name)
                pre_final_df = pd.concat(dfs, ignore_index=True)

            return pre_final_df

        def filter_report_type(self, df_report_type, report_type):
            self.df = df_report_type
            final_df = self.df[self.df['Report Type'] == report_type]
            # print("selected report",final_df)
            return final_df

        def weather_report_formating(self, final_df):
            self.final_df = final_df

            print("---BELOW IS FINAL DF BEFORE PROCESSING PHASE-2" * 10)

            columns = ['Date', 'Time ( UTC)', 'Latitude', 'Longitude', 'Steaming Distance Since Last Noon Report',
                       'Steaming Hours Since Last noon Report',
                       'Average Speed Since Last noon Report',
                       'Average Ships Course since Last Report', 'Point Type',
                       'swellDirection', 'swellHeight', 'waveDirection', 'waveHeight',
                       'windDirection', 'Wind Force(BF Scale)', 'windSpeed', 'airTemperature',
                       'waterTemperature', 'pressure', 'precipitation', 'visibility',
                       'windWaveDirection', 'windWaveHeight', 'currentSpeed',
                       'currentDirection', 'Current factor', 'Current Distance', 'Time Period',
                       'Good Weather', 'Total Good Weather Period', 'Daily Current Factor', 'GWD']

            # final_df=final_df[columns].T  # performance report output
            self.final_df = self.final_df[columns]
            self.final_df = self.final_df.reset_index(drop=True)

            return self.final_df


    class sealog_abstract:

        def __init__(self, wx, processed_file, cp_ordered_cons_go, bf_limit):
            '''Process sealog abstract'''
            print('The below 2 files are used for processing in phase 2')
            # print(wx)

            print("is this the processed file")
            # print(processed_file)

            self.cp_ordered_cons_go = cp_ordered_cons_go

            self.green_processed_file = processed_file

            self.green_wx = wx.set_index('Unnamed: 0').T
            self.green_wx = self.green_wx[self.green_wx.loc[:, 'Point Type'] == 'Original'].reset_index(drop=True)
            self.green_wx = self.green_wx
            self.bf_limit = bf_limit

            # adding the required rows(example B100) which is required for future processing

            self.green_processed_file['Actual Total Consumption FO'] = self.green_processed_file['HSFO ME Cons.'] + \
                                                                       self.green_processed_file['HSFO AE Cons.'] + \
                                                                       self.green_processed_file['LSFO ME Cons.'] + \
                                                                       self.green_processed_file['LSFO AE Cons.'] + \
                                                                       self.green_processed_file['MGO ME Cons.']
            self.green_processed_file[
                'C/P Ordered Consumption GO(AE)'] = self.cp_ordered_cons_go  # This was not given in the input file and not sure how to include this.. so hardcoded it

            print('formated are the below 2 files are used for processing in phase 2')
            # print(self.green_processed_file,self.green_wx)

        def gwx_daily_find(self, weather_report):
            '''process the daily aggregate gwx summary that is to be used in the sealog abstract input'''
            self.gwx_day = weather_report
            # print('gwx_day',self.gwx_day)

            index_range = self.gwx_day[self.gwx_day['Point Type'] == 'Original'].index + 1
            index_range = index_range.insert(0, 0)  # index range to aggregate individual day good weather day status
            gw_period_column = self.gwx_day.iloc[:, -1]
            print('All the individual dataframe below are used to get the aggregate of the particular day weather status')

            gwd_agg_arr = []
            for i in range(len(index_range) - 1):
                print(index_range[i], index_range[i + 1])
                gwd_agg_arr.append(self.gwx_day.iloc[index_range[i]:index_range[i + 1]]['GWD'].max())
                print(gwd_agg_arr)

            print("INDEX_RANGE", index_range)

            return gwd_agg_arr, gw_period_column

        def green_table(self,
                        waranted_weather_yes_no,
                        current_tolerance,
                        tolerance,
                        mistolerance,
                        gwx_day_summary,
                        About_Cons_MaxTolerence,
                        not_sure_L78):

            # print("3333333567",self.green_processed_file)

            self.waranted_weather_yes_no = waranted_weather_yes_no  # j78
            self.current_tolerance = current_tolerance  # G79
            self.tolerance = tolerance  # I79
            self.mistolerance = mistolerance  # K79
            self.About_Cons_MaxTolerence = About_Cons_MaxTolerence  # M79
            self.not_sure_L78 = not_sure_L78  # L78

            self.gwx_day_summary = gwx_day_summary
            # print("self.gwx_day_summary",self.gwx_day_summary)

            # print("green_wx_1",self.green_wx)

            G79 = self.current_tolerance
            I79 = self.tolerance
            K79 = mistolerance / 100
            M79 = self.About_Cons_MaxTolerence / 100

            L78 = self.not_sure_L78

            B13 = self.green_processed_file['Observed distance (NM)'].astype(float)
            B14 = self.green_processed_file['Steaming Time (Hrs)'].astype(float)
            B9 = self.green_processed_file['Ordered Speed (Kts)'].astype(float)
            B10 = self.green_processed_file['CP Consumptions'].astype(float)
            # B96=self.green_wx['Current factor'].astype(float)
            B96 = self.green_wx['Daily Current Factor'].astype(float)
            print('current factor_B96', B96)
            B100 = self.green_processed_file['Actual Total Consumption FO'].astype(float)
            B11 = self.green_processed_file['C/P Ordered Consumption GO(AE)']
            print("B11", B11)
            print('B13', B13)
            print('B9', B9)
            print('G79', G79)

            self.green_processed_file['Performance Distance'] = B13 - B14 * B96

            # print("performancec distance",self.green_processed_file)

            # self.green_processed_file['Current Distance']=self.green_wx['Current Distance']
            self.green_processed_file['Current Distance'] = B96 * B14

            # self.green_processed_file['Current factor_added_later']=self.green_wx['Current factor']
            self.green_processed_file['Current factor_added_later'] = self.green_wx['Daily Current Factor']
            self.green_processed_file['Sea Currents (kts)'] = self.green_processed_file['Sea Currents (kts)_forc_wx']

            self.green_processed_file['Max TCP Allowed Time'] = np.where(self.waranted_weather_yes_no == 'YES',
                                                                         B13 / (B9 - G79),
                                                                         B13 / B9)

            self.green_processed_file['Max TCP Allowed Time with  Current Factor'] = np.where(
                self.waranted_weather_yes_no == 'YES',
                (B13 - B14 * B96) / (B9 - G79),
                (B13 - B14 * B96) / (B9))

            self.green_processed_file['Min TCP Allowed Time'] = np.where(self.waranted_weather_yes_no == 'YES',
                                                                         (B13 / (B9 + I79)),
                                                                         B13 / B9)

            self.green_processed_file['Min TCP Allowed Time with  Current Factor'] = np.where(
                self.waranted_weather_yes_no == 'YES',
                ((B13 - B14 * B96) / (B9 + I79)), (B13 - B14 * B96) / B9)

            self.green_processed_file['Max TCP Allowed FO Consumption'] = np.where(self.waranted_weather_yes_no == 'YES',
                                                                                   ((B13 / B9 / 24) * B10 * (1 + K79)),
                                                                                   (B13 / B9 / 24) * B10)

            self.green_processed_file['Max TCP Allowed  FO Cons. with  Current Factor'] = np.where(
                self.waranted_weather_yes_no == 'YES',
                ((B13 - B14 * B96) / B9 / 24) * B10 * (1 + K79),
                ((B13 - B14 * B96) / B9 / 24) * B10)

            self.green_processed_file['GWD_agg'] = self.gwx_day_summary
            # print("self.green_processed_file['GWD_agg']",self.green_processed_file['GWD_agg'])

            self.green_processed_file['Min TCP Allowed FO Consumption'] = np.where(self.waranted_weather_yes_no == 'YES',
                                                                                   (B13 / B9 / 24) * B10 * (1 - M79),
                                                                                   (B13 / B9 / 24) * B10)

            self.green_processed_file['Min TCP Allowed FO Consumption with  Current Factor'] = np.where(
                self.waranted_weather_yes_no == 'YES',
                (((B13 - B14 * B96) / B9 / 24) * B10 * (1 - M79)),
                ((B13 - B14 * B96) / B9 / 24) * B10)

            self.green_processed_file['Max TCP Allowed GO Consumption'] = np.where(self.waranted_weather_yes_no == 'YES',
                                                                                   ((B13 / B9 / 24) * B11 * (1 + K79)),
                                                                                   (B13 / B9 / 24) * B11)

            self.green_processed_file['Max TCP Allowed  GO Cons. with  Current Factor'] = np.where(
                self.waranted_weather_yes_no == 'YES',
                ((B13 - B14 * B96) / B9 / 24) * B11 * (1 + K79),
                ((B13 - B14 * B96) / B9 / 24) * B11)

            self.green_processed_file['Actual Total Consumption  GO'] = self.green_processed_file['MGO AE Cons.']
            B101 = self.green_processed_file['Actual Total Consumption  GO']

            self.green_processed_file['Min TCP Allowed GO Consumption'] = np.where(self.waranted_weather_yes_no == 'YES',
                                                                                   (B13 / B9 / 24) * B11 * (1 - M79),
                                                                                   (B13 / B9 / 24) * B11)
            B113 = self.green_processed_file['Min TCP Allowed GO Consumption']

            self.green_processed_file['Min TCP Allowed GO Consumption with  Current Factor'] = np.where(
                self.waranted_weather_yes_no == 'YES',
                (((B13 - B14 * B96) / B9 / 24) * B11 * (1 - M79)),
                ((B13 - B14 * B96) / B9 / 24) * B11)

            B115 = self.green_processed_file['Min TCP Allowed GO Consumption with  Current Factor']

            B104 = self.green_processed_file['Max TCP Allowed Time']
            # print("I104",B104)

            B105 = self.green_processed_file['Max TCP Allowed Time with  Current Factor']
            B106 = self.green_processed_file['Min TCP Allowed Time']
            B107 = self.green_processed_file['Min TCP Allowed Time with  Current Factor']
            B110 = self.green_processed_file['Max TCP Allowed  FO Cons. with  Current Factor']
            B108 = self.green_processed_file['Max TCP Allowed FO Consumption']
            B112 = self.green_processed_file['Min TCP Allowed FO Consumption']
            B114 = self.green_processed_file['Min TCP Allowed FO Consumption with  Current Factor']
            print("B112", B112)
            print("B114", B114)

            self.green_processed_file['Time Gain'] = np.where(((self.waranted_weather_yes_no == 'YES') & (B107 > B14)),
                                                              B107 - B14,
                                                              np.where(
                                                                  ((self.waranted_weather_yes_no == 'NO') & (B106 > B14)),
                                                                  B106 - B14, 0))
            self.green_processed_file['Time Loss'] = np.where(((self.waranted_weather_yes_no == 'YES') & (B14 > B105)),
                                                              B14 - B105,
                                                              np.where(
                                                                  ((self.waranted_weather_yes_no == 'NO') & (B14 > B104)),
                                                                  B14 - B104, 0))
            self.green_processed_file['Fuel oil Loss'] = np.where(((self.waranted_weather_yes_no == 'YES') & (B100 > B110)),
                                                                  B100 - B110,
                                                                  np.where(((self.waranted_weather_yes_no == 'NO') & (
                                                                              B100 > B108)), B100 - B108, 0))
            self.green_processed_file['Fuel  Oil Gain'] = np.where(
                ((self.waranted_weather_yes_no == 'YES') & (B114 > B100)), B114 - B100,
                np.where(((self.waranted_weather_yes_no == 'NO') & (B112 > B100)), B112 - B100, 0))
            self.green_processed_file['Gas oil Loss'] = np.where(((self.not_sure_L78 == 'NO') & (B114 > B100)), B114 - B100,
                                                                 np.where(((self.not_sure_L78 == 'NO') & (B112 > B100)),
                                                                          B112 - B100, 0))

            self.green_processed_file['Gas oil Gain'] = np.where(((self.not_sure_L78 == 'NO') & (B115 > B101)), B115 - B101,
                                                                 np.where(((self.not_sure_L78 == 'NO') & (B113 > B101)),
                                                                          B113 - B101, 0))

            # print("3333333568",self.green_processed_file)

            return self.green_processed_file

        def report_analysis_summary_table(self, green_processed_file):
            """first table in report_analysis_summary page"""
            print("first table in report_analysis_summary page")

            style1 = [{'selector': 'th',
                       'props': [('text-align', 'center'),
                                 ('background-color', '#a8d7c5'),
                                 ('color', 'black'),
                                 ('font-size', '12px'),
                                 # ('max-width', '80px'),
                                 ('border-style', 'solid'),
                                 ('border-width', '1px'),
                                 ('font-weight', 'normal'),
                                 ('font-family', 'play'),
                                 ]
                       },

                      {'selector': 'td',
                       'props': [('text-align', 'center'),
                                 ('border-style', 'solid'),
                                 ('border-width', '1px'),
                                 ('font-size', '12px'),
                                 ('border-color', 'black')]
                       },

                      {"selector": "tr", "props": "line-height: 14px;"},
                      {"selector": "td,th", "props": "line-height: inherit; padding: 6;"},
                      ]

            # print(green_processed_file)

            column_names = ['ATD(z)', 'Time gain/loss', 'V/U/L SFO gain/loss', 'HSFO gain/loss', 'MGO gain/loss',
                            'MDO gain/loss']

            # dfi.export(df,"report_analysis_summary.png")

        def report_analysis_summary_voyage_details(self, green_processed_file, from_port, to_port, adverse_current):
            ''' dont return , just use this for exporting table'''
            print("dont return , just use this for exporting table")
            # print(green_processed_file)
            # print(green_processed_file.loc[0,'Date'])
            # print(green_processed_file.loc[len(green_processed_file)-1,'Date'])
            self.adverse_current = adverse_current

            header = {"selector": "thead", "props": "background-color:#a8d7c5; color:black;"}
            text = {"selector": "th",
                    "props": "text-align:center;border-style:solid;border-width:1px;font-weight:normal;font-family:play;max-width:45px"}
            row = {"selector": ".row1", "props": "background-color:#a8d7c5; color:black;max-width:45px"}
            column_index = pd.MultiIndex.from_tuples([("Leg Details", ""),
                                                      ("ATD(Z)", ""),
                                                      ("ETA(Z)", ""),
                                                      ("Good Weather", "Distance"),
                                                      ("Good Weather", "Steaming\n Hours"),
                                                      ("Good Weather", "Speed"),
                                                      ("Good Weather", "Total Cons"),
                                                      ("Performance", "Distance\n(Exc currents)"),
                                                      ("Performance", "Speed"),
                                                      ("Overall Weather", "Distance"),
                                                      ("Overall Weather", "Steaming\n Hours"),
                                                      ("Overall Weather", "Speed"),
                                                      ("Overall Weather", "Total Cons.")])

            style1 = [{'selector': 'th',
                       'props': [('text-align', 'center'),
                                 ('background-color', '#a8d7c5'),
                                 ('color', 'black'),
                                 ('font-size', '24px'),
                                 ('max-width', '100px'),
                                 ('border-style', 'solid'),
                                 ('border-width', '2px'),
                                 ('font-weight', 'normal'),
                                 ('font-family', 'play'),
                                 ]
                       },

                      {'selector': 'td',
                       'props': [('text-align', 'center'),
                                 ('border-style', 'solid'),
                                 ('border-width', '2px'),
                                 ('font-size', '24px'),
                                 ('border-color', 'black')
                                 ]
                       },

                      {"selector": "tr", "props": "line-height: 24px;"},
                      # {"selector": "tr", "props": [('line-height': '24px'),('background-color', '#a8d7c5')]},
                      {"selector": "td,th,tr", "props": "line-height: inherit; padding: 15;"},
                      ]

            print("hi_aa", green_processed_file['Actual Total Consumption  GO'])

            # Overal weather
            overal_weather_distance = green_processed_file['Observed distance (NM)'].sum()
            # overal_weather_steaming_time=green_processed_file['Steaming Time (Hrs)'].sum().round(2)
            overal_weather_steaming_time = green_processed_file['Steaming Time (Hrs)'].sum()
            overal_weather_Average_speed = (green_processed_file['Observed distance (NM)'].sum() / green_processed_file[
                'Steaming Time (Hrs)'].sum()).round(2)

            print("print_ee",
                  green_processed_file['Observed distance (NM)'].sum() / green_processed_file['Steaming Time (Hrs)'].sum())
            print("total_cons_at_sea", green_processed_file['Actual Total Consumption FO'].sum(),
                  green_processed_file['Actual Total Consumption  GO'].sum())

            # overal_weather_Actual_Total_Consumption_FO=green_processed_file['Actual Total Consumption FO'].sum()+green_processed_file['Actual Total Consumption  GO'].sum()
            overal_weather_Actual_Total_Consumption_FO = green_processed_file['Actual Total Consumption FO'].sum() + \
                                                         green_processed_file['MGO ME Cons.'].sum() + green_processed_file[
                                                             'MGO AE Cons.'].sum() + green_processed_file[
                                                             'MGO Boiler'].sum() + green_processed_file['MGO Others'].sum()

            # Good weather

            conds = green_processed_file["GWD_agg"].isin(['YES'])
            green_processed_file_temp = green_processed_file.loc[conds]
            # print("green_processed_file_temp",green_processed_file_temp)

            green_processed_file_temp_distance = green_processed_file_temp['Observed distance (NM)'].sum()
            green_processed_file_temp_steaming_time = green_processed_file_temp['Steaming Time (Hrs)'].sum()
            green_processed_file_temp_average_speed = green_processed_file_temp['Average speed (Kts)'].mean()

            if math.isnan(green_processed_file_temp_average_speed):
                print("yes its is nan")
                green_processed_file_temp_average_speed = 0

            # print("green_processed_file_temp_average_speed",green_processed_file_temp_average_speed)

            green_processed_file_temp_Actual_Total_Consumption_FO = green_processed_file_temp['Actual Total Consumption FO'].sum()
            green_processed_file_temp_perform_distance = green_processed_file_temp['Performance Distance'].sum()

            temp_a = green_processed_file_temp["Observed distance (NM)"].sum()
            temp_b = green_processed_file_temp["Steaming Time (Hrs)"].sum()

            # green_processed_file_temp_average_performance_speed = green_processed_file_temp_perform_distance/green_processed_file_temp_steaming_time

            if adverse_current == "excluded":
                green_processed_file_temp_perform_distance = green_processed_file_temp['Observed distance (NM)'].sum()
                green_processed_file_temp_average_performance_speed = green_processed_file_temp_average_speed
            #             if (math.isnan(green_processed_file_temp_average_speed)):
            #                 green_processed_file_temp_average_performance_speed=0.0
            #             else:

            #                 green_processed_file_temp_average_performance_speed = green_processed_file_temp_average_speed

            # print("green_processed_file_temp_average_performance_speed",green_processed_file_temp_average_performance_speed)

            else:
                green_processed_file_temp_perform_distance = green_processed_file_temp['Performance Distance'].sum()
                green_processed_file_temp_average_performance_speed = green_processed_file_temp_perform_distance / green_processed_file_temp_steaming_time

            data = np.array([[from_port + ' to ' + to_port,
                              green_processed_file.loc[0, 'Date'],
                              green_processed_file.loc[len(green_processed_file) - 1, 'Date'],
                              str(green_processed_file_temp_distance.round(2)),
                              str(round(green_processed_file_temp_steaming_time, 2)),
                              str(round(green_processed_file_temp_average_speed, 2)),
                              str(green_processed_file_temp_Actual_Total_Consumption_FO),
                              str(green_processed_file_temp_perform_distance.round(2)),
                              str(round(green_processed_file_temp_average_performance_speed, 2)),
                              # dont know what this is.. domething was here else
                              str(overal_weather_distance.round(2)),
                              str(round(overal_weather_steaming_time, 2)),
                              str(overal_weather_Average_speed.round(2)),
                              str(overal_weather_Actual_Total_Consumption_FO.round(2))]])

            df = pd.DataFrame(data=data, columns=column_index)

            print("what is this df",df.columns[:3])
            #df.loc["Total"] = df.sum(axis=0)
            df.loc["Total"]=df.sum(axis=0)

            #df.loc['Total', ('ETA(Z)', '')]=np.nan

            df.loc['Total', [('Leg Details', ''),
                (     'ATD(Z)', ''),
                (     'ETA(Z)', '')]] = np.nan

            print('look',df.loc['Total',('ETA(Z)', '')])

            #df.append(df.sum(numeric_only=True), ignore_index=True)
            #print('printing index', df.index)
            df = df.fillna('').style. \
                set_properties(subset=pd.IndexSlice[['Total'], :], **{'background-color': '#a8d7c5'}). \
                set_properties(subset=pd.IndexSlice[[0], :], **{'background-color': '#FFFFFF'}). \
                set_table_styles(style1).hide()

            dfi.export(df, "./png/report_analysis_summary_voyage_details.png")

        def voyage_detail(self, green_processed_file):

            print(green_processed_file.columns,green_processed_file.info())


            print(green_processed_file[['Steaming Time (Hrs)','Actual Total Consumption FO']])

            print('below 2 dataframes for building voyage details by combining the required columns.. This should match the excel table under voyage details')


            #green_processed_file['Actual Cons for 24 hrs'] = np.where(green_processed_file["Steaming Time (Hrs)"] == 0, 0.0, 24 * green_processed_file["Actual Total Consumption FO"]/green_processed_file["Steaming Time (Hrs)"])
            green_processed_file['Actual Cons for 24 hrs'] = np.where(green_processed_file['Steaming Time (Hrs)'] == 0, 0,
                                                    (24 * green_processed_file['Actual Total Consumption FO'].astype(float) / green_processed_file['Steaming Time (Hrs)'].astype(float)))



            green_processed_file['Actual Cons for 24 hrs'] = np.where(green_processed_file["Steaming Time (Hrs)"]==0,0,10)
            print(green_processed_file['Actual Cons for 24 hrs'])

            green_processed_file["Difference1"] = green_processed_file["Ordered Speed (Kts)"] - green_processed_file["Average speed (Kts)"]
            green_processed_file["Difference1%"] = green_processed_file["Difference1"] / green_processed_file["Ordered Speed (Kts)"]
            green_processed_file["Instruction Followed1"] = np.where(green_processed_file['Ordered Speed (Kts)'] == 0,"N/A", np.where((green_processed_file["Difference1"] > 0.5) | (green_processed_file["Difference1"] < -0.5), "NO","YES"))

            green_processed_file["Difference2"] = green_processed_file["CP Consumptions"] - green_processed_file["Actual Cons for 24 hrs"]
            green_processed_file["Difference2%"] = green_processed_file["Difference2"] / green_processed_file["CP Consumptions"]
            # green_processed_file["Instruction Followed2"]='NO'

            green_processed_file["Instruction Followed2"] = np.where(green_processed_file['Ordered Speed (Kts)'] == 0,"N/A", np.where((green_processed_file["Difference2%"] > 0.05) | (green_processed_file["Difference2%"] < -0.05), "NO","YES"))

            green_processed_file["Good3"] = 'N/A'
            green_processed_file["Good4"] = 'N/A'
            self.green_wx["Good5"] = self.green_processed_file['GWD_agg']

            green_processed_file = green_processed_file.replace(np.nan, 0)

            voyage_details_sub_table = green_processed_file[["Date", "Ship's Time (UTC)", "Latitude", "Longitude",
                                                             "Steaming Time (Hrs)", "Max TCP Allowed Time",
                                                             "Ordered Speed (Kts)", "Average speed (Kts)", "Difference1",
                                                             "Difference1%", "Instruction Followed1",
                                                             "CP Consumptions", "Actual Total Consumption FO",
                                                             "Actual Cons for 24 hrs", "Difference2", "Difference2%",
                                                             "Instruction Followed2",
                                                             "True Wind Force (BF)_forc_wx",
                                                             'Significant Wave Height_forc_wx', 'Swell height (m)_forc_wx',
                                                             "Good3",
                                                             'True Wind Force (BF)', 'Wind Sea Height (m)',
                                                             'Swell height (m)', "Good4"]]

            voyage_details_sub_table2 = self.green_wx[["Wind Force(BF Scale)", "swellHeight", "waveHeight", "Good5"]]  # "Ship's Time (UTC)"

            voyage_detail_table = pd.concat([voyage_details_sub_table, voyage_details_sub_table2], axis=1)

            print('above 2 tables needs to be combined to form voyage details table')

            for m in range(len(voyage_detail_table)):
                voyage_detail_table["Date"][m] = str(voyage_detail_table["Date"][m]) + " " + str(
                    voyage_detail_table["Ship's Time (UTC)"][m])

            return voyage_detail_table

        def beautifying_voyage_detail_table(self, voyage_detail_table):

            voyage_detail_table['Latitude'] = voyage_detail_table['Latitude'].astype(float)
            voyage_detail_table['Longitude'] = voyage_detail_table['Longitude'].astype(float)

            voyage_detail_table["Date"] = pd.to_datetime(voyage_detail_table["Date"], format="%Y-%m-%d %H:%M:%S",
                                                         errors='coerce')

            def custom_date_format1(date):  # Define a custom function to format the date
                day = date.day
                suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
                return date.strftime(f"%d{suffix} %b %Y %H:%M")

            voyage_detail_table["Date"] = voyage_detail_table["Date"].apply(custom_date_format1)

            # print("beautifying_voyage_detail_table",voyage_detail_table)

            len_of_table = len(voyage_detail_table)
            print(len_of_table)
            records_per_page = 40
            no_of_page_voyage = len_of_table / records_per_page

            import math
            math.ceil(no_of_page_voyage)

            j = 0
            for i in range(1, int(math.ceil(no_of_page_voyage)) + 1):
                # print(j*5,i*5)
                voyage_detail_table = voyage_detail_table.iloc[j * records_per_page:i * records_per_page]
                voyage_detail_table['Good5'] = voyage_detail_table['Good5'].apply(lambda x: "NO" if x is np.nan else x)
                voyage_detail_table['Good4'] = voyage_detail_table['True Wind Force (BF)'].apply(
                    lambda x: "N/A" if x == 0 else ("YES" if x >= self.bf_limit else "NO"))
                voyage_detail_table = voyage_detail_table.drop(
                    ["Ship's Time (UTC)", 'True Wind Force (BF)_forc_wx', 'Significant Wave Height_forc_wx',
                     'Swell height (m)_forc_wx', 'Good3'], axis=1)
                #             voyage_detail_table["Date"] = pd.to_datetime(voyage_detail_table["Date"],errors='coerce')
                #             voyage_detail_table["Date"] = voyage_detail_table["Date"].apply(lambda x: x.strftime("%d/%m/%Y %H:%M"))
                print("This is final_voyage_detail_table")
                # print(voyage_detail_table)
                j = j + 1

                print('Table formating/beautification')

                tuple_list = [("Date", "", ""),
                              ("Position", "", "Lat"),
                              ("Position", "", "Lon"),
                              ("Predicted Time", "", ""),
                              ("Steaming Hours", "", "hrs"),
                              ("Speed", "Instructed SOG", "Knots"),
                              ("Speed", "Actual SOG", "Knots"),
                              ("Speed", "Difference", "Knots"),
                              ("Speed", "Difference", "%"),
                              ("Speed", "Instructions Followed", ""),
                              ("Consumption", "Instructed Consumption", "MT/d"),
                              ("Consumption", "Actual Consumption", "MT"),
                              ("Consumption", "Actual Cons for 24 hrs", "MT/d"),
                              ("Consumption", "Difference", "MT/d"),
                              ("Consumption", "Difference", "%"),
                              ("Consumption", "Instructions Followed", ""),
                              ("Master Reported Weather", "Wind", "BF"),
                              ("Master Reported Weather", "Wave Height (m)", "Wind"),
                              ("Master Reported Weather", "Wave Height (m)", "Swell"),
                              ("Master Reported Weather", "Good Weather", "Y/N"),
                              ("Analysed Weather", "Wind", "BF"),
                              ("Analysed Weather", "Wave Height (m)", "Wind"),
                              ("Analysed Weather", "Wave Height (m)", "Swell"),
                              ("Analysed Weather", "Good Weather", "Y/N")]

                index = pd.MultiIndex.from_tuples(tuple_list)

                style3 = [{'selector': 'th',
                           'props': [('text-align', 'center'),
                                     ('background-color', 'lightgrey'),
                                     ('color', 'darkblue'),
                                     ('font-size', '25px'),
                                     ('border-style', 'solid'),
                                     ('border-width', '1px'),
                                     ('font-weight', 'normal'),
                                     ('font-family', 'play')]
                           },

                          {'selector': 'td',
                           'props': [('text-align', 'center'),
                                     ('border-style', 'solid'),
                                     ('border-width', '1px'),
                                     ('font-size', '24px'),
                                     ('border-color', 'black'),
                                     ('font-family', 'play'),
                                     ('font-weight','normal')]
                           },

                          {"selector": "tr", "props": "line-height: 30px;"},
                          {"selector": "td,th", "props": "line-height: inherit; padding: 6;"}

                          ]

                # TEST---------------

                #             def add_color(x):
                #                 if x =='YES':
                #                     color='#BDEDFF'
                #                 elif x == 'NO':
                #                     color='#6667AB'
                #                 else:
                #                     color='#C5908E'
                #                 return f"background:{color}"

                def add_speed_color(x):
                    if ((x[("Speed", "Difference", "Knots")] >= -0.5) & (x[("Speed", "Difference", "Knots")] < 0)) or (
                            (x[("Speed", "Difference", "Knots")] <= 0.5) & (x[("Speed", "Difference", "Knots")] > 0)):
                        return ['', '', '', '', '', '', '', '', '', 'background-color: #DEEEE9', '', '', '', '', '', '', '',
                                '', '', '', '', '', '', '']

                    elif ((x[("Speed", "Difference", "Knots")] >= -1) & (x[("Speed", "Difference", "Knots")] <= -0.51)) or (
                            (x[("Speed", "Difference", "Knots")] <= 1) & (x[("Speed", "Difference", "Knots")] > 0.51)):
                        return ['', '', '', '', '', '', '', '', '', 'background-color: #FEE09A', '', '', '', '', '', '', '',
                                '', '', '', '', '', '', '']

                    elif (x[("Speed", "Difference", "Knots")] == 0):
                        return ['', '', '', '', '', '', '', '', '', 'background-color: #FFFFFF', '', '', '', '', '', '', '',
                                '', '', '', '', '', '', '']

                    else:
                        return ['', '', '', '', '', '', '', '', '', 'background-color: #C3687D', '', '', '', '', '', '', '',
                                '', '', '', '', '', '', '']

                def add_consumption_color(x):
                    if ((x[("Consumption", "Difference", "MT/d")] >= -5) & (
                            x[("Consumption", "Difference", "MT/d")] < 0)) or (
                            (x[("Consumption", "Difference", "MT/d")] <= 5) & (
                            x[("Consumption", "Difference", "MT/d")] > 0)):
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'background-color: #DEEEE9', '',
                                '', '', '', '', '', '', '']

                    elif ((x[("Consumption", "Difference", "MT/d")] >= -8) & (
                            x[("Consumption", "Difference", "MT/d")] <= -5.01)) or (
                            (x[("Consumption", "Difference", "MT/d")] <= 8) & (
                            x[("Consumption", "Difference", "MT/d")] > 5.01)):
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'background-color: #FEE09A', '',
                                '', '', '', '', '', '', '']

                    elif (x[("Consumption", "Difference", "MT/d")] == 0):
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'background-color: #FFFFFF', '',
                                '', '', '', '', '', '', '']

                    else:
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'background-color: #C3687D', '',
                                '', '', '', '', '', '', '']

                def add_weather_color(x):
                    if x[("Analysed Weather", "Good Weather", "Y/N")] == 'YES':
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                                'background-color: #DEEEE9']
                    elif x[("Analysed Weather", "Good Weather", "Y/N")] == 'NO':
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                                'background-color: #C3687D']
                    else:
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                                'background-color: #FFFFFF']

                def bf_limit_color(x):
                    if x[("Master Reported Weather", "Good Weather", "Y/N")] == 'N/A':
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                                'background-color: #FFFFFF', '', '', '', '']
                    elif x[("Master Reported Weather", "Good Weather", "Y/N")] == 'YES':
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                                'background-color: #DEEEE9', '', '', '', '']
                    else:
                        return ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                                'background-color: #C3687D', '', '', '', '']

                # original----
                df = pd.DataFrame(voyage_detail_table.values, columns=index).style.format(precision=2).apply(
                    add_speed_color, axis=1).apply(add_consumption_color, axis=1).apply(add_weather_color, axis=1).apply(
                    bf_limit_color, axis=1).set_table_styles(style3).hide()

                # Original-----

                #             df=pd.DataFrame(voyage_detail_table.values,columns=index).style.\
                #                             format({("Position","","Lon"):'{:.2f}'}).\
                #                             format({("Position","","Lat"):'{:.2f}'}).\
                #                             applymap(add_color,subset=[("Speed","Instructions Followed",""),("Consumption","Instructions Followed",""),("Analysed Weather","Good Weather","Y/N")]).\
                #                             set_table_styles(style3).hide_index()

                #                                             format({("Position","","Lon"):'{:.2f}'}).\
                #                                             format({("Position","","Lat"):'{:.2f}'}).\
                #                                                     highlight_max(subset=[("Position","","Lon")]).\
                #                                                 highlight_min(subset=[("Position","","Lon")],color='red').\

                # print(df)
                dfi.export(df, "./png/voyage_details_table" + str(i) + ".png")

        def voyage_summary(self, voyage_detail_table):
            print('Building Voyage Summary')
            voyage_summary = pd.DataFrame(voyage_detail_table.loc[:,
                                          ['Steaming Time (Hrs)', 'Max TCP Allowed Time', 'Difference1', 'CP Consumptions',
                                           'Actual Total Consumption FO', 'Difference2']].sum()).T.round(2)
            tuple_list1 = [('Predicted Time', 'Hrs'),
                           ('Actual Time', 'Hrs'),
                           ('Time Saved\Lost', 'Hrs'),
                           ('Predicted Total Voyage Consumption', 'MT'),
                           ('Actual Total Voyage Consumption', 'MT'),
                           ('Over/under Consumption', 'MT')]

            index1 = pd.MultiIndex.from_tuples(tuple_list1)

            df_voyage_summary = pd.DataFrame(voyage_summary.values, columns=index1).style.format(precision=2). \
                set_table_styles([{'selector': 'th',
                                   'props': [('text-align', 'center'), ('background-color', '#a8d7c5'), ('color', 'black'),
                                             ('max-width', '150px'),
                                             ('border-style', 'solid'), ('border-width', '2px'), ('font-weight', 'normal'),
                                             ('font-family', 'play'), ('font-size', '24px')]
                                   },

                                  {'selector': 'td',
                                   'props': [('text-align', 'center'), ('border-style', 'solid'), ('border-width', '2px'),
                                             ('font-size', '24px'), ('border-color', 'black')]
                                   }]
                                 ).hide()

            dfi.export(df_voyage_summary, './png/voyage_details_summary.png')

        def weather_detail(self, weather_report, green_processed_file):

            '''Processing weather details'''
            print('using the below 2 files to  process weather details')
            weather_report = weather_report.T
            print("--" * 20)
            columns = ["Date",
                       "Time ( UTC)",
                       "Latitude",
                       "Longitude",
                       "Wind Force(BF Scale)",
                       "windDirection",
                       "waveHeight",
                       "windWaveHeight",
                       "swellHeight",
                       "swellDirection",
                       "Current factor",
                       "Total Good Weather Period"]
            temp_weather_report = weather_report.loc[:, columns]

            columns1 = [
                "Date",
                "Ship's Time (UTC)",
                "Steaming Time (Hrs)",
                "Sea Currents (kts)",
                "Observed distance (NM)",
                "True Wind Force (BF)",
                "Ordered Speed (Kts)",
                "Average speed (Kts)",
                "Average Engine RPM",
                "Slip %",
                "Course"]

            temp_green_processed_file = green_processed_file.loc[:, columns1]

            print("after taking only the required columns for weather details - the below 2 tables")

            temp_weather_report['Date'] = pd.to_datetime(temp_weather_report['Date'], format='%d %b %Y')
            temp_weather_report['Date'] = temp_weather_report['Date'].apply(lambda x: x.strftime('%Y%m%d'))
            temp_weather_report['Time ( UTC)'] = pd.to_datetime(temp_weather_report['Time ( UTC)'], format="%H:%M:%S")
            temp_weather_report['new_timie'] = temp_weather_report['Time ( UTC)'].apply(lambda x: x.strftime("%H%M%S"))
            temp_weather_report['new_time'] = temp_weather_report['Date'] + temp_weather_report['new_timie']
            temp_weather_report['Latitude'] = temp_weather_report['Latitude'].astype(float)
            temp_weather_report['Longitude'] = temp_weather_report['Longitude'].astype(float)
            temp_green_processed_file['Date'] = temp_green_processed_file['Date'].apply(lambda x: x.strftime('%Y%m%d'))
            temp_green_processed_file["Ship's Time (UTC)"] = temp_green_processed_file["Ship's Time (UTC)"].apply(lambda x: x.strftime("%H%M%S"))
            temp_green_processed_file['new_time'] = temp_green_processed_file['Date'] + temp_green_processed_file["Ship's Time (UTC)"]
            temp_green_processed_file['Point Type'] = "Original"

            weather_details_print_table = temp_weather_report.merge(temp_green_processed_file, on='new_time', how="left")

            weather_details_print_table.drop(["Date_x", "Date_y", "Time ( UTC)"], axis=1, inplace=True)

            # converting the convereted time back and re arrangeing the coloumsn in order

            weather_details_print_table['new_time'] = pd.to_datetime(weather_details_print_table['new_time']).dt.strftime('%m/%d/%Y %H:%M')

            columns_list = ['new_time',
                            'Latitude',
                            'Longitude',
                            'Wind Force(BF Scale)',
                            'windDirection',
                            'waveHeight',
                            'windWaveHeight',
                            'swellHeight',
                            'swellDirection',
                            'Current factor',
                            'Total Good Weather Period',
                            'Steaming Time (Hrs)',
                            'Observed distance (NM)',
                            'True Wind Force (BF)',
                            'Sea Currents (kts)',
                            'Ordered Speed (Kts)',
                            'Average speed (Kts)',
                            'Average Engine RPM',
                            'Slip %',
                            'Course']

            # , this was removed from above column_list

            weather_details_print_table = weather_details_print_table[columns_list]

            # renaming the new_time columns

            weather_details_print_table.rename(columns={'new_time': 'Date/Time'}, inplace=True)

            print("after merging tables to get the desired table for prininting")

            # print(weather_details_print_table.columns)

            # print(weather_details_print_table)

            # print("is this string obj",weather_details_print_table['Date/Time'])

            weather_details_print_table['Date/Time'] = pd.to_datetime(weather_details_print_table['Date/Time'])
            print("what it is", weather_details_print_table['Date/Time'][0])

            weather_details_print_table['Date/Time'] = weather_details_print_table['Date/Time'].apply(
                lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M"))

            return weather_details_print_table

        def beautifying_weather_table(self, weather_detail_table, gw_period_column):
            '''1.beautifying the detail weather table  2.build weather summary'''

            weather_detail_table = weather_detail_table.join(gw_period_column)

            len_of_table = len(weather_detail_table)
            weather_detail_table.loc[weather_detail_table["Course"].isna(), ["Latitude", "Longitude"]] = ""
            weather_detail_table.loc[weather_detail_table["Course"].notna(), ["Total Good Weather Period"]] = ""

            weather_detail_table["Date/Time"] = pd.to_datetime(weather_detail_table["Date/Time"], format='%d/%m/%Y %H:%M', errors='coerce')

            def custom_date_format2(date):  # Define a custom function to format the date
                day = date.day
                suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
                return date.strftime(f"%d{suffix} %b %Y %H:%M")

            weather_detail_table["Date/Time"] = weather_detail_table["Date/Time"].apply(custom_date_format2)

            weather_detail_table = weather_detail_table.replace(np.nan, "")
            # print("formated_table",weather_detail_table)
            # ---

            print(len_of_table)
            records_per_page = 50
            no_of_page_weather = len_of_table / records_per_page

            # import math
            print("math.ceil", math.ceil(no_of_page_weather))

            j = 0
            for i in range(1, int(math.ceil(no_of_page_weather)) + 1):
                # print(j*5,i*5)
                print(j * records_per_page, i * records_per_page)
                new_weather_detail_table = weather_detail_table.iloc[j * records_per_page:i * records_per_page]
                # dfi.export(data,"test_voyage_detail"+str(i)+".png")
                j = j + 1

                tuple_list2 = [("Date/Time", ""),
                               ("Lat", ""),
                               ("Lon", ""),
                               ("Wind", "BFT"),
                               ("Wind", "Dir.(rel.)"),
                               ("SWH", "Hgt(m)"),
                               ("Wind Wave", "(m)"),
                               ("Swell", "Hgt (m)"),
                               ("Swell", "Dir. (rel.)"),
                               ("Current factor", "Kts"),
                               ("Bad Weather Details", ""),
                               ("Report Data by Ship", "Steaming Hours"),
                               ("Report Data by Ship", "Distance (NM)"),
                               ("Report Data by Ship", "Wind (Bft)"),
                               ("Report Data by Ship", "Current Factor (Kts)"),
                               ("Report Data by Ship", "Ordered Speed (Kts)"),
                               ("Report Data by Ship", "Avg. Speed (Kts)"),
                               ("Report Data by Ship", "RPM"),
                               ("Report Data by Ship", "Slip (%)"),
                               ("Report Data by Ship", "Course"),
                               ("Report Data by Ship","GWP")]

                index2 = pd.MultiIndex.from_tuples(tuple_list2)

                style1 = [{'selector': 'th',
                           'props': [('text-align', 'center'),
                                     ('background-color', '#aaffdc'),
                                     ('color', 'darkblue'),
                                     ('font-size', '25px'),
                                     ('border-style', 'solid'),
                                     ('border-width', '1px'),
                                     ('font-weight', 'normal'),
                                     ('font-family', 'play')]
                           },

                          {'selector': 'td',
                           'props': [('text-align', 'center'),
                                     ('border-style', 'solid'),
                                     ('border-width', '1px'),
                                     ('font-size', '24px'),
                                     ('border-color', 'black'),
                                     ('font-family','play'),
                                     ('font-weight','normal')]
                           },

                          {"selector": "tr", "props": "line-height: 30px;"},
                          {"selector": "td,th", "props": "line-height: inherit; padding: 6;"}

                          ]  # ('font-family','play'),('font-weight','normal')

                def fonts_bold(x):
                    if x[("Report Data by Ship", "Course")] != '':
                        return ['font-weight: bold'] * len(x)
                    else:
                        return [''] * len(x)

                def add_row_color(x):
                    if (x[-1] == 'YES'):
                        return ['background-color: #aaffdc'] * len(x)
                    elif (x[-1] == 'GWD'):
                        return ['background-color: #aaffdc'] * len(x)
                    else:
                        return ['background-color: white'] * len(x)

                df = pd.DataFrame(new_weather_detail_table.values, columns=index2).style. \
                    format(precision=2, na_rep='-').apply(fonts_bold, axis=1).apply(add_row_color,axis=1). \
                    set_table_styles(style1).hide_index().hide_columns([("Report Data by Ship","GWP")])

                dfi.export(df, "./png/weather_detail" + str(i) + ".png")

                # lightgrey

        def weather_summary(self, weather_report, green_processed_file):

            print("WEATHER SUMMARY PROCESS STARTS HERE")

            print('using the below 2 files to  process weather summary')
            weather_report = weather_report.T
            # print(weather_report)
            print("--" * 20)
            print("-green_processed_file-")
            # print(green_processed_file)

            columns = ["Date",
                       "Latitude",
                       "Longitude",
                       "Ship's Time (UTC)",
                       "Steaming Time (Hrs)",
                       "Max TCP Allowed Time",
                       "Min TCP Allowed FO Consumption",
                       "Observed distance (NM)",
                       "Average Engine RPM",
                       "Slip %",
                       "Course",
                       "True Wind Force (BF)",
                       "Wind Sea Height (m)",
                       "Sea Currents (kts)",
                       "HSFO ROB",
                       "HSFO ME Cons.",
                       "HSFO AE Cons.",
                       "HSFO Boiler",
                       "HSFO Others",
                       "LSFO ROB",
                       "LSFO ME Cons.",
                       "LSFO AE Cons.",
                       "LSFO Boiler",
                       "LSFO Others",
                       "MGO ROB",
                       "MGO ME Cons.",
                       "MGO AE Cons.",
                       "MGO Boiler",
                       "MGO Others",
                       "LNG ROB",
                       "LNG ME Cons.",
                       "LNG AE Cons.",
                       "LNG Boiler",
                       "LNG Others",
                       "Max TCP Allowed FO Consumption"]

            temp_green_processed_file = green_processed_file.loc[:, columns]
            temp_green_processed_file = temp_green_processed_file.assign(
                HSFO_Total=temp_green_processed_file["HSFO ME Cons."] + temp_green_processed_file["HSFO AE Cons."] +
                           temp_green_processed_file["HSFO Boiler"] + temp_green_processed_file["HSFO Others"],
                LSFO_Total=temp_green_processed_file["LSFO ME Cons."] + temp_green_processed_file["LSFO AE Cons."] +
                           temp_green_processed_file["LSFO Boiler"] + temp_green_processed_file["LSFO Others"],
                MGO_Total=temp_green_processed_file["MGO ME Cons."] + temp_green_processed_file["MGO AE Cons."] +
                          temp_green_processed_file["MGO Boiler"] + temp_green_processed_file["MGO Others"],
                LNG_Total=temp_green_processed_file["LNG ME Cons."] + temp_green_processed_file["LNG AE Cons."] +
                          temp_green_processed_file["LNG Boiler"] + temp_green_processed_file["LNG Others"]

            )

            # print(type(temp_green_processed_file))
            columns1 = ["Wind Force(BF Scale)", "waveHeight", "Current factor", "Point Type", "GWD"]
            temp_weather_report = weather_report.loc[:, columns1]

            mask = temp_weather_report['Point Type'] == "Original"

            temp_weather_report_masked = temp_weather_report[mask].reset_index()

            # print(temp_weather_report_masked)

            temp_weather_report_before_print = pd.concat([temp_green_processed_file, temp_weather_report_masked], axis=1)

            # select required columsn to print
            filter_columns = ["Date", "Ship's Time (UTC)", "Latitude", "Longitude", "Steaming Time (Hrs)",
                              "Max TCP Allowed Time",
                              "Observed distance (NM)", "Average Engine RPM", "Slip %", "Course",
                              "Wind Force(BF Scale)", "waveHeight", "Current factor",
                              "True Wind Force (BF)", "Wind Sea Height (m)", "Sea Currents (kts)",
                              "HSFO ROB", "LSFO ROB", "MGO ROB", "LNG ROB",
                              "HSFO_Total", "LSFO_Total", "MGO_Total", "LNG_Total",
                              "Max TCP Allowed FO Consumption", "GWD"
                              ]

            temp_weather_report_for_print = temp_weather_report_before_print.loc[:, filter_columns]

            print("temp_weather_report_for_print")

            # print(temp_weather_report_for_print)

            return temp_weather_report_for_print

        def beautifying_weather_summary(self, weather_summary_table, from_port, to_port):

            # print("beautifying_weather_summary",weather_summary_table)

            weather_summary_table['Latitude'] = weather_summary_table['Latitude'].astype(float)
            weather_summary_table['Longitude'] = weather_summary_table['Longitude'].astype(float)
            weather_summary_table['GWD'] = weather_summary_table['GWD'].apply(lambda x: "NO" if x is np.nan else x)

            weather_summary_table.iloc[0, 2] = "COSP"
            weather_summary_table.iloc[-1, 2] = "EOSP"

            weather_summary_table.iloc[0, 3] = from_port
            weather_summary_table.iloc[-1, 3] = to_port

            for g in range(len(weather_summary_table)):
                weather_summary_table["Date"][g] = str(weather_summary_table["Date"][g]) + " " + str(
                    weather_summary_table["Ship's Time (UTC)"][g])

            weather_summary_table["Date"] = pd.to_datetime(weather_summary_table["Date"], format="%Y-%m-%d %H:%M:%S",
                                                           errors='coerce')

            def custom_date_format3(date):  # Define a custom function to format the date
                day = date.day
                suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
                return date.strftime(f"%d{suffix} %b %Y %H:%M")

            weather_summary_table["Date"] = weather_summary_table["Date"].apply(custom_date_format3)

            weather_summary_table = weather_summary_table.drop(
                ["Ship's Time (UTC)", "Wind Force(BF Scale)", "waveHeight", "Current factor", "True Wind Force (BF)",
                 "Wind Sea Height (m)", "Sea Currents (kts)"], axis=1)

            # print("beautifying_weather_summary2",weather_summary_table)
            # weather_summary_table['Date']=weather_summary_table['Date'].apply(lambda x:x.strftime("%d/%m/%Y"))

            # print("beautifying_weather_summary2",weather_summary_table)

            tuple_list4 = [("Date", ""),
                           ("Lat", ""),
                           ("Log", ""),
                           ("Steaming Hours", ""),
                           ("Allowed Steaming Hours", ""),
                           ("Distance (NM)", ""),
                           ("Avg - RPM", ""),
                           ("Slip (%)", ""),
                           ("Course", ""),
                           ("Bunker ROB (MT)", "HSFO"),
                           ("Bunker ROB (MT)", "VULSFO"),
                           ("Bunker ROB (MT)", "MGO"),
                           ("Bunker ROB (MT)", "MDO"),
                           ("Bunker Cons. (MT)", "HSFO"),
                           ("Bunker Cons. (MT)", "VULSFO"),
                           ("Bunker Cons. (MT)", "MGO"),
                           ("Bunker Cons. (MT)", "MDO"),
                           ("Allowed Cons. MT", ""),
                           ("Good Weather", "")]

            style1 = [{'selector': 'th',
                       'props': [('text-align', 'center'),
                                 ('background-color', '#aaffdc'),
                                 ('color', 'darkblue'),
                                 ('font-size', '25px'),
                                 ('border-style', 'solid'),
                                 ('border-width', '1px'),
                                 ('font-weight', 'normal'),
                                 ('font-family', 'play')]
                       },

                      {'selector': 'td',
                       'props': [('text-align', 'center'),
                                 ('border-style', 'solid'),
                                 ('border-width', '1px'),
                                 ('font-size', '24px'),
                                 ('border-color', 'black'),
                                 ('font-family', 'play'),
                                 ('font-weight', 'normal')]
                       },


                      {"selector": "tr", "props": "line-height: 30px;"},
                      {"selector": "td,th", "props": "line-height: inherit; padding: 6;"}

                      ]

            index4 = pd.MultiIndex.from_tuples(tuple_list4)

            #         df=pd.DataFrame(weather_summary_table.values,columns=index4).style.\
            #                                                                             format(precision=2,na_rep='-').\
            #                                                                             set_table_styles(style1).hide_index()

            def add_row_color(x):
                print("printing rows")
                # print(x)
                if (x['Good Weather'] == 'GWD').bool():
                    return ['background-color: #aaffdc'] * len(x)
                elif (x['Good Weather'] == 'YES').bool():
                    return ['background-color: #aaffdc'] * len(x)
                else:
                    return ['background-color: white'] * len(x)

            df10 = pd.DataFrame(weather_summary_table.values, columns=index4).style. \
                format(precision=2, na_rep='-'). \
                apply(add_row_color, axis=1). \
                set_table_styles(style1).hide()

            print("--" * 20)
            # print('page_bc',df10)
            print("--" * 20)

            dfi.export(df10, "./png/weather_detail_summary.png")

        def message_traffic(self, processed_file_add_missing_columns, raw_df, from_port, to_port):

            dep_port = "Departure" + "-" + from_port
            arrival_port = "Arrival" + "-" + to_port

            # print(processed_file_add_missing_columns) # This is the dataframe you are going to conver to required format for prinint

            raw_df.dropna(axis=1, how='all', inplace=True)
            filter_from_row = raw_df[raw_df['Unnamed: 0'] == 'VESSEL'].index[0]
            raw_df = raw_df.loc[filter_from_row:, :].drop(['Unnamed: 0'], axis=1).T
            raw_df = raw_df.rename(columns=raw_df.iloc[0]).drop(raw_df.index[0])
            raw_df.reset_index(drop=True, inplace=True)
            raw_df.fillna(0, inplace=True)
            # print("raw_df",raw_df)
            # print("this is raw df for message traffic",raw_df)
            # print("raw_df_columns",raw_df.columns[-1])

            #         required_columns_list=["Report type","Latitude","Longitude","Date","True Wind Force Direction",
            #                                "Wind Sea Height Direction",
            #                                 "Ordered Speed (Kts)","Average speed (Kts)","Course","Average Engine RPM","Slip %",
            #                                "Observed distance (NM)","Distance to go (NM)","Next Port ETA","HSFO ROB","MGO ROB","LNG ROB",
            #                                "LSFO ROB",raw_df.columns[-1]]

            required_columns_list = ["Report type", "Latitude", "Longitude", "Date", "True Wind Force Direction",
                                     "Wind Sea Height Direction",
                                     "Ordered Speed (Kts)", "Average speed (Kts)", "Course", "Average Engine RPM", "Slip %",
                                     "Observed distance (NM)", "Distance to go (NM)", "Next Port ETA", "HSFO ROB",
                                     "LSFO ROB", "MGO ROB",
                                     "LNG ROB", raw_df.columns[-1]]

            # "Ship's Time (UTC)" this was used after "date"

            # required columnd filtering
            processed_file_add_missing_columns = raw_df.loc[:, required_columns_list]

            for k in range(len(processed_file_add_missing_columns)):
                processed_file_add_missing_columns['True Wind Force Direction'][k] = str(
                    raw_df['True Wind Force Direction'][k]) + ' x ' + str(raw_df['True Wind Force (BF)'][k])
                processed_file_add_missing_columns['Wind Sea Height Direction'][k] = str(
                    raw_df['Wind Sea Height Direction'][k]) + ' x ' + str(raw_df['Wind Sea Height (m)'][k])
            # processed_file_add_missing_columns['True Wind Force Direction']=processed_file_add_missing_columns['True Wind Force Direction']+'x'+processed_file_add_missing_columns['True Wind Force (BF)']

            processed_file_add_missing_columns.iloc[0, 0] = dep_port
            processed_file_add_missing_columns.iloc[-1, 0] = arrival_port

            # print("after required columns selected",processed_file_add_missing_columns)
            processed_file_add_missing_columns.iloc[:, -1] = processed_file_add_missing_columns.iloc[:, -1].apply(
                lambda x: str(x).replace('0', ''))

            #         print("after required columns selected1",processed_file_add_missing_columns)
            #         processed_file_add_missing_columns['Date']=processed_file_add_missing_columns['Date'].apply(lambda x:x.strftime("%d/%m/%Y %H:%M:%S"))
            processed_file_add_missing_columns['Date'] = pd.to_datetime(processed_file_add_missing_columns['Date'],
                                                                        format="%Y-%m-%d %H:%M:%S", errors='coerce')

            def custom_date_format4(date):  # Define a custom function to format the date
                day = date.day
                suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
                return date.strftime(f"%d{suffix} %b %Y %H:%M")

            processed_file_add_missing_columns['Date'] = processed_file_add_missing_columns['Date'].apply(
                custom_date_format4)

            # print("after required columns selected1",processed_file_add_missing_columns)

            tuple_list5 = [("Report Type", ""),
                           ("Position", "Lat"),
                           ("Position", "Log"),
                           ("Date/ Time (GMT)", ""),
                           ("Since last report", "Avg Wind (Dir. x Bft)"),
                           ("Since last report", "Avg Sea (Dir. x Height)"),
                           ("Since last report", "Ordered Speed (Kts)"),
                           ("Since last report", "Avg. Speed (Kts)"),
                           ("Since last report", "Course"),
                           ("Since last report", "RPM"),
                           ("Since last report", "Slip (%)"),
                           ("Since last report", "Distance Sailed (NM)"),
                           ("DTG (NM)", ""), ("ETA (LT)", ""),
                           ("BROB(MT)", "HSFO"),
                           ("BROB(MT)", "V/ULSFO"),
                           ("BROB(MT)", "MGO"),
                           ("BROB(MT)", "MDO"),
                           ("Remarks", "")]

            style1 = [{'selector': 'th',
                       'props': [('text-align', 'center'),
                                 ('background-color', 'lightgrey'),
                                 ('color', 'darkblue'),
                                 ('font-size', '25px'),
                                 ('border-style', 'solid'),
                                 ('border-width', '1px'),
                                 ('font-weight', 'normal'),
                                 ('font-family', 'play')]
                       },

                      {'selector': 'td',
                       'props': [('text-align', 'center'),
                                 ('border-style', 'solid'),
                                 ('border-width', '1px'),
                                 ('font-size', '24px'),
                                 ('border-color', 'black'),
                                 ('font-family','play'),
                                 ('font-weight','normal')]
                       },

                      {"selector": "tr", "props": "line-height: 30px;"},
                      {"selector": "td,th", "props": "line-height: inherit; padding: 6;"}

                      ]  # ('font-family','play'),('font-weight','normal')

            index5 = pd.MultiIndex.from_tuples(tuple_list5)

            df = pd.DataFrame(processed_file_add_missing_columns.values, columns=index5).style. \
                format(precision=2, na_rep='-'). \
                set_table_styles(style1).hide()

            dfi.export(df, "./png/message_traffic_summary.png")

        # ----test message traffic end

        def generate_bar_chart1(self, processed_file_add_missing_columns):
            # GRAPH

            # print("fuel con graph1",processed_file_add_missing_columns)

            # generate required dataframe

            temp_required_file = processed_file_add_missing_columns.loc[:,
                                 ["Date", "Ship's Time (UTC)", "Actual Total Consumption FO",
                                  "Max TCP Allowed FO Consumption"]]
            # print("temp_required_file",temp_required_file)

            for k in range(len(temp_required_file)):
                temp_required_file['Date'][k] = str(temp_required_file['Date'][k]) + " " + str(
                    temp_required_file["Ship's Time (UTC)"][k])
                # temp_required_file['Date'][k]=str(temp_required_file['Date'][k])#+str("temp_required_file["Ship's Time (UTC)"][k]")

            # df = pd.DataFrame({'Date': processed_file_add_missing_columns["Date"], 'Actual IFO Cons.': processed_file_add_missing_columns["Actual Total Consumption FO"],'Allowed IFO Cons.':processed_file_add_missing_columns["Max TCP Allowed FO Consumption"]})
            df = pd.DataFrame(
                {'Date': temp_required_file["Date"], 'Actual IFO Cons.': temp_required_file["Actual Total Consumption FO"],
                 'Allowed IFO Cons.': temp_required_file["Max TCP Allowed FO Consumption"]})
            # print("This_is_it")
            # df["Date"]=pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M')
            # print(df)

            # print("dataframe for bar_chart1",df)
            # print("df_index",df.index[0])
            df.drop(index=df.index[0], axis=0, inplace=True)
            # print("after droping df",df)

            df['Date'] = df['Date'].apply(lambda x: x.split(":")[0] + ":" + x.split(":")[1])
            #         df = df[df['Actual IFO Cons.']>0.0]

            #         print("last row check")
            #         print(df.iloc[-1,0],df.iloc[-2,0])

            # below block of code was used to delete last before row if both are of same date
            #         if df.iloc[-1,0]==df.iloc[-2,0]:
            #             print("Yes they are same")
            #             df.drop(df.index[-2],inplace = True)
            #             print("dropped")

            df.sort_values(by='Date', inplace=True)

            # df.drop_duplicates(subset ="Date", keep = 'first', inplace = True)

            # print("bar_1",df)

            df = df[df['Date'].notna()]
            # print("print date",df)
            fig, ax = plt.subplots(figsize=(12, 7))
            tick_spacing = 1
            # plt.figure(figsize=(12, 7))
            bar = plt.bar(df['Date'], df['Actual IFO Cons.'], color='skyblue', width=0.5)
            # plt.bar_label(ax.containers[0],fontname='Arial Black',size=10,label_type ='center')
            # print("mdates",mdates)
            # print("generate_bar_chart1")
            # myFmt = DateFormatter("%Y-%m-%d")#
            # ax.xaxis.set_major_formatter(myFmt)#
            # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=0))
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing)) #
            # formatter = mdates.DateFormatter("%Y-%m-%d")
            # ax.xaxis.set_major_formatter(formatter)
            # locator = mdates.DayLocator()
            # ax.xaxis.set_major_locator(locator)

            plt.xticks(fontsize=10, color='grey')
            plt.yticks(np.arange(0, max(df['Actual IFO Cons.']), 10), fontsize=10, color='grey')
            # ax.xaxis.grid(False)
            # plt.line(df['Allowed IFO Cons.'],marker='*', color='black', ms=10)
            # line = df['Allowed IFO Cons.'].plot(kind='line', marker='.', color='black')
            line = plt.plot(df['Date'], df['Allowed IFO Cons.'], color='black', linewidth='1.9', marker='.', markersize=12)
            for x, y in zip(df['Date'], df['Allowed IFO Cons.']):
                label = "{:.2f}".format(y)
                plt.annotate(label,  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')
            plt.xlabel('Report Date(UTC)', fontname='Arial Black', fontsize=12, color='grey', labelpad=3)
            plt.ylabel('Cons.(mts)', fontname='Arial Black', fontsize=12, color='grey')
            lns = [bar] + line
            labels = ['Actual IFO Cons.', 'Allowed IFO Cons.']
            legend1 = plt.legend(lns, labels, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False,
                                 fontsize=12)
            # fig, ax = plt.subplots()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('red')
            # lin= line
            # label =['Allowed IFO Cons.']
            # legend2 = plt.legend(lin, label, bbox_to_anchor=(0.5, -0.2),ncol=4,frameon=False,fontsize = 10)
            # plt.grid()
            # ax.xaxis.grid(False)
            ## Rotate date labels automatically
            fig.autofmt_xdate()
            try:
                os.makedirs(folder_name + "/bar_chart/")
            except Exception as e:
                print("File Exists. Hence proceeding with execution")
            plt.savefig("./png/bar_chart1.png", bbox_inches='tight', dpi=100)
            # plt.show()
            plt.close(fig)

        def generate_bar_chart2(self, processed_file_add_missing_columns):

            # print("fuel con graph2",processed_file_add_missing_columns)

            temp_required_file = processed_file_add_missing_columns.loc[:,
                                 ["Date", "Ship's Time (UTC)", "Steaming Time (Hrs)", "Max TCP Allowed Time"]]
            # print("temp_required_file1",temp_required_file)

            for k in range(len(temp_required_file)):
                temp_required_file['Date'][k] = str(temp_required_file['Date'][k]) + " " + str(
                    temp_required_file["Ship's Time (UTC)"][k])

            # print("temp_required_file3",temp_required_file)

            df = pd.DataFrame(
                {'Date': temp_required_file["Date"], 'Actual steaming': temp_required_file["Steaming Time (Hrs)"],
                 'Allowed steaming': temp_required_file["Max TCP Allowed Time"]})
            # print("dataframe for bar_chart2",df)
            df.drop(index=df.index[0], axis=0, inplace=True)
            df['Date'] = df['Date'].apply(lambda x: x.split(":")[0] + ":" + x.split(":")[1])
            df.sort_values(by='Date', inplace=True)
            # print("bar_2",df)
            df = df[df['Date'].notna()]
            fig, ax = plt.subplots(figsize=(12, 7))
            tick_spacing = 1
            # plt.figure(figsize=(12, 7))
            bar = plt.bar(df['Date'], df['Actual steaming'], color='skyblue', width=0.5)
            plt.xticks(fontsize=10, color='grey')
            plt.yticks(np.arange(0, max(df['Actual steaming']), 10), fontsize=10, color='grey')
            # ax.xaxis.grid(False)
            # plt.line(df['Allowed IFO Cons.'],marker='*', color='black', ms=10)
            # line = df['Allowed IFO Cons.'].plot(kind='line', marker='.', color='black')
            line = plt.plot(df['Date'], df['Allowed steaming'], color='black', linewidth='1.9', marker='.', markersize=12)
            for x, y in zip(df['Date'], df['Allowed steaming']):
                label = "{:.2f}".format(y)
                plt.annotate(label,  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')
            plt.xlabel('Report Date(UTC)', fontname='Arial Black', fontsize=12, color='grey')
            plt.ylabel('Hours', fontname='Arial Black', fontsize=12, color='grey')
            lns = [bar] + line
            labels = ['Actual steaming', 'Allowed steaming']
            legend1 = plt.legend(lns, labels, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2, frameon=False,
                                 fontsize=12)
            # fig, ax = plt.subplots()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('red')

            fig.autofmt_xdate()
            try:
                os.makedirs(folder_name + "/bar_chart/")
            except Exception as e:
                print("File Exists. Hence proceeding with execution")
            plt.savefig("./png/bar_chart2.png", bbox_inches='tight', dpi=100)
            # plt.show()
            plt.close(fig)


    # ---selection criteria


    weather = gwx(wx)

    if report_type == "SWH_DSS":
        swh_dss = weather.SWH_DSS(gwx_hours, bf_limit, swh_limit, windwave_limit)
        print('swh_dss selected')
    elif report_type == "DSS":
        dss = weather.DSS(gwx_hours, bf_limit, swh_limit, windwave_limit)
        print('dss selected')
    elif report_type == "SWH":
        swh = weather.SWH(gwx_hours, bf_limit, swh_limit)
        print('swh selected')
    else:
        bf = weather.BF(gwx_hours, bf_limit)
        print('bf selected')

    # bf=weather.BF(gwx_hours,bf_limit)
    # swh=weather.SWH(gwx_hours,bf_limit,swh_limit)
    # dss=weather.DSS(gwx_hours,bf_limit,swh_limit,windwave_limit)
    # swh_dss=weather.SWH_DSS(gwx_hours,bf_limit,swh_limit,windwave_limit)

    # ----selection criteria


    weather_table = weather.splitting_df()  # calculating individual report types.. weather it is bf or swh or dss or dss_swh
    weather.Gwx_pairing(weather_table, gwx_hours, report_type, gwx_type, gwx_method)  # calculating good weather day
    # report_type=swh/BF/DSS/DSS+SWH
    df_report_type = weather.report_type_df(
        weather_table)  # appending the list of individual dataframe units into a single dataframe
    final_df = weather.filter_report_type(df_report_type,
                                          report_type)  # filtering the required report type for final print
    weather_report = weather.weather_report_formating(final_df)
    # print(weather_report.T) # final weather output file as processed in python notebook

    # phase 2 processing

    sla = sealog_abstract(wx, processed_file, cp_ordered_cons_go, bf_limit)
    gwx_day_summary, gw_period_column = sla.gwx_daily_find(weather_report)  # daily good weather day summary to be used in sealog abstract input
    print("column of good weather period", gw_period_column)
    # print("1111",gwx_day_summary)
    # gwx_day_summary=sla.blue_table(gwx_day_summary)
    green_processed_file = sla.green_table(waranted_weather_yes_no,
                                           current_tolerance,
                                           tolerance,
                                           mistolerance,
                                           gwx_day_summary,  # passing the good weather summart agg list
                                           About_Cons_MaxTolerence,  # M79
                                           not_sure_L78  # L78
                                           )
    # report analysis summary voyage details

    report_analysis_summary_voyage_details = sla.report_analysis_summary_voyage_details(green_processed_file, from_port,to_port, adverse_current)
    report_analysis_summary_table = sla.report_analysis_summary_table(green_processed_file)
    # warranted_consumption=sla.warranted_consumption()


    # print("222222",green_processed_file)

    print('This is the green processed file from where we build voyage summary' * 5)
    # print(green_processed_file)

    voyage_detail_table = sla.voyage_detail(green_processed_file)
    voyage_summary = sla.voyage_summary(voyage_detail_table)
    voyage_detail_beautification = sla.beautifying_voyage_detail_table(voyage_detail_table)

    weather_detail_table = sla.weather_detail(weather_report.T, green_processed_file)
    # blank all old values
    # apply new conditions
    weather_table_beautification = sla.beautifying_weather_table(weather_detail_table, gw_period_column)

    weather_summary_table = sla.weather_summary(weather_report.T, green_processed_file)
    weather_table_beautification_summary = sla.beautifying_weather_summary(weather_summary_table, from_port, to_port)

    message_traffic = sla.message_traffic(processed_file_add_missing_columns, raw_df, from_port, to_port)

    barchart1 = sla.generate_bar_chart1(processed_file_add_missing_columns)
    barchart2 = sla.generate_bar_chart2(processed_file_add_missing_columns)


    # all removed--
    # all lib removed--


    # from google.colab import drive
    # drive.mount('/content/drive')
    class Pdf_process():
        '''Generate PDF'''

        def __init__(self, filename, green_processed_file, prepared_basis, constant_speed, weather_detail_table,
                     not_sure_L78, performance_calculation, current_tolerance, tolerance, mistolerance,
                     About_Cons_MaxTolerence, extrapolation_Allowed, fuel_type_used,
                     co2_factor_hsfo, co2_factor_mdo, co2_factor_lng, co2_factor, gwx_type, adverse_current,
                     waranted_weather_yes_no, GW_consumption_GO_added, voyage_phase):
            self.green_processed_file = green_processed_file
            # print("self.green_processed_file",self.green_processed_file.info(verbose=True))
            self.width, self.height = A4
            self.filename = filename
            self.margin = 0.4 * inch
            self.ratio = [1, 1, 4, 1, 1]
            self.ratio1 = [1, 6, 1]
            self.prepared_basis = prepared_basis
            self.constant_speed = constant_speed
            self.cp_ae_cons = green_processed_file['C/P Ordered Consumption GO(AE)'][0]
            self.current_tolerance = current_tolerance
            self.mistolerance = mistolerance
            self.About_Cons_MaxTolerence = About_Cons_MaxTolerence
            self.extrapolation_Allowed = extrapolation_Allowed
            self.fuel_type_used = fuel_type_used
            self.gwx_type = gwx_type
            self.adverse_current = adverse_current
            self.waranted_weather_yes_no = waranted_weather_yes_no
            self.GW_consumption_GO_added = GW_consumption_GO_added
            self.voyage_phase = voyage_phase

            self.cp_speed = self.green_processed_file['Ordered Speed (Kts)'][1].round(2)
            self.cp_cons = self.green_processed_file['CP Consumptions'][1].round(2)

            # print('pdf_process',self.green_processed_file)

            self.ship_name = self.green_processed_file["Vessel Name"][0]
            self.condition = self.green_processed_file['Voyage Type'][0]

            # print("green_process_arrival",self.green_processed_file)
            self.Dep_Date = self.green_processed_file['Date'][0].strftime("%d-%m-%Y")
            print("self.Dep_Date", self.Dep_Date)
            self.Arrival_Date = self.green_processed_file['Date'][len(self.green_processed_file) - 1].strftime("%d-%m-%Y")

            self.departure_time_str = self.green_processed_file["Ship's Time (UTC)"][0].strftime("%H:%M")
            # print("depart time",type(self.departure_time_str),self.departure_time_str)

            # self.arrival_time_str=self.green_processed_file["Ship's Time (UTC)"][7].strftime("%H:%M")
            self.arrival_time_str = self.green_processed_file.loc[:, "Ship's Time (UTC)"][
                len(green_processed_file["Ship's Time (UTC)"]) - 1].strftime("%H:%M")
            # print("self.arrival_time_str",self.arrival_time_str)

            # print("GREEN FILE",self.green_processed_file)

            self.total_distance = self.green_processed_file['Observed distance (NM)'].sum()
            B123 = self.total_distance
            self.total_time = self.green_processed_file['Steaming Time (Hrs)'].sum()
            print("total_time", self.total_time)
            # self.average_speed=self.green_processed_file['Average speed (Kts)'].mean()
            self.average_speed = self.total_distance / self.total_time
            print("self.average_speed", self.average_speed, self.total_distance, self.total_time)

            fuel_cols = ["HSFO ME Cons.",
                         "HSFO AE Cons.",
                         "HSFO Boiler",
                         "HSFO Others",
                         "LSFO ME Cons.",
                         "LSFO AE Cons.",
                         "LSFO Boiler",
                         "LSFO Others"]

            self.Total_FO_Bunker_Consumption = green_processed_file[fuel_cols].sum().sum()
            print("self.Total_FO_Bunker_Consumption", self.Total_FO_Bunker_Consumption)

            fuel_cols_mgo = ["MGO ME Cons.",
                             "MGO AE Cons.",
                             "MGO Boiler",
                             "MGO Others"]

            self.Total_Bunker_GO_Consumption = green_processed_file[fuel_cols_mgo].sum().sum()
            print("self.Total_Bunker_GO_Consumption", self.Total_Bunker_GO_Consumption)

            self.Total_bunkers_Consumed_at_sea = self.Total_FO_Bunker_Consumption + self.Total_Bunker_GO_Consumption
            print("self.Total_bunkers_Consumed_at_sea", self.Total_bunkers_Consumed_at_sea)

            self.total_time = self.green_processed_file["Steaming Time (Hrs)"].sum()

            self.Total_Average_Daily_Consumption = (self.Total_bunkers_Consumed_at_sea / self.total_time) * 24

            fuel_cols_hsfo = ["HSFO ME Cons.",
                              "HSFO AE Cons.",
                              "HSFO Boiler",
                              "HSFO Others", ]

            self.hdfo_total_con = self.green_processed_file[fuel_cols_hsfo].sum().sum()

            fuel_cols_ifo = ["LSFO ME Cons.",
                             "LSFO AE Cons.",
                             "LSFO Boiler",
                             "LSFO Others"]

            self.ifo_total_con = self.green_processed_file[fuel_cols_ifo].sum().sum()

            Total_FO_Bunker_Consumption = self.green_processed_file["Actual Total Consumption FO"].sum()
            B139 = Total_FO_Bunker_Consumption

            print("avg_fo", self.Total_FO_Bunker_Consumption, self.total_time)

            Average_FO_Daily_Consumption = self.Total_FO_Bunker_Consumption / self.total_time * 24
            print("Average_FO_Daily_consumption", Average_FO_Daily_Consumption)

            Total_Bunker_GO_Consumption_AE = self.green_processed_file["Actual Total Consumption  GO"].sum()
            print("Total_Bunker_GO_Consumption_AE", Total_Bunker_GO_Consumption_AE)

            # ----------------------calcualtions above---

            # ----filtering only good weather---

            conds = green_processed_file["GWD_agg"].isin(['YES','GWD'])
            green_processed_file_good_weather_only = green_processed_file.loc[conds]

            # print("green_processed_file_good_weather_only",green_processed_file_good_weather_only)

            self.Good_Weather_Distance = green_processed_file_good_weather_only['Observed distance (NM)'].sum()
            print("self.Good_Weather_Distance", self.Good_Weather_Distance)

            self.Good_Weather_Time = green_processed_file_good_weather_only['Steaming Time (Hrs)'].sum()
            print("self.Good_Weather_Time", self.Good_Weather_Time)
            self.Average_Speed_In_Good_Weather = self.Good_Weather_Distance / self.Good_Weather_Time
            print("self.Average_Speed_In_Good_Weather_type", self.Average_Speed_In_Good_Weather)

            if math.isnan(self.Average_Speed_In_Good_Weather):
                # self.Average_Speed_In_Good_Weather=self.Average_Speed_In_Good_Weather.astype(float)
                self.Average_Speed_In_Good_Weather = 0.0
                print("self.Average_Speed_In_Good_Weather_type_2", self.Average_Speed_In_Good_Weather)
            else:
                self.Average_Speed_In_Good_Weather = self.Average_Speed_In_Good_Weather
                print("self.Average_Speed_In_Good_Weather_type_3", self.Average_Speed_In_Good_Weather)

            print("self.Average_Speed_In_Good_Weather", self.Average_Speed_In_Good_Weather)

            self.Current_Distance = green_processed_file_good_weather_only['Current Distance'].astype(float).sum()
            print("Current_Distance", self.Current_Distance)

            self.current_factor = self.Current_Distance / self.Good_Weather_Time
            # print("self.current_factor",self.current_factor,self.Current_Distance,self.Good_Weather_Time)

            if math.isnan(self.current_factor):
                # self.Average_Speed_In_Good_Weather=self.Average_Speed_In_Good_Weather.astype(float)
                self.current_factor = 0.0

            print("self.current_factor", self.current_factor, self.Good_Weather_Time, self.Current_Distance)

            B126 = self.Good_Weather_Distance
            B127 = self.Good_Weather_Time
            B130 = self.current_factor
            L78 = not_sure_L78
            p78 = performance_calculation
            G79 = self.current_tolerance
            G78 = self.adverse_current
            I79 = tolerance
            B9 = self.green_processed_file["Ordered Speed (Kts)"][0]
            B10 = self.green_processed_file["CP Consumptions"][0]
            K79 = self.mistolerance / 100
            B11 = self.green_processed_file['C/P Ordered Consumption GO(AE)'][0]
            M79 = About_Cons_MaxTolerence
            O79 = self.extrapolation_Allowed

            print("-*-" * 20)
            print(L78, p78, green_processed_file_good_weather_only['Max TCP Allowed Time'].sum(), B126, B9, G79,
                    green_processed_file_good_weather_only['Max TCP Allowed Time with  Current Factor'].sum(), B130, B127)
            print("-*-" * 20)

            self.Max_Total_Allowed_Time_in_GW = np.where(L78 == 'NO',
                                                         np.where(p78 == 'Daily', green_processed_file_good_weather_only[
                                                             'Max TCP Allowed Time'].sum(), (B126 / (B9 - G79))),
                                                         np.where(p78 == 'Daily', green_processed_file_good_weather_only[
                                                             'Max TCP Allowed Time with  Current Factor'].sum(),
                                                                  ((B126 - B130 * B127) / (B9 - G79))))

            if math.isnan(self.Max_Total_Allowed_Time_in_GW):
                self.self.Max_Total_Allowed_Time_in_GW = np.array(0)

            print("self.Max_Total_Allowed_Time_in_GW", type(self.Max_Total_Allowed_Time_in_GW))

            self.Min_Total_Allowed_Time_in_GW = np.where(L78 == "NO",
                                                         np.where(p78 == "Daily", green_processed_file_good_weather_only[
                                                             'Min TCP Allowed Time'].sum(), (B126 / (B9 + I79))),
                                                         np.where(p78 == "Daily", green_processed_file_good_weather_only[
                                                             'Min TCP Allowed Time with  Current Factor'].sum(),
                                                                  ((B126 - B127 * B130) / (B9 + I79))))

            #         if self.Min_Total_Allowed_Time_in_GW):
            #             self.Min_Total_Allowed_Time_in_GW=0.0

            #         print("Max_Total_Allowed_Time_in_GW",Max_Total_Allowed_Time_in_GW)
            #         print("Min_Total_Allowed_Time_in_GW",Min_Total_Allowed_Time_in_GW)
            print("self.Min_Total_Allowed_Time_in_GW", self.Min_Total_Allowed_Time_in_GW)
            #                 green_processed_file_good_weather_only['Min TCP Allowed Time with  Current Factor'].sum(),
            #                 B126,B127,B130,B9,I79,self.Min_Total_Allowed_Time_in_GW)

            B131 = green_processed_file_good_weather_only['Performance Distance'].sum()
            print("performance_distance", B131)
            B133 = self.Max_Total_Allowed_Time_in_GW
            B134 = self.Min_Total_Allowed_Time_in_GW
            Performance_Speed = B131 / B127

            if math.isnan(Performance_Speed):
                Performance_Speed = 0.0

            self.B132 = Performance_Speed

            print("B131", B131)
            print("B133", B133)
            print("B134", B134)
            print("self.B132", self.B132)

            # temp_GW_Time_Gain=np.where(p78=='Daily',green_processed_file_good_weather_only['Time Gain'].sum()-green_processed_file_good_weather_only['Time Loss'].sum(),B134-B127)
            temp_GW_Time_Gain = np.where(G78 == "not_excluded", B134 - (B126 / self.B132), np.where(p78 == 'Daily',
                                                                                                    green_processed_file_good_weather_only[
                                                                                                        'Time Gain'].sum() -
                                                                                                    green_processed_file_good_weather_only[
                                                                                                        'Time Loss'].sum(),
                                                                                                    B134 - B127))
            print("temp_GW_Time_Gain", temp_GW_Time_Gain)

            GW_Time_Gain = np.where(temp_GW_Time_Gain <= 0, 0, temp_GW_Time_Gain)
            B136 = GW_Time_Gain
            print("B136", B136)

            # temp_GW_Time_Loss=np.where(p78=="Daily",green_processed_file_good_weather_only['Time Loss'].sum()-green_processed_file_good_weather_only['Time Gain'].sum(),B127-B133)
            temp_GW_Time_Loss = np.where(G78 == "not_excluded", (B126 / self.B132) - B133, np.where(p78 == "Daily",
                                                                                                    green_processed_file_good_weather_only[
                                                                                                        'Time Loss'].sum() -
                                                                                                    green_processed_file_good_weather_only[
                                                                                                        'Time Gain'].sum(),
                                                                                                    B127 - B133))
            print("temp_GW_Time_Loss", temp_GW_Time_Loss)

            GW_Time_Loss = np.where(temp_GW_Time_Loss <= 0, 0, temp_GW_Time_Loss)
            B135 = GW_Time_Loss
            print("B135", B135)

            GW_FO_Consumption = green_processed_file_good_weather_only['Actual Total Consumption FO'].sum()
            print("GW_FO_Consumption", GW_FO_Consumption)

            self.Actual_Usage_in_Good_Weather = GW_FO_Consumption
            print("self.Actual_Usage_in_Good_Weather", self.Actual_Usage_in_Good_Weather)

            GW_Consumption_GO = green_processed_file_good_weather_only['Actual Total Consumption  GO'].sum()
            B153 = GW_Consumption_GO
            print("B153", B153)
            print("B127",B127)

            if self.GW_consumption_GO_added == "included":
                self.B167 = self.Actual_Usage_in_Good_Weather + B153
                print("B167", self.B167)
            else:
                self.B167 = self.Actual_Usage_in_Good_Weather
                print("B167", self.B167)

            #Average_Daily_Consumption = self.B167 / B127 * 24

            Average_Daily_Consumption = 0 if self.B167 ==0 else self.B167 / B127 * 24

            self.B168 = Average_Daily_Consumption
            print("self.B168", self.B168)

            Min_Allowed_FO_Cons_Time_in_GW = np.where(L78 == "NO", np.where(p78 == "Daily",
                                                                            green_processed_file_good_weather_only[
                                                                                'Min TCP Allowed FO Consumption'].sum(),
                                                                            (B126 / (B9 - G79) / 24 * B10 * (1 - K79))),
                                                      np.where(p78 == "Daily", green_processed_file_good_weather_only[
                                                          'Min TCP Allowed FO Consumption with  Current Factor'].sum(),
                                                               ((B126 - B127 * B130) / (B9 - G79) / 24 * B10 * (1 - K79))))
            print("Min_Allowed_FO_Cons_Time_in_GW", Min_Allowed_FO_Cons_Time_in_GW)

            Min_Allowed_GO_Cons_Time_in_GW = np.where(L78 == "NO", np.where(p78 == "Daily",
                                                                            green_processed_file_good_weather_only[
                                                                                'Min TCP Allowed GO Consumption'].sum(),
                                                                            (B126 / (B9 - G79) / 24 * B11 * (1 - K79))),
                                                      np.where(p78 == "Daily", green_processed_file_good_weather_only[
                                                          'Min TCP Allowed GO Consumption with  Current Factor'].sum(),
                                                               ((B126 - B127 * B130) / (B9 - G79) / 24 * B11 * (1 - K79))))
            print("Min_Allowed_GO_Cons_Time_in_GW", Min_Allowed_GO_Cons_Time_in_GW)
            B156 = Min_Allowed_GO_Cons_Time_in_GW

            print("B156", B156)

            # original-----
            #         Max_Allowed_Total_GO_Cons_in_GW=np.where(L78=="No",np.where(p78=="Daily",green_processed_file_good_weather_only['Max TCP Allowed GO Consumption'].sum(),(B126/(B9-G79)/24*B11*(1+(M79/100)))),np.where(p78=="Daily",green_processed_file_good_weather_only['Max TCP Allowed  GO Cons. with  Current Factor'].sum(),((B126-B127*B130)/(B9-G79)/24*B11*1+M79/100)))
            #         B155=Max_Allowed_Total_GO_Cons_in_GW

            #         print("Max_Allowed_Total_GO_Cons_in_GW",Max_Allowed_Total_GO_Cons_in_GW)

            # originalend------

            # Test----

            Max_Allowed_Total_GO_Cons_in_GW = np.where(L78 == "NO", np.where(p78 == "Daily",
                                                                             green_processed_file_good_weather_only[
                                                                                 'Max TCP Allowed GO Consumption'].sum(), (
                                                                                         B126 / (B9 - G79) / 24 * B11 * (
                                                                                             1 + (M79 / 100)))),
                                                       np.where(p78 == "Daily", green_processed_file_good_weather_only[
                                                           'Max TCP Allowed  GO Cons. with  Current Factor'].sum(), (
                                                                            (B126 - B127 * B130) / (
                                                                                B9 - G79) / 24 * B11 * 1 + (M79 / 100))))
            B155 = Max_Allowed_Total_GO_Cons_in_GW

            print("Max_Allowed_Total_GO_Cons_in_GW", Max_Allowed_Total_GO_Cons_in_GW, B126, B127, B130, B9, G79, B10, M79,
                    B11)

            # Test end----

            # print("values1",B126,B127,B130,B9,G79,B10,M79)

            # Original
            Max_Allowed_Total_FO_Cons_in_GW = np.where(L78 == "NO",
                                                       np.where(p78 == "Daily", green_processed_file_good_weather_only[
                                                           'Max TCP Allowed FO Consumption'].sum(),
                                                                (B126 / (B9 - G79) / 24 * B10 * (1 + (M79 / 100)))),
                                                       np.where(p78 == "Daily", green_processed_file_good_weather_only[
                                                           'Max TCP Allowed  FO Cons. with  Current Factor'].sum(),
                                                                ((((B126 - (B127 * B130)) / (B9 - G79))) / 24) * B10) * (
                                                                   1 + (M79 / 100)))  # /(B9-G79)/24*B10*(1+M79)))
            # original
            # TEST

            # print("values1",B126,B127,B130,B9,G79,B10,M79)
            # (B126-(B127*B130))/(B9-G79))/24)*B10*(1+M79)
            a = (B126 - (B127 * B130))
            b = (B9 - G79)
            part1 = (a / b) / 24
            c = B10 * (1 + M79 / 100)

            final = part1 * c

            print("result", a, b, part1, c, final)

            #         Max_Allowed_Total_FO_Cons_in_GW=np.where(L78=="No",np.where(p78=="Daily",green_processed_file_good_weather_only['Max TCP Allowed FO Consumption'].sum(),(B126/(B9-G79)/24*B10*(1+M79))),
            #                                                  np.where(p78=="Daily",green_processed_file_good_weather_only['Max TCP Allowed  FO Cons. with  Current Factor'].sum(),((B126-B127*B130)/(B9-G79)/24*B10*(1+M79))))
            #         #TEST

            print("Max_Allowed_Total_FO_Cons_in_GW", Max_Allowed_Total_FO_Cons_in_GW)

            GW_Avg_Daily_FO_Consumption = 0 if GW_FO_Consumption==0 else GW_FO_Consumption / self.Good_Weather_Time * 24
            print("GW_Avg_Daily_FO_Consumption", GW_Avg_Daily_FO_Consumption)

            #         GW_Consumption_GO=green_processed_file_good_weather_only['Actual Total Consumption  GO'].sum()
            #         B153=GW_Consumption_GO
            #         print("B153",B153)

            #         B167=self.Actual_Usage_in_Good_Weather+B153
            #         print("B167",B167)

            GW_Avg_Daily_Consumption_GO = GW_Consumption_GO / self.Good_Weather_Time * 24

            print("GW_Avg_Daily_Consumption_GO", GW_Avg_Daily_Consumption_GO)

            GW_Gasoil_Gain = np.where(p78 == "Daily", green_processed_file_good_weather_only['Gas oil Gain'].sum() -
                                      green_processed_file_good_weather_only['Gas oil Loss'].sum(), B156 - B153)
            GW_Gasoil_Loss = np.where(p78 == "Daily", green_processed_file_good_weather_only['Gas oil Loss'].sum() -
                                      green_processed_file_good_weather_only['Gas oil Gain'].sum(), B153 - B155)
            B157 = GW_Gasoil_Loss
            B158 = GW_Gasoil_Gain
            print("B158", B158)

            # ---filtering only good weather---

            # ---dont know what type to add--

            Average_Daily_GO_Consumption = Total_Bunker_GO_Consumption_AE / self.total_time * 24
            print("Average_Daily_GO_Consumption", Average_Daily_GO_Consumption)

            # self.Min_Allowable_Usage=Min_Allowed_GO_Cons_Time_in_GW+Min_Allowed_FO_Cons_Time_in_GW
            self.Min_Allowable_Usage = Min_Allowed_FO_Cons_Time_in_GW + 0.0

            if math.isnan(self.Min_Allowable_Usage):
                self.Min_Allowable_Usage = 0.0
                print("this is nan")

            print("self.Min_Allowable_Usage", self.Min_Allowable_Usage, Min_Allowed_GO_Cons_Time_in_GW,
                    Min_Allowed_FO_Cons_Time_in_GW)

            # print("self.Min_Allowable_Usage",Min_Allowed_GO_Cons_Time_in_GW,Min_Allowed_FO_Cons_Time_in_GW,self.Min_Allowable_Usage)

            # self.Max_Allowable_Usage=Max_Allowed_Total_GO_Cons_in_GW+Max_Allowed_Total_FO_Cons_in_GW
            self.Max_Allowable_Usage = Max_Allowed_Total_FO_Cons_in_GW + 0.0

            if math.isnan(self.Max_Allowable_Usage):
                self.Max_Allowable_Usage = 0.0
                print("this is nan")

            print("self.Max_Allowable_Usage", Max_Allowed_Total_GO_Cons_in_GW, Max_Allowed_Total_FO_Cons_in_GW,
                    self.Max_Allowable_Usage)

            self.GW_Fuel_Loss_temp = self.B167 - self.Max_Allowable_Usage
            self.GW_Fuel_Gain_temp = self.Min_Allowable_Usage - self.B167

            print("self.GW_Fuel_Gain_temp", self.Min_Allowable_Usage, self.B167)

            if self.GW_Fuel_Loss_temp < 0:
                self.GW_Fuel_Loss = 0
            else:
                self.GW_Fuel_Loss = self.GW_Fuel_Loss_temp

            if self.GW_Fuel_Gain_temp < 0:
                self.GW_Fuel_Gain = 0
            else:
                self.GW_Fuel_Gain = self.GW_Fuel_Gain_temp

            print("gw_fuel_loss_gain", self.GW_Fuel_Loss, self.GW_Fuel_Gain)
            # self.total_fuel_loss=np.where(O79=='YES',self.GW_Fuel_Loss*self.total_distance/self.Good_Weather_Distance,self.GW_Fuel_Loss)

            self.total_fuel_loss = np.where(O79 == 'YES',
                                            self.GW_Fuel_Loss * self.total_distance / self.Good_Weather_Distance,
                                            self.GW_Fuel_Loss)
            self.total_fuel_gain = np.where(O79 == 'YES',
                                            self.GW_Fuel_Gain * self.total_distance / self.Good_Weather_Distance,
                                            self.GW_Fuel_Gain)

            print("total_fuel_loss and total_fuel_gain")
            print(self.total_fuel_loss, self.total_fuel_gain)

            self.total_time_loss = np.where(O79 == "YES", B135 * B123 / B126, B135)
            self.total_time_gain = np.where(O79 == "YES", B136 * B123 / B126, B136)

            print("total_time_loss_gain")
            print(self.total_time_loss, self.total_time_gain)

            GW_fueloil_loss_temp = np.where(p78 == 'Daily', green_processed_file_good_weather_only['Fuel oil Loss'].sum() -
                                            green_processed_file_good_weather_only['Fuel  Oil Gain'].sum(),
                                            GW_FO_Consumption - Max_Allowed_Total_FO_Cons_in_GW)

            if GW_fueloil_loss_temp < 0:
                GW_fueloil_loss = ""
            else:
                GW_fueloil_loss = GW_fueloil_loss_temp

            GW_fueloil_Gain_temp = np.where(p78 == "Daily", green_processed_file_good_weather_only['Fuel  Oil Gain'].sum() -
                                            green_processed_file_good_weather_only['Fuel oil Loss'].sum(),
                                            Min_Allowed_FO_Cons_Time_in_GW - GW_FO_Consumption)

            if GW_fueloil_Gain_temp < 0:
                GW_fueloil_Gain = ""
            else:
                GW_fueloil_Gain = GW_fueloil_Gain_temp

            print("GW_fuel_oil_loss_gain")
            print(GW_fueloil_loss, GW_fueloil_Gain)

            self.track_time_loss = GW_Time_Loss
            self.track_time_gain = GW_Time_Gain
            self.total_time_loss = self.total_time_loss
            self.total_time_gain = self.total_time_gain

            print("total_time_loss and total_time_gain")
            print(self.total_time_loss, self.total_time_gain)

            Total_Fuel_Oil_Loss_temp = np.where(O79 == "YES", GW_fueloil_loss_temp * B123 / B126, GW_fueloil_loss_temp)
            Total_Fuel_Oil_Gain_temp = np.where(O79 == "YES", GW_fueloil_Gain_temp * B123 / B126, GW_fueloil_Gain_temp)

            print("Total_Fuel_Oil_Gain_temp", Total_Fuel_Oil_Gain_temp)
            print("Total_Fuel_Oil_Loss_temp", Total_Fuel_Oil_Loss_temp)

            if Total_Fuel_Oil_Loss_temp < 0:
                Total_Fuel_Oil_Loss = 0
            else:
                Total_Fuel_Oil_Loss = Total_Fuel_Oil_Loss_temp

            if Total_Fuel_Oil_Gain_temp < 0:
                Total_Fuel_Oil_Gain = 0
            else:
                Total_Fuel_Oil_Gain = Total_Fuel_Oil_Gain_temp

            print("Fuel_oil_loss_gain", Total_Fuel_Oil_Loss, Total_Fuel_Oil_Gain)

            Total_Gasoil_Loss = np.where(O79 == "YES", B157 * B123 / B126, B157)
            Total_Gasoil_Gain = np.where(O79 == "YES", B158 * B123 / B126, B158)

            print("total_gas_oil", Total_Gasoil_Loss, Total_Gasoil_Gain)

            # -------

            self.report_date = datetime.utcnow().strftime("%d-%b-%y")
            print("report_date", self.report_date)

            self.weather_detail_table = weather_detail_table  # weather table imported from weather detail processing

            print('value of cp_ae_cons', self.cp_ae_cons)

            # the below codes are to calculate the number of pages required for weather tables

            self.len_of_table = len(weather_detail_table)
            print(self.len_of_table)
            self.records_per_page = 50
            self.no_of_page_weather = self.len_of_table / self.records_per_page
            print("math.ceil weather_detail_table", math.ceil(self.no_of_page_weather))

            # the below codes are to calculate the number of pages required for weather tables
            len_of_table = len(green_processed_file)
            print(len_of_table)
            records_per_page = 45
            self.no_of_page_voyage = len_of_table / records_per_page
            print("math.ceil green_processed_file", math.ceil(self.no_of_page_voyage))

        def make_directories(self):

            self.folder_name = self.filename.rsplit('.', maxsplit=1)[0]
            print(self.folder_name)
            try:
                os.makedirs(self.folder_name + "/")
                # os.makedirs("./images/"+self.folder_name+"/")
            except Exception as e:
                print("Folder exists. Proceeding with execution")

        def good_weather_summary(self):
            gwx_summary_table = self.green_processed_file
            '''filter out only the required columns'''

            return gwx_summary_table

        def colWidth(self):

            allowableWidths = (self.width - (self.margin * 2))
            maxColCount = len(self.ratio)
            self.colwidths = [(i / sum(self.ratio)) * allowableWidths for i in self.ratio]
            # print(self.colwidths)

        def colWidth5(self):

            allowableWidths = (self.width - (self.margin * 2))
            maxColCount = len(self.ratio)
            self.colwidths5 = [(i / sum(self.ratio1)) * allowableWidths for i in self.ratio1]
            # print(self.colwidths)

        def colWidth2(self):

            allowableWidths = (self.width - (self.margin * 2))
            maxColCount = len(self.ratio)
            self.colwidths2 = [(i / sum(self.ratio)) * allowableWidths for i in self.ratio]
            print(self.colwidths2)
            self.colwidths2[0] = 30
            self.colwidths2[1] = 180
            self.colwidths2[2] = 117.675
            self.colwidths2[3] = 180
            self.colwidths2[4] = 30

            print(self.colwidths2)

        def colWidth3(self):

            allowableWidths1 = (self.width)
            self.colwidths3 = allowableWidths1 / 3.4

            print(self.colwidths3)

        def colWidth4(self):

            allowableWidths2 = (self.width - (self.margin * 2))
            self.colwidths4 = [165, 75, 75, 75, 70, 70]
            print("colWidth4", self.colwidths4)

        def generate_pdf(self, from_port, to_port, bad_weather_period_definition_as_per_CP_in_hrs,
                         report_type,
                         # current_excluded,
                         prepared_basis,
                         waranted_weather_yes_no,
                         extrapolation_Allowed,
                         current_tolerance,
                         tolerance,
                         mistolerance,
                         About_Cons_MaxTolerence,
                         bf_limit_dss
                         ):

            fbold = font_manager.FontProperties(fname="./font/Play-Bold.ttf")
            freg = font_manager.FontProperties(fname="./font/Play-Regular.ttf")
            styles = getSampleStyleSheet()
            play_font = r"./font/Play-Regular.ttf"
            playbold_font = r"./font/Play-Bold.ttf"
            pdfmetrics.registerFont(TTFont("Play-Regular", play_font))
            pdfmetrics.registerFont(TTFont("Play-Bold", playbold_font))

            # colour background
            colour = {'title': {'text': '#000000'},
                      'subtitle': {'text': '#000000'},
                      'sectionheader': {'text': '#000000', 'line': '#969696'},
                      'tablecontents': {'rowlabeltext': '#000000', 'datatext': '#000000', 'columntext': '#000000',
                                        'line': '#969696'},
                      'status': {'alarm': {'text': '#231f20', 'back': '#faccd1', 'symbol': '#FF0000'},
                                 'warn': {'text': '#231f20', 'back': '#fff2bd', 'symbol': '#eed202'},
                                 'normal': {'text': '#231f20', 'back': '#cbe2a7', 'symbol': '#00FF00'},
                                 'null': {'text': '#D3D3D3', 'symbol': '#D3D3D3'}}}
            size = {'title': {'text': 12},
                    'subtitle': {'text': 12},
                    'sectionheader': {'text': 14, 'line': 2 / 200 * inch},
                    'tablecontents': {'rowlabeltext': 10, 'datatext': 10, 'columntext': 10, 'line': 0.5 / 200 * inch},
                    'status': {'alarm': {'text': 10},
                               'warn': {'text': 10},
                               'normal': {'text': 10},
                               'null': {'text': 10}}}
            font = {'title': {'text': 'Bold'},
                    'subtitle': {'text': 'Regular'},
                    'sectionheader': {'text': 'Bold'},
                    'tablecontents': {'rowlabeltext': 'Bold', 'datatext': 'Regular', 'columntext': 'Bold'},
                    'status': {'alarm': {'text': 'Regular'},
                               'warn': {'text': 'Regular'},
                               'normal': {'text': 'Regular'},
                               'null': {'text': 'Regular'}}}
            margin = 0.4 * inch
            width, height = A4
            # ('SPAN',(0,0),(-1,0)),

            space = Spacer(width - margin * 2, 0.2 * inch)
            space2 = Spacer(width - margin * 2, 0.1 * inch)
            space3 = Spacer(width - margin * 2, 0.8 * inch)

            sectionHeaderStyle3 = [
                ('BOTTOMPADDING', (0, 8), (-1, -1), 40),
                ('TOPPADDING', (0, 0), (-1, -8), 20),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 0), (-1, -1),
                 '#a8d7c5')]  # colors.mediumturquoise)] ## middle alighment # First row combined # first row background colour light grey
            #     ('LINEBELOW', (0, 0), (-1, -1), size['sectionheader']['line'], colors.HexColor(colour['sectionheader']['line']))]
            sectionHeaderStyle = [
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
            ### for single columns
            sectionHeaderStyle2 = [
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),  ## middle alighment, # first row background colour light grey
                ('LINEBELOW', (0, 0), (0, 0), size['sectionheader']['line'],
                 colors.HexColor(colour['sectionheader']['line']))]

            sectionHeaderStyle4 = [
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), '#a8d7c5'),
                ('BACKGROUND', (2, 1), (-1, -1), '#4fb28b'),  ## middle alighment, # first row background colour light grey
                ('LINEBELOW', (0, 0), (0, 0), size['sectionheader']['line'],
                 colors.HexColor(colour['sectionheader']['line']))]

            sectionHeaderStyle5 = [
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), '#a8d7c5'),  ## middle alighment, # first row background colour light grey
                ('LINEBELOW', (0, 0), (0, 0), size['sectionheader']['line'],
                 colors.HexColor(colour['sectionheader']['line']))]

            doc = SimpleDocTemplate(f"./assets/" + self.folder_name + ".pdf", leftMargin=margin, rightMargin=margin,
                                    topMargin=margin, bottomMargin=margin)
            orden = ParagraphStyle('orden')
            orden.leading = 16
            story = []

            if self.voyage_phase == "MID":
                I_zn = rImage("./static/MID-ZN.jpg")
            elif self.voyage_phase == "END":
                I_zn = rImage("./static/END-ZN.jpg")
            else:
                I_zn = rImage("./static/ZN.jpg")
            I_zn.drawHeight = 8.1 * inch * I_zn.drawHeight / I_zn.drawWidth
            I_zn.drawWidth = 8.3 * inch * 0.9
            story.append(I_zn)

            rawdata = [['', '', self.ship_name, '', ''],
                       ['', '', '', '', ''],
                       ['', '', '<b>Prepared Basis : ' + self.prepared_basis + '</b>', '', ''],
                       ['', '', from_port + ' to ' + to_port, '', ''],
                       ['', '', 'Dep.Date:' + " " + str(self.Dep_Date) + " " + str(self.departure_time_str) + ' UTC', '',
                        ''],
                       ['', '', 'Arrival.Date:' + " " + str(self.Arrival_Date) + " " + str(self.arrival_time_str) + ' UTC',
                        '', ''],
                       ['', '', 'Condition : ' + str(self.condition), '', ''],
                       ['', '', '<b>Report Date</b> : ' + str(self.report_date), '', ''],
                       ['', '', '<b>Reference No.</b> : ' + '', '', '']]

            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-Bold" size="16" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx == 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-Bold" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 7) and (colidx == 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-Bold" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 3):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)

            sectionHeaderStyle3 = [
                ('BOTTOMPADDING', (0, 8), (-1, -1), 40),
                ('TOPPADDING', (0, 0), (-1, -8), 20),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 0), (-1, -1), '#a8d7c5')]
            t3Style1 = sectionHeaderStyle3
            t = Table(data, self.colwidths, style=TableStyle(t3Style1))
            # print('printing',t)
            story.append(t)
            story.append(PageBreak())

            print("self.constan_speed", self.constant_speed)


            if (self.prepared_basis == "CP Speed") & (self.waranted_weather_yes_no == "YES"):
                if (self.cp_ae_cons is not None and self.cp_ae_cons > 0):
                    cp_string = '<b>CP Warranties : </b>' + 'About ' + str(self.cp_speed) + ' Kts on ' + 'About ' + str(
                        self.cp_cons) + ' Mts FO + ' + str(self.cp_ae_cons) + ' Mt GO'
                else:
                    cp_string = '<b>CP Warranties : </b>' + 'About ' + str(self.cp_speed) + ' Kts on ' + 'About ' + str(
                        self.cp_cons) + ' Mts Fuel'

            elif (self.prepared_basis == "CP Speed") & (self.waranted_weather_yes_no == "NO"):
                if (self.cp_ae_cons is not None and self.cp_ae_cons > 0):
                    cp_string = '<b>CP Warranties : </b>' + '' + str(self.cp_speed) + ' Kts on ' + '' + str(
                        self.cp_cons) + ' Mts FO + ' + str(self.cp_ae_cons) + ' Mt GO'
                else:
                    cp_string = '<b>CP Warranties : </b>' + '' + str(self.cp_speed) + ' Kts on ' + '' + str(
                        self.cp_cons) + ' Mts Fuel'

            else:
                cp_string = '<b>CP Warranties : </b>' + 'Optimal Speed & Consumption'

            rawdata = [['', '<b>VOYAGE MAP</b>', ''],
                       ['', '', ''],
                       ['', '<b>Itinerary : </b>' + from_port + ' - ' + to_port, ''],
                       # ['','','<b>Voyage Leg Date(UTC) : </b>'+ str(self.departure_time_str) +' - '+ str(self.arrival_time_str),'',''],
                       ['', '<b>Voyage Leg Date(UTC) : </b>' + str(self.Dep_Date) + " " + str(
                           self.departure_time_str) + ' - ' + str(self.Arrival_Date) + " " + str(self.arrival_time_str),
                        ''],
                       ['', cp_string, '']]
            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-Bold" size="16" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'

                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths5, style=TableStyle(t3Style1))
            story.append(t)

            print("folder name", self.folder_name)
            # I_map = rImage('../content/drive/My Drive/Colab Notebooks/images/'+self.folder_name+'.png')
            story.append(space2)
            I_map = rImage('./images/' + self.folder_name + '.png')
            # I_map.drawHeight = 11*inch*I_map.drawHeight / I_map.drawWidth
            # I_map.drawWidth = 8*inch*0.8

            image_width, image_height = I_map.drawWidth, I_map.drawHeight
            image_aspect = image_height / float(image_width)
            print("Map Image Width:" + str(image_width))
            print("Map Image Height:" + str(image_height))

            # Determine the dimensions of the image in the overview
            print_width = A4[0] * 0.89
            print_height = print_width * image_aspect
            if (print_height > 500):
                print_height = 500
            I_map.drawWidth = print_width
            I_map.drawHeight = print_height
            story.append(I_map)
            story.append(PageBreak())

            # Report analysis summary start page 1
            story.append(space3)
            rawdata = [['', '<b>Report Analysis Summary</b>', ''],
                       ['', '<b>Itinerary : </b>' + from_port + ' - ' + to_port, ''],
                       ['', '<b>Voyage Leg Date(UTC) : </b>' + str(self.Dep_Date) + " " + str(
                           self.departure_time_str) + ' - ' + str(self.Arrival_Date) + " " + str(self.arrival_time_str),
                        '']]

            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 1):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths5, style=TableStyle(t3Style1))
            story.append(t)
            story.append(space2)
            story.append(space2)

            print("time_gain_loss1", self.total_time_loss, self.total_time_gain)

            if (self.total_time_loss == 0 and self.total_time_gain == 0):
                time_str = "Nil"

            elif math.isnan(self.total_time_loss) & math.isnan(self.total_time_gain):
                time_str = "Nil"

            elif (self.total_time_loss == 0):
                time_str = str(self.total_time_gain.round(2)) + " hours(Gain)"
            else:
                time_str = str(self.total_time_loss.round(2)) + " hours(Loss)"

            print("time_str", time_str)

            print("fuel_gain_loss1", self.total_fuel_loss, self.total_fuel_gain)

            if (self.total_fuel_loss == "" and self.total_fuel_gain == ""):
                fuel_string = "Nil"

            elif math.isnan(self.total_fuel_loss) & math.isnan(self.total_fuel_gain):
                fuel_string = "Nil"

            elif (self.total_fuel_gain != 0) & (self.total_fuel_loss == 0):
                fuel_string = str(self.total_fuel_gain.round(2)) + " mt(Gain)"


            elif (self.total_fuel_loss == 0):  # &(self.total_fuel_gain==0):
                fuel_string = "Nil"

            else:
                fuel_string = str(self.total_fuel_loss.round(2)) + " mt(Loss)"

            print("fuel_string", fuel_string)

            if (self.fuel_type_used == "HSFO"):
                v_u_l_sfo = "Nil"
                hsfo_val = fuel_string
                mgo_val = "Nil"
                mdo_val = "NA"
            else:
                v_u_l_sfo = fuel_string
                hsfo_val = "Nil"
                mgo_val = "Nil"
                mdo_val = "NA"

            #         rawdata = [['ATD(Z)','Speed Analysis','V/U/L SFO Analysis','HSFO Analysis','MGO Analysis','MDO Analysis'],
            #                     [from_port+' - '+to_port+" \n"+str(self.Dep_Date)+" "+str(self.departure_time_str),time_str,v_u_l_sfo,hsfo_val,mgo_val,mdo_val]]

            rawdata = [
                ['ATD(Z)', 'Time gain/loss', 'V/U/L SFO gain/loss', 'HSFO gain/loss', 'MGO gain/loss', 'MDO gain/loss'],
                [from_port + ' - ' + to_port + " \n" + str(self.Dep_Date) + " " + str(self.departure_time_str), time_str,
                 v_u_l_sfo, hsfo_val, mgo_val, mdo_val]]

            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['tablecontents']['datatext'], size['tablecontents']['datatext'], \
                        colour['tablecontents']['datatext']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 1):
                        f, s, c = font['tablecontents']['datatext'], size['tablecontents']['datatext'], \
                        colour['tablecontents']['datatext']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = [
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('BACKGROUND', (0, 0), (-1, 0), '#a8d7c5'),
                ('BACKGROUND', (2, 1), (-1, -1), '#4fb28b'),  ## middle alighment, # first row background colour light grey
                ('LINEBELOW', (0, 0), (0, 0), size['sectionheader']['line'],
                 colors.HexColor(colour['sectionheader']['line']))]

            # t3Style1 = sectionHeaderStyle4
            if (self.total_time_loss == ""):
                t3Style1.add('TEXTCOLOR', (1, 1), (1, 1), '#c3677d')
            elif self.total_time_loss < 0:
                t3Style1.add('TEXTCOLOR', (1, 1), (1, 1), '#c3677d')

            for row, values in enumerate(rawdata):
                print("raw-data")
                print(rawdata)
                for column, value in enumerate(values):
                    print("column, value")
                    print(column, value)
                    if ("Loss" in value):
                        t3Style1.append(('BACKGROUND', (column, row), (column, row), "#F59C9D"))
                    elif ("Gain" in value):
                        t3Style1.append(('BACKGROUND', (column, row), (column, row), "#5EBA7D"))
                    elif ("Nil" in value or "NA" in value):
                        t3Style1.append(('BACKGROUND', (column, row), (column, row), "#ffffff"))
            print("--------------------------------")
            print(t3Style1)
            print("--------------------------------")

            t = Table(data, self.colwidths4, style=TableStyle(t3Style1))
            story.append(t)
            story.append(space2)
            story.append(space2)

            f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader']['text']
            story.append(Paragraph(f'<font name="Play-{f}" size="12" color="{c}"> Voyage Details</font>'))
            story.append(space2)
            story.append(space2)

            I_vd = rImage("./png/report_analysis_summary_voyage_details.png")
            image_width, image_height = I_vd.drawWidth, I_vd.drawHeight
            image_aspect = image_height / float(image_width)
            print_width = A4[0] * 0.89
            print_height = print_width * image_aspect
            I_vd.drawWidth = print_width
            I_vd.drawHeight = print_height
            story.append(I_vd)
            story.append(space2)
            story.append(Paragraph(f'<font name="Play-{f}" size="12" color="{c}">  Warranted Consumption</font>'))
            story.append(space2)
            story.append(space2)

            if (self.constant_speed):
                cp_speed_string = str(self.cp_speed) + " kts"
                if (self.cp_ae_cons is not None and self.cp_ae_cons > 0):
                    cp_cons_string = str(self.cp_cons) + ' Mts FO + ' + str(self.cp_ae_cons) + ' Mt GO'
                else:
                    cp_cons_string = str(self.cp_cons) + ' MT'
            else:
                cp_speed_string = ' Optimal Speed'
                cp_cons_string = ' Optimal Consumption'

            if self.waranted_weather_yes_no == 'YES':
                rawdata = [['Leg Details', 'CP Speed', 'Total Cons.'],
                           [from_port + ' to ' + to_port + " ", f"About {cp_speed_string}", f"About {cp_cons_string}"]]
            else:

                rawdata = [['Leg Details', 'CP Speed', 'Total Cons.'],
                           [from_port + ' to ' + to_port, cp_speed_string, cp_cons_string]]

            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['tablecontents']['datatext'], size['tablecontents']['datatext'], \
                        colour['tablecontents']['datatext']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 1):
                        f, s, c = font['tablecontents']['datatext'], size['tablecontents']['datatext'], \
                        colour['tablecontents']['datatext']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle5
            t = Table(data, self.colwidths3, style=TableStyle(t3Style1))
            story.append(t)
            story.append(PageBreak())

            story.append(space3)
            story.append(Paragraph(
                f'<para align=center><font name="Play-Bold" size="14" color="{c}"><b> Report Analysis Summary</b></font></para>'))

            story.append(space2)
            story.append(space2)
            story.append(space2)
            story.append(space2)

            # ------

            line3 = ""
            if (bad_weather_period_definition_as_per_CP_in_hrs < 24):
                line3 = "A noon report is counted as fair weather if majority of the noon period is good weather basis analyzed weather"
            else:
                line3 = "A noon report is counted as fair weather if 24 hours of continuous good weather is observed in analyzed weather"
            filter_type_str = ""
            if (report_type == "DSS"):
                filter_type_str = ", DSS =" + str(bf_limit_dss) + ""
            elif (report_type == "SWH" or report_type == "SWH+DSS"):
                filter_type_str = ", Significant Wave Height <= " + str(swh_limit) + " m"
            elif (report_type == "BF"):
                filter_type_str = ""
            line5 = ""
            if (self.adverse_current == "excluded"):
                line5 = "\t• Adverse Currents are excluded"
            else:
                line5 = "\t• Adverse Currents are not excluded"
            line9 = ""
            if (prepared_basis == "CP Speed"):
                line9 = "All comparisons are done against CP Speed"
            else:
                line9 = "All comparisons are done against Daily CP Speed"
                line10 = ""
            line11 = ""
            line12 = "***Note: The calculations for the report are done on the performed speed by adjusting the effect of currents (If applicable)."
            if (extrapolation_Allowed == "YES"):
                line11 = "Good weather performance is extrapolated to overall voyage"
            else:
                line11 = "Good weather performance is not extrapolated to overall voyage"

            if (waranted_weather_yes_no == "YES"):
                # line10 = "“About” Tolerance:\n"+"\t•  For speed : -"+str(about_speed_min_tolerance)+" Kts to +"+str(about_speed_max_tolerance)+" Kts\n"+"\t•  For consumption : -"+str(about_cons_min_tolerance)+" % to +"+str(about_cons_max_tolerance)+" %\n\n"
                line100 = "“About” Tolerance:"

                if tolerance == 0:
                    line101 = "    •  For speed : -" + str(current_tolerance) + " Kts"
                elif current_tolerance == 0:
                    line101 = "    •  For speed : -" + str(tolerance) + " Kts"
                else:
                    line101 = "    •  For speed : -" + str(current_tolerance) + " / +" + str(tolerance) + " Kts"

                if mistolerance == 0:
                    line102 = "    •  For consumption : -" + str(About_Cons_MaxTolerence) + " %"
                elif About_Cons_MaxTolerence == 0:
                    line102 = "    •  For consumption : -" + str(mistolerance) + " %"
                else:
                    line102 = "    •  For consumption : -" + str(mistolerance) + " / +" + str(
                        About_Cons_MaxTolerence) + " %"

                # line103 = ""
                cp_summary = ["Interpretation of good weather criteria as per CP:", "Weather Definition:", line3,
                              "\t•  Wind Force <= " + str(bf_limit) + " Bf" + filter_type_str, line5,
                              "Noon Report excluded from evaluation :", "Weather Source : Analyzed",
                              "Speed used for Analysis : Performed speed", line9, "", line100, line101, line102, line11,
                              line12]

            if (waranted_weather_yes_no == "NO"):
                cp_summary = ["Interpretation of good weather criteria as per CP:", "Weather Definition:", line3,
                              "\t•  Wind Force <= " + str(bf_limit) + " Bf" + filter_type_str, line5,
                              "Noon Report excluded from evaluation :", "Weather Source : Analyzed",
                              "Speed used for Analysis : Performed speed", line9, line11, line12]

            # ------

            bold_texts = ["Weather Definition:", "Noon Report excluded from evaluation :", "“About” Tolerance:",
                          "About Tolerance:"]
            for i in range(0, 20):
                try:
                    text_to_be_pasted = cp_summary[i]  # cp_worksheet.cell_value(i,0)
                    # print("-"+text_to_be_pasted+"-")
                    if (text_to_be_pasted.strip() in bold_texts):
                        story.append(Paragraph(
                            f'<font name="Play-Bold" size="12" color="{c}"><b>' + text_to_be_pasted + '</b></font>'))
                        story.append(space2)
                    elif (text_to_be_pasted.strip() == ""):
                        story.append(space2)
                    else:
                        story.append(
                            Paragraph(f'<font name="Play-{f}" size="12" color="{c}">' + text_to_be_pasted + '</font>'))
                        story.append(space2)
                    # cp_text_list.append(worksheet.cell_value(i,0))
                except Exception as e:
                    print(e)
                    print()
            story.append(PageBreak())

            # Report analysis summary end page 2

            # ----------------
            # Page3
            # page fuel consumption summary
            #         if(self.constant_speed):
            #             if (self.cp_ae_cons is not None and self.cp_ae_cons>0):
            #                 cp_string = '<b>CP Warranties : </b>'+ 'About '+'str(cp_speed)'+' Kts on '+'About '+'str(cp_cons)'+' Mts FO + '+'str(cp_ae_cons)'+' Mt GO'
            #             else:
            #                 cp_string = '<b>CP Warranties : </b>'+ 'About '+'str(cp_speed)'+' Kts on '+'About '+'str(cp_cons)'+' Mts Fuel'
            #         else:
            #                 cp_string = '<b>CP Warranties : </b>'+ 'Optimal Speed & Consumption'
            story.append(space3)
            #speed_logo = '<img src="C:Users/DELL/PycharmProjects/performance_form/static/Speedometer(1).jpg" valign="middle" width = "25" height="25"/>'
            rawdata = [['',  '<b>Speed Summary</b>', ''],
                       ['', '', ''],
                       ['', '<b>Itinerary : </b>' + from_port + ' - ' + to_port, ''],
                       ['', '<b>Voyage Leg Date(UTC) : </b>' + str(self.Dep_Date) + " " + str(
                           self.departure_time_str) + ' - ' + str(self.Arrival_Date) + " " + str(self.arrival_time_str),
                        ''],
                       ['', cp_string, '']]
            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths5, style=TableStyle(t3Style1))
            story.append(t)
            story.append(space3)
            story.append(space2)
            rawdata = [['', '', '<b>Overall</b>', '', ''],
                       ['', '', '', '', ''],
                       ['', 'Total Distance Sailed', '', str(round(self.total_distance, 2)) + " NM", ''],
                       ['', 'Time at Sea', '', str(round(self.total_time, 2)) + " hrs", ''],
                       ['', 'Average Speed', '', str(round(self.average_speed, 2)) + " kts", '']]
            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{11}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths2, style=TableStyle(t3Style1))
            story.append(t)

            story.append(space3)

            print("page_3", self.track_time_loss, self.track_time_gain, self.total_time_loss)

            track_time_action = ""
            if (self.track_time_loss == 0):
                track_time_action = "Gain"
                track_time_val = self.track_time_gain
            else:
                track_time_action = "Loss"
                track_time_val = self.track_time_loss

            total_time_action = ""
            if (self.total_time_loss == 0):
                total_time_action = "Gain"
                total_time_val = self.total_time_gain
                print("A")

            elif math.isnan(self.total_time_loss):
                total_time_action = "Gain"
                total_time_val = np.array(0)
                print("B")

            else:
                total_time_action = "Loss"
                total_time_val = self.total_time_loss
                print("C")

            print("self.current_factor", self.current_factor)

            # if(adverse_curr_excluded=="Excluded" or adverse_curr_excluded=="Yes"):
            if (self.gwx_type == "x"):
                rawdata = [['', '', '<b>Good Weather</b>', '', ''],
                           ['', '', '', '', ''],
                           ['', 'Total Distance Sailed', '', str(round(self.Good_Weather_Distance, 2)) + " NM", ''],
                           ['', 'Time at Sea', '', str(round(self.Good_Weather_Time, 2)) + " hrs", ''],
                           ['', 'Average Speed', '', str(round(self.Average_Speed_In_Good_Weather, 2)) + " kts", ''],
                           ['', 'C/P Min.Allowable Time', '', str(self.Min_Total_Allowed_Time_in_GW.round(2)) + " hrs", ''],
                           ['', 'C/P Max.Allowable Time', '', str(self.Max_Total_Allowed_Time_in_GW.round(2)) + " hrs", ''],
                           ['', 'Track Time ' + track_time_action, '', str(track_time_val.round(2)) + " hrs", ''],
                           ['', 'Applied to Overall Track Time ' + total_time_action, '',
                            str(total_time_val.round(2)) + " hrs", '']]
            # else:
            # elif(adverse_curr_excluded=="Not Excluded" or adverse_curr_excluded=="No"):
            else:
                rawdata = [['', '', '<b>Good Weather</b>', '', ''],
                           ['', '', '', '', ''],
                           ['', 'Total Distance Sailed', '', str(round(self.Good_Weather_Distance, 2)) + " NM", ''],
                           ['', 'Time at Sea', '', str(round(self.Good_Weather_Time, 2)) + " hrs", ''],
                           ['', 'Average Speed', '', str(round(self.Average_Speed_In_Good_Weather, 2)) + " kts", ''],
                           ['', 'Current Factor', '', str(round(self.current_factor, 2)) + " kts", ''],
                           ['', 'Performance Speed', '', str(round(self.B132, 2)) + " kts", ''],
                           ['', 'C/P Min.Allowable Time', '', str(self.Min_Total_Allowed_Time_in_GW.round(2)) + " hrs", ''],
                           ['', 'C/P Max.Allowable Time', '', str(self.Max_Total_Allowed_Time_in_GW.round(2)) + " hrs", ''],
                           ['', 'Track Time ' + track_time_action, '', str(track_time_val.round(2)) + " hrs", ''],
                           ['', 'Applied to Overall Track Time ' + total_time_action, '',
                            str(total_time_val.round(2)) + " hrs", '']]
            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{11}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths2, style=TableStyle(t3Style1))
            story.append(t)

            story.append(space2)
            story.append(PageBreak())

            # ----Fuel page

            story.append(space3)
            #fuel_logo = '<img src="./static/fuellogo(1).jpg" valign="middle" width = "25" height="25"/>'
            rawdata = [['', '<b>Fuel Consumption Summary</b>', ''],
                       ['', '', ''],
                       ['', '<b>Itinerary : </b>' + from_port + ' - ' + to_port, ''],
                       ['', '<b>Voyage Leg Date(UTC) : </b>' + str(self.Dep_Date) + " " + str(
                           self.departure_time_str) + ' - ' + str(self.Arrival_Date) + " " + str(self.arrival_time_str),
                        ''],
                       ['', cp_string, '']]
            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths5, style=TableStyle(t3Style1))
            story.append(t)
            story.append(space3)

            # Test start

            #         if math.isnan(self.Total_Average_Daily_Consumption):
            #             self.Total_Average_Daily_Consumption=0.00

            # Test End

            rawdata = [['', '', '<b>Overall</b>', '', ''],
                       ['', '', '', '', ''],
                       ['', 'Average Daily Consumption', '', str(round(self.Total_Average_Daily_Consumption, 2)) + " mts",
                        ''],
                       ['', 'Total Bunkers Consumed at Sea', '', str(round(self.Total_bunkers_Consumed_at_sea, 2)) + " mts",
                        ''],
                       ['', 'Gradewise Distribution of Bunkers consumed at sea', '', '', ''],
                       ['', 'HSFO', '', str(round(self.hdfo_total_con, 2)) + " mts", ''],
                       ['', 'IFO', '', str(round(self.ifo_total_con, 2)) + " mts", ''],
                       ['', 'GO', '', str(round(self.Total_Bunker_GO_Consumption, 2)) + " mts", '']]

            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{11}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths2, style=TableStyle(t3Style1))
            story.append(t)

            print("fuel_page", self.total_fuel_loss, self.total_fuel_gain)

            if (self.total_fuel_loss == 0 and self.total_fuel_gain == 0):
                fuel_string1 = "No Fuel Gain/Loss"
                fuel_val1 = 0.0
            elif (self.total_fuel_loss == 0):
                fuel_string1 = "Fuel Gain"
                fuel_val1 = self.GW_Fuel_Gain
            else:
                fuel_string1 = "Fuel Loss"
                fuel_val1 = self.GW_Fuel_Loss

            print("fuel_page1", self.GW_Fuel_Loss, self.GW_Fuel_Gain)

            if (self.GW_Fuel_Loss == 0 and self.GW_Fuel_Gain == 0):
                fuel_string2 = "No Loss/Gain applied to overall track"
                fuel_val2 = np.array(0)
                print("A")
            elif (self.GW_Fuel_Loss == 0):
                fuel_string2 = "Fuel Gain applied to overall track"
                fuel_val2 = self.total_fuel_gain
                print("B")
            else:
                fuel_string2 = "Fuel Loss applied to overall track"
                fuel_val2 = self.total_fuel_loss
                print("C")
            # test

            if math.isnan(self.B168):
                self.B168 = 0.00
                # test end

            story.append(space3)
            rawdata = [['', '', '<b>Good Weather</b>', '', ''],
                       ['', '', '', '', ''],
                       ['', 'Actual Usage in Good Weather', '', str(self.B167) + " mts", ''],
                       ['', 'Average Daily Consumption', '', str(self.B168) + " mts", ''],
                       ['', 'Min.Allowable Usage', '', str(self.Min_Allowable_Usage) + " mts", ''],
                       ['', 'Max Allowable Usage', '', str(self.Max_Allowable_Usage) + " mts", ''],
                       ['', fuel_string1, '', str(fuel_val1) + " mts", ''],
                       ['', fuel_string2, '', str(fuel_val2) + " mts", '']]
            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{11}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths2, style=TableStyle(t3Style1))
            story.append(t)
            # good weather end
            # emission summary start
            # total_co2_produced = HSFO*co2_factor_hsfo+IFO*co2_factor_hsfo+GO*co2_factor_mdo
            total_co2_produced = self.hdfo_total_con * co2_factor_hsfo + self.ifo_total_con * co2_factor_hsfo + self.Total_Bunker_GO_Consumption * co2_factor_mdo
            #co2 = '<img src="./static/co2(1).jpg" valign="middle" width = "25" height="25"/>'
            rawdata = [['', '',  '<b>CO2 Emissions Summary</b>', '', ''],
                       ['', '', '', '', ''],
                       ['', '', '<b>Overall</b>', '', ''],
                       ['', '', '', '', ''],
                       ['', 'Total CO2 produced at sea (MT)', '', str(total_co2_produced.round(2)) + " mts", '']]

            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{11}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths2, style=TableStyle(t3Style1))
            story.append(t)
            story.append(PageBreak())

            # emission summary end
            # speed summary start

            # speed summary end

            # voyage summary start
            if (self.constant_speed):
                story.append(space3)
                rawdata = [['', '<b>Voyage Summary</b>', ''],
                           ['', '', ''],
                           ['', '<b>Itinerary : </b>' + from_port + ' - ' + to_port, ''],
                           ['', '<b>Voyage Leg Date(UTC) : </b>' + str(self.Dep_Date) + " " + str(
                               self.departure_time_str) + ' - ' + str(self.Arrival_Date) + " " + str(self.arrival_time_str),
                            ''],
                           ['', cp_string, '']]

                data = []
                for row in rawdata:
                    newrow = []
                    rowidx = rawdata.index(row)
                    for col in row:
                        colidx = row.index(col)
                        # print (col)
                        if (rowidx == 0):  ## section header
                            # print(col)
                            f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                                'text']
                            newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                                col) + '</font></para>'
                        elif (rowidx >= 2):
                            f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                            newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                                col) + '</font></para>'
                        else:
                            newdata = col
                        newrow.append(Paragraph(newdata))
                    data.append(newrow)
                t3Style1 = sectionHeaderStyle
                t = Table(data, self.colwidths5, style=TableStyle(t3Style1))
                # story.append(t)
                # story.append(space2)

                # voyage summary end

                I_vs = rImage("./png/voyage_details_summary.png")
                image_width, image_height = I_vs.drawWidth, I_vs.drawHeight
                image_aspect = image_height / float(image_width)
                print_width = A4[0] * 0.7
                print_height = (print_width) * image_aspect
                I_vs.drawWidth = print_width
                I_vs.drawHeight = print_height

                # Voyage details
            if (self.prepared_basis == "Optimal Speed"):
                story.append(space3)
                rawdata = [['', '<b>Voyage Details</b>', ''],
                           ['', '', ''],
                           ['', '<b>Itinerary : </b>' + from_port + ' - ' + to_port, ''],
                           ['', '<b>Voyage Leg Date(UTC) : </b>' + str(self.Dep_Date) + " " + str(
                               self.departure_time_str) + ' - ' + str(self.Arrival_Date) + " " + str(self.arrival_time_str),
                            ''],
                           ['', cp_string, '']]

                data = []
                for row in rawdata:
                    newrow = []
                    rowidx = rawdata.index(row)
                    for col in row:
                        colidx = row.index(col)
                        # print (col)
                        if (rowidx == 0):  ## section header
                            # print(col)
                            f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                                'text']
                            newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                                col) + '</font></para>'
                        elif (rowidx >= 2):
                            f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                            newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                                col) + '</font></para>'
                        else:
                            newdata = col
                        newrow.append(Paragraph(newdata))
                    data.append(newrow)
                t3Style1 = sectionHeaderStyle

                t = Table(data, self.colwidths5, style=TableStyle(t3Style1))
                story.append(t)
                story.append(space2)

                print("ceiling_voyage_detail", math.ceil(self.no_of_page_voyage))

                for i in range(math.ceil(self.no_of_page_voyage)):
                    print(i)
                    I_vd1 = rImage("./png/voyage_details_table" + str(i + 1) + ".png")
                    image_width, image_height = I_vd1.drawWidth, I_vd1.drawHeight
                    print("Original Width1, Height:")
                    print((image_width, image_height))
                    image_aspect = image_height / float(image_width)

                    # Determine the dimensions of the image in the overview
                    print_width = A4[0] * 0.9
                    print_height = print_width * image_aspect
                    print("Image Aspect:" + str(image_aspect))
                    print("Print Width, Height:")
                    print((print_width, print_height))
                    I_vd1.drawWidth = print_width
                    I_vd1.drawHeight = print_height
                    story.append(I_vd1)
                    story.append(space2)
                    if math.ceil(self.no_of_page_voyage) == i:
                        story.append(PageBreak())

                    # voyage detail end

                    I_legend = rImage("./static/Legend(1).png")
                    image_width, image_height = I_legend.drawWidth, I_legend.drawHeight
                    image_aspect = image_height / float(image_width)
                    print_width = A4[0] * 0.50
                    print_height = print_width * image_aspect

                    I_legend.drawWidth = print_width
                    I_legend.drawHeight = print_height
                    story.append(I_legend)

                    story.append(space2)
                    story.append(PageBreak())

            # ---Weather page

            rawdata = [['', '<b>Detailed Weather Analysis</b>', ''],
                       ['', '', ''],
                       ['', '<b>Itinerary : </b>' + from_port + ' - ' + to_port, ''],
                       ['', '<b>Voyage Leg Date(UTC) : </b>' + str(self.Dep_Date) + " " + str(
                           self.departure_time_str) + ' - ' + str(self.Arrival_Date) + " " + str(self.arrival_time_str),
                        ''],
                       ['', cp_string, '']]
            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths5, style=TableStyle(t3Style1))
            story.append(t)

            print("ceiling_weather", math.ceil(self.no_of_page_weather))

            for i in range(math.ceil(self.no_of_page_weather)):
                # story.append(space2)

                if i == 0:
                    I_dw = rImage("./png/weather_detail" + str(i + 1) + ".png")
                    image_width, image_height = I_dw.drawWidth, I_dw.drawHeight
                    print("Original Width2, Height:")
                    print((image_width, image_height))
                    image_aspect = image_height / float(image_width)

                    # Determine the dimensions of the image in the overview
                    print_width = A4[0] * 0.89
                    print_height = print_width * image_aspect
                    print("Image Aspect:" + str(image_aspect))
                    print("Print Width2, Height:")
                    print((print_width, print_height))
                    I_dw.drawWidth = print_width
                    I_dw.drawHeight = print_height
                    story.append(I_dw)
                    story.append(PageBreak())
                else:
                    story.append(space2)
                    story.append(space3)
                    story.append(space2)

                    I_dw = rImage("./png/weather_detail" + str(i + 1) + ".png")
                    image_width, image_height = I_dw.drawWidth, I_dw.drawHeight
                    print("Original Width2, Height:")
                    print((image_width, image_height))
                    image_aspect = image_height / float(image_width)

                    # Determine the dimensions of the image in the overview
                    print_width = A4[0] * 0.89
                    print_height = print_width * image_aspect
                    print("Image Aspect:" + str(image_aspect))
                    print("Print Width2, Height:")
                    print((print_width, print_height))
                    I_dw.drawWidth = print_width
                    I_dw.drawHeight = print_height
                    story.append(I_dw)
                    story.append(PageBreak())

            #             if i!=0:
            #                 story.append(PageBreak())

            story.append(space3)
            rawdata = [['', '<b>Good Weather Summary</b>', ''],
                       ['', '', ''],
                       ['', '<b>Itinerary : </b>' + from_port + ' - ' + to_port, ''],
                       ['', '<b>Voyage Leg Date(UTC) : </b>' + str(self.Dep_Date) + " " + str(
                           self.departure_time_str) + ' - ' + str(self.Arrival_Date) + " " + str(self.arrival_time_str),
                        ''],
                       # ['','','<b>Voyage Leg Date(UTC) : </b>'+  str(self.departure_time_str) +' - '+ str(self.arrival_time_str),'',''],
                       ['', cp_string, '']]
            data = []
            for row in rawdata:
                # print(row)
                newrow = []
                rowidx = rawdata.index(row)
                # print(rowidx)
                for col in row:
                    colidx = row.index(col)
                    # print (col)
                    if (rowidx == 0):  ## section header
                        # print(col)
                        f, s, c = font['sectionheader']['text'], size['sectionheader']['text'], colour['sectionheader'][
                            'text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                    elif (rowidx >= 2):
                        f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                        newdata = f'<para align=center><font name="Play-{f}" size="{s}" color="{c}">' + str(
                            col) + '</font></para>'
                        #         elif (rowidx >= 2):
                    #             f, s, c = font['subtitle']['text'], size['subtitle']['text'], colour['subtitle']['text']
                    #             newdata = f'<para align=left><font name="Play-{f}" size="{s}" color="{c}">'+ str(col) + '</font></para>'
                    else:
                        newdata = col
                    newrow.append(Paragraph(newdata))
                data.append(newrow)
            t3Style1 = sectionHeaderStyle
            t = Table(data, self.colwidths5, style=TableStyle(t3Style1))
            story.append(t)
            # story.append(space3)
            print("test1")
            I_gw = rImage("./png/weather_detail_summary.png")

            image_width, image_height = I_gw.drawWidth, I_gw.drawHeight
            image_aspect = image_height / float(image_width)
            print_width = A4[0] * 0.89
            print_height = print_width * image_aspect
            I_gw.drawWidth = print_width
            I_gw.drawHeight = print_height
            story.append(I_gw)
            story.append(PageBreak())
            print("test2")

            story.append(space3)

            story.append(Paragraph(
                f'<para align=center><font name="Play-Bold" size="17" color="{c}"><b>Message Traffic</b></font></para>'))
            story.append(space3)
            #         for ind in range(mt_count):
            #         if(ind!=0):
            #           story.append(space3)
            #           #story.append(space3)
            #           #story.append(space3)
            I_mt = rImage("./png/message_traffic_summary.png")
            # I_mt.drawHeight = 10*inch*I_mt.drawHeight / I_mt.drawWidth
            # I_mt.drawWidth = 8*inch*0.9
            image_width, image_height = I_mt.drawWidth, I_mt.drawHeight
            # print("Original Width, Height:")
            # print((image_width, image_height))
            image_aspect = image_height / float(image_width)

            # Determine the dimensions of the image in the overview
            print_width = A4[0] * 0.89
            print_height = print_width * image_aspect
            # print("Image Aspect:"+str(image_aspect))
            # print("Print Width, Height:")
            # print((print_width,print_height))
            # if(ind!=0):
            I_mt.drawWidth = print_width
            I_mt.drawHeight = print_height
            # else:
            I_mt.drawWidth = print_width
            I_mt.drawHeight = print_height
            story.append(I_mt)
            story.append(PageBreak())

            print("test3")

            def addPageNumber(canvas, doc):
                """
                Add the page number
                """
                #     canvas.setStrokeColorRGB(0.7,0.5,0.3)
                #     canvas.line(7.3*inch, 11.0*inch, 7.7*inch, 11.4*inch)
                #     canvas.setFillColorRGB(0,145,23)
                #     canvas.ellipse(7.3*inch, 11.0*inch, 7.7*inch, 11.4*inch, stroke=1, fill=1)
                canvas.saveState()
                canvas.setFillColorRGB(0, 0, 0)
                canvas.setStrokeColorRGB(0, 0, 0)
                canvas.setFont("Play-Regular", 6)
                # logo = "ZN logo.jpg"
                # w, h = logo.wrap(doc.width, doc.topMargin)
                im = PIL.Image.open("./static/ZN Logo (2).jpg")
                canvas.drawInlineImage(im, 470, 780, width=100, height=50)
                im2 = PIL.Image.open("./static/mgtg (1).jpg")
                canvas.drawInlineImage(im2, 240, 25, width=100, height=10)

                current_page = canvas.getPageNumber()  # -- adding page numbers
                page_num_text = f"Page {current_page}"

                canvas.drawRightString(7.6 * inch, 0.4 * inch, page_num_text)

                # logo.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h)
                # canvas.drawImage(logo, 500,330)
                # canvas.line(0.5*inch,0.5*inch,7.8*inch,0.5*inch)
                # page_num = "Page %s" % canvas.getPageNumber()
                # page_num = "Page %s" % canvas.getNumPages()
                # canvas.drawRightString(7.5*inch, 0.4*inch, page_num)
                # current_time = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                # generated = f'Alpha Ori Technologies {current_time}'
                # canvas.drawRightString(2.2*inch, 0.4*inch, generated)
                canvas.restoreState()
                print("test777")

            story.append(space3)
            story.append(Paragraph(
                f'<para align=center><font name="Play-Bold" size="17" color="{c}"><b>Fuel Graph</b></font></para>'))
            story.append(space)
            story.append(Paragraph(
                f'<para align=center><font name="Play-{f}" size="14" color="{c}"><b>Comparison between Actual vs Allowed IFO Cons.</b></font></para>'))
            story.append(space)
            I_bc1 = rImage("./png/bar_chart1.png")
            I_bc1.drawHeight = 5 * inch * I_bc1.drawHeight / I_bc1.drawWidth
            I_bc1.drawWidth = 6 * inch * 0.9
            story.append(I_bc1)

            story.append(space3)
            story.append(Paragraph(
                f'<para align=center><font name="Play-Bold" size="17" color="{c}"><b>Steaming Graph</b></font></para>'))
            story.append(space)
            story.append(Paragraph(
                f'<para align=center><font name="Play-{f}" size="14" color="{c}"><b>Comparison between Actual vs Allowed Steaming</b></font></para>'))
            story.append(space)
            I_bc2 = rImage("./png/bar_chart2.png")
            I_bc2.drawHeight = 5 * inch * I_bc2.drawHeight / I_bc2.drawWidth
            I_bc2.drawWidth = 6 * inch * 0.9
            story.append(I_bc2)
            story.append(PageBreak())
            print("test4")

            # Page11
            story.append(space3)
            story.append(Paragraph(
                f'<para align=center><font name="Play-{f}" size="17" color="{c}"><b>Annex A - Speed Calculation Detail</b></font></para>'))
            story.append(space3)
            story.append(space3)

            I_p1f = rImage("./static/page_1_formula (1).png")
            image_width, image_height = I_p1f.drawWidth, I_p1f.drawHeight
            image_aspect = image_height / float(image_width)
            print_width = A4[0] * 0.89
            print_height = print_width * image_aspect
            I_p1f.drawWidth = print_width
            I_p1f.drawHeight = print_height
            story.append(I_p1f)
            story.append(PageBreak())
            print("test5")

            # page13

            story.append(space3)
            story.append(Paragraph(
                f'<para align=center><font name="Play-{f}" size="17" color="{c}"><b>Annex B - Fuel Consumption Calculation Detail</b></font></para>'))
            story.append(space)
            # story.append(space3)
            I_p2f = rImage("./static/page_2_formula (1).png")
            image_width, image_height = I_p2f.drawWidth, I_p2f.drawHeight
            image_aspect = image_height / float(image_width)
            print_width = A4[0] * 0.89
            print_height = print_width * image_aspect
            I_p2f.drawWidth = print_width
            I_p2f.drawHeight = print_height
            # I_p2f.drawHeight = 6*inch*I_p2f.drawHeight / I_p2f.drawWidth
            # I_p2f.drawWidth = 8*inch*0.9
            story.append(I_p2f)
            story.append(space)
            story.append(Paragraph(
                f'<para align=center><font name="Play-{f}" size="17" color="{c}"><b>Annex C - CO2 Emission Calculation Detail</b></font></para>'))
            story.append(space)
            story.append(Paragraph(
                f'<para align=center><font name="Play-{f}" size="11" color="{c}">Total CO2 produced at sea (MT) =	Σ(bunker consumed x CO2 factor for particular grade)</font></para>'))
            story.append(space)
            story.append(Paragraph(
                f'<para align=center><font name="Play-{f}" size="9" color="{c}"><i>*all CO2 factors are considered as mentioned in IMO GHG Study 2020 (pg.74; Table 21)</i></font></para>'))
            story.append(PageBreak())

            # page12
            story.append(space3)
            story.append(Paragraph(
                f'<para align=center><font name="Play-{f}" size="17" color="{c}"><b>Weather DataSources</b></font></para>'))
            story.append(space3)
            story.append(Paragraph(
                f'<font name="Play-{f}" size="11" color="{c}">Our weather forecast is based on data from several sources including NOAA server along with<br /> two other agencies.The weather projection model consist of 05 days accurate weather forecast along<br /> with 09 days extended forecast. For subsequent days,information from historical weather database is used.</font>',
                orden))
            story.append(space2)
            story.append(
                Paragraph(f'<font name="Play-Bold" size="12" color="{c}"><b>WAVEWATCH III for Wind/Waves/Swell</b></font>'))
            story.append(space2)
            story.append(Paragraph(
                f'<font name="Play-{f}" size="11" color="{c}"><b>WAVEWATCH III is a third generation multi-grid wave model at NOAA/NCEP in the spirit of WAM model.</b></font>'))
            story.append(space2)
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><b>Update Interval : 6 Hours</b></font>', orden))
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><b>Average Resolution Time : 3 Hours</b></font>',
                          orden))
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><b>Time Period : 5 Days</b></font>', orden))
            story.append(Paragraph(
                f'<font name="Play-{f}" size="11" color="{c}"><b>Provider : NOAA (National Oceanic & Atmospheric Administration</b></font>',
                orden))
            story.append(space2)
            story.append(Paragraph(
                f'<font name="Play-Bold" size="12" color="{c}"><b>GEFS (Global Ensemble Forecast System) for Wind/Waves/Swell</b></font>'))
            story.append(space2)
            story.append(Paragraph(
                f'<font name="Play-{f}" size="11" color="{c}"><b>The Global Ensemble Forecast System (GEFS) is a weather forecast model made up of 21 separate forecast or ensemble members.</b></font>'))
            story.append(space2)
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><b>Update Interval : 6 Hours</b></font>', orden))
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><b>Average Resolution Time : 3 Hours</b></font>',
                          orden))
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><b>Time Period : 16 Days</b></font>', orden))
            story.append(Paragraph(
                f'<font name="Play-{f}" size="11" color="{c}"><b>Provider : NOAA (National Oceanic & Atmospheric Administration</b></font>',
                orden))
            story.append(space2)
            story.append(Paragraph(
                f'<font name="Play-Bold" size="12" color="{c}"><b>Copernicus Marine Environment Monitoring Service- for Sea Currents</b></font>'))
            story.append(space2)
            story.append(Paragraph(
                f'<font name="Play-{f}" size="11" color="{c}">The Copernicus Marine Environment Monitoring Service is part of the Copernicus Pro- gramme, which is an EU Programme managed by the European Commission (EC) and implemented in partnership with the Member States, the European Space Agency (ESA), the European Organisation for the Exploitation of Meteorological Satellites (EUMETSAT), the European Centre for medium-range Weather Forecasts (ECMWF), EU Agencies and Mercator Ocean. The Programme is aimed at developing a set of European information services based on satellite Earth Observation and in-situ (non-space) data.</font>',
                orden))
            story.append(space2)
            story.append(Paragraph(
                f'<font name="Play-{f}" size="11" color="{c}"><b>Spatial Resolution : 0.08 degree (Lat) x 0.08 degree (Lon)</b></font>',
                orden))
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><b>Temporal Resolution : Hourly mean</b></font>',
                          orden))
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><b>Time Period : 7 Days</b></font>', orden))
            story.append(
                Paragraph(f'<font name="Play-{f}" size="11" color="{c}"><strong>Provider : Copernicus</strong></font>',
                          orden))
            story.append(space3)
            print("test6")
            doc.build(story, onLaterPages=addPageNumber)


    my_obj = Pdf_process(filename,
                         green_processed_file,
                         prepared_basis,
                         constant_speed, weather_detail_table,
                         not_sure_L78,
                         performance_calculation,
                         current_tolerance,
                         tolerance,
                         mistolerance,
                         About_Cons_MaxTolerence,
                         extrapolation_Allowed,
                         fuel_type_used,
                         co2_factor_hsfo,
                         co2_factor_mdo,
                         co2_factor_lng,
                         co2_factor,
                         gwx_type,
                         adverse_current,
                         waranted_weather_yes_no,
                         GW_consumption_GO_added,
                         voyage_phase
                         )
    my_obj.make_directories()  # creating directories to store
    gwx_pdf_table = my_obj.good_weather_summary()  # creating good_weather_summary table , same as performance report for print purpose
    my_obj.colWidth()
    my_obj.colWidth2()
    my_obj.colWidth3()
    my_obj.colWidth4()
    my_obj.colWidth5()
    my_obj.generate_pdf(from_port, to_port, bad_weather_period_definition_as_per_CP_in_hrs,
                        report_type,
                        # current_excluded,
                        prepared_basis,
                        waranted_weather_yes_no,
                        extrapolation_Allowed,
                        current_tolerance,
                        tolerance,
                        mistolerance,
                        About_Cons_MaxTolerence,
                        bf_limit_dss
                        )


