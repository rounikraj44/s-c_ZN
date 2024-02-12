from flask import Flask, session, abort, redirect, request,make_response
import json
from threading import Thread
import model
from functools import wraps
import os
app = Flask(__name__)

@app.route("/")
def main():
    return "welcome to s&c form entry"

@app.post('/scperformance/generate_report')
def generate_pdf():
    try:
        json_string = request.get_data()
        print(type(json_string))
        request_data = json.loads(json_string)
        print(type(request_data))

        filename_inp = request_data['ship']
        from_port_inp = request_data['pFrom']
        to_port_inp=request_data['pTo']
        prepared_basis_inp=request_data['typeSpeed']
        voyage_phase_inp=request_data['reportType']
        fuel_type_used_inp=request_data['cpFuel']
        waranted_weather_yes_no_inp = request_data['aboutSc']
        bf_limit_inp = int(request_data['windBeaufort'])
        windwave_limit_inp = float(request_data['windWave'])
        swell_height_inp= float(request_data['swellHeight'])
        swh_limit_inp = float(request_data['significantWave'])
        gwx_type_inp= request_data['adverseCurrent']
        not_sure_L78_inp = request_data['currentFactor']
        gwx_hours_inp= int(request_data['goodWeatherPeriod'])
        performance_calculation_inp = request_data['performanceCal']
        current_tolerance_inp = float(request_data['speedMinTol'])
        tolerance_inp = float(request_data['speedMaxTol'])
        mistolerance_inp = float(request_data['consMinTol'])
        About_Cons_MaxTolerence_inp = float(request_data['consMaxTol'])
        extrapolation_Allowed_inp = request_data['extrapolationMethod']
        report_type_inp = request_data['wetherCalculation']

    except Exception as e:
        print("Error Getting Parameters from the Request")
        print(e)
        return "Error Getting Parameters from the Request"

    thread = Thread(target=model.trigger_pdf,args=[filename_inp,from_port_inp,to_port_inp,prepared_basis_inp,voyage_phase_inp,fuel_type_used_inp,waranted_weather_yes_no_inp,bf_limit_inp,windwave_limit_inp,swell_height_inp,swh_limit_inp,gwx_type_inp,not_sure_L78_inp,gwx_hours_inp,performance_calculation_inp,current_tolerance_inp,tolerance_inp,mistolerance_inp,About_Cons_MaxTolerence_inp,extrapolation_Allowed_inp,report_type_inp])
    thread.start()
    print("Process Triggered...")
    print("Responding: Process Triggered...")
    return "Responding: Process Triggered..."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)