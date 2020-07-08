# coding: utf-8
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import math

def sigmoid(x):
    return(1 / (1 + math.e**-x))

def get_inputdata(kiki, current, h_pressure, l_pressure, in_temp, out_temp, outside_temp):
    
    df_kiki = pd.read_csv('kiki_ave.csv', index_col = 0)
    df_kiki = df_kiki.loc[kiki]

    input_A = (current - df_kiki['ave_A']) / df_kiki['std_A']
    input_Ph = (h_pressure - df_kiki['ave_P_H1']) / df_kiki['std_P_H1']
    input_Pl = (l_pressure - df_kiki['ave_P_L1']) / df_kiki['std_P_L1']
    input_To = (outside_temp - df_kiki['ave_To']) / df_kiki['std_To']
    input_dP = ((h_pressure - l_pressure) - df_kiki['ave_dP']) / df_kiki['std_dP']
    input_dT = ((out_temp - in_temp) - df_kiki['ave_dT']) / df_kiki['std_dT']

    return(input_A, input_Ph, input_Pl, input_To, input_dP, input_dT)

def get_result(input_A, input_Ph, input_Pl, input_To, input_dP, input_dT):
    df_rst = pd.read_csv('learning_rst.csv', index_col = 0)
    sr_rst = pd.Series()

    for fx in df_rst.index:
        df_rst_i = df_rst.loc[fx]

        np_inp0 = np.array([[1, input_A, input_Ph, input_Pl, input_To, input_dP, input_dT]])
        np_w1 = np.array([df_rst_i.iloc[:7].tolist(), df_rst_i.iloc[7:14].tolist(), df_rst_i.iloc[14:21].tolist(), 
                          df_rst_i.iloc[21:28].tolist(), df_rst_i.iloc[28:35].tolist(),df_rst_i.iloc[35:42].tolist()])
        np_inp1 = sigmoid(np_inp0 @ np_w1.T)
        np_inp1 = [1, np_inp1[0][0],np_inp1[0][1],np_inp1[0][2],np_inp1[0][3],np_inp1[0][4],np_inp1[0][5]]
        np_w2 = np.array([df_rst_i.iloc[42:49].tolist(), df_rst_i.iloc[49:56].tolist(), df_rst_i.iloc[56:63].tolist(), 
                          df_rst_i.iloc[63:70].tolist(), df_rst_i.iloc[70:77].tolist(),df_rst_i.iloc[77:84].tolist()])
        np_out = sigmoid(np_inp1 @ np_w2.T)
        
        np_out = np.nan_to_num(np_out)
        sr_rst[fx] = int(np_out[1] * 100)

    return(sr_rst)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/output', methods=['POST'] )
def output():
    req = request.form
    input_A, input_Ph, input_Pl, input_To, input_dP, input_dT = get_inputdata(req['kiki'], float(req['current']), float(req['h_pressure']), float(req['l_pressure']), float(req['in_temp']), float(req['out_temp']), float(req['outside_temp']))
    score = get_result(input_A, input_Ph, input_Pl, input_To, input_dP, input_dT)
    return render_template('output.html', req = req, score = score)

@app.route('/send')
def send():
    return render_template('send.html')

if __name__ == '__main__':
    app.run(debug=True)