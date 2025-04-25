from flask import Flask,request,render_template
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

# Route for a home page
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(

            battery_power=request.form.get('battery_power'),
            blue=request.form.get('blue'),
            clock_speed=request.form.get('clock_speed'),
            dual_sim=request.form.get('dual_sim'),
            fc=request.form.get('fc'),
            four_g=request.form.get('four_g'),
            int_memory=request.form.get('int_memory'),
            m_dep=request.form.get('m_dep'),
            mobile_wt=request.form.get('mobile_wt'),
            n_cores=request.form.get('n_cores'),
            pc=request.form.get('pc'),
            px_height=request.form.get('px_height'),
            px_width=request.form.get('px_width'),
            ram=request.form.get('ram'),
            sc_h=request.form.get('sc_h'),
            sc_w=request.form.get('sc_w'),
            talk_time=request.form.get('talk_time'),
            three_g=request.form.get('three_g'),
            touch_screen=request.form.get('touch_screen'),
            wifi=request.form.get('wifi')

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)