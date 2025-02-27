from flask import Flask, render_template, request, redirect
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow as tf
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def data():
    try:
        # Get form data
        data = request.form.to_dict()
        print(data)
        
        # Save original data to CSV
        data_df = pd.DataFrame([data])
        data_df.to_csv('test.csv')
        
        # Create separate dataframes for right and left eyes
        rd = data_df[['headache','near_reading_problems','far_reading_problems','watering_eyes',
                     'dizziness','eye_strain','age','gender','color_vision_r','spokes_r',
                     'long_range_vision_r','short_range_vision_r']]
        
        ld = data_df[['headache','near_reading_problems','far_reading_problems','watering_eyes',
                     'dizziness','eye_strain','age','gender','color_vision_l','spokes_l',
                     'long_range_vision_l','short_range_vision_l']]
        
        # Export to Excel for later use
        rd.to_excel('right_eye.xlsx')
        ld.to_excel('left_eye.xlsx')
        
        # Process left eye data
        ld = ld.replace(to_replace=['No','Yes','no', 'yes','Female','Male','female','male'], 
                       value=[0,1,0,1,1,0,1,0])
        
        # Process right eye data
        rd = rd.replace(to_replace=['No','Yes','no', 'yes','Female','Male','female','male'], 
                       value=[0,1,0,1,1,0,1,0])
        
        # Create one-hot encoded columns for color vision - Left eye
        color_val_l = ld['color_vision_l'].iloc[0]
        ld['color_vision_l_r'] = 1 if color_val_l == 'Red' else 0
        ld['color_vision_l_g'] = 1 if color_val_l == 'Green' else 0
        ld['color_vision_l_b'] = 1 if color_val_l == 'Blue' else 0
        
        # Create one-hot encoded columns for color vision - Right eye
        color_val_r = rd['color_vision_r'].iloc[0]
        rd['color_vision_r_r'] = 1 if color_val_r == 'Red' else 0
        rd['color_vision_r_g'] = 1 if color_val_r == 'Green' else 0
        rd['color_vision_r_b'] = 1 if color_val_r == 'Blue' else 0
        
        # Convert string numbers to float for left eye
        numeric_cols = ['age', 'spokes_l', 'long_range_vision_l', 'short_range_vision_l']
        for col in numeric_cols:
            ld[col] = pd.to_numeric(ld[col], errors='coerce')
        
        # Convert string numbers to float for right eye
        numeric_cols = ['age', 'spokes_r', 'long_range_vision_r', 'short_range_vision_r']
        for col in numeric_cols:
            rd[col] = pd.to_numeric(rd[col], errors='coerce')
        
        # Handle any missing values
        ld = ld.fillna(0)
        rd = rd.fillna(0)
        
        # Save processed data
        ld.to_excel('FLD.xlsx')
        rd.to_excel('RLD.xlsx')
        
        # Import models with custom objects
        from tensorflow import keras
        
        custom_objects = {
            'mse': tf.keras.losses.MeanSquaredError(),
            'mean_squared_error': tf.keras.losses.MeanSquaredError()
        }
        
        # Define metrics for compatibility
        tf.keras.metrics.mse = tf.keras.metrics.MeanSquaredError()
        
        # Load models with custom objects
        model_axis = keras.models.load_model('axis.h5', custom_objects=custom_objects)
        model_sph = keras.models.load_model('model_spherical_r.h5', custom_objects=custom_objects)
        model_cyl = keras.models.load_model('model_cyl_r.h5', custom_objects=custom_objects)
        model_add = keras.models.load_model('model_addition_r.h5', custom_objects=custom_objects)
        
        # AXIS PREDICTION
        # RIGHT EYE
        x_spokes_right = float(rd['spokes_r'].iloc[0])
        x_spokes_right = np.array([x_spokes_right]).reshape(1, 1)  # Reshape to (1, 1)
        axis_right_pred = model_axis.predict(x_spokes_right)
        r_axis = str(int(axis_right_pred[0][0])) if axis_right_pred.ndim > 1 else str(int(axis_right_pred[0]))
        
        # LEFT EYE  
        x_spokes_left = float(ld['spokes_l'].iloc[0])
        x_spokes_left = np.array([x_spokes_left]).reshape(1, 1)  # Reshape to (1, 1)
        axis_left_pred = model_axis.predict(x_spokes_left)
        l_axis = str(int(axis_left_pred[0][0])) if axis_left_pred.ndim > 1 else str(int(axis_left_pred[0]))
        
        # SPHERICAL PREDICTION
        # RIGHT EYE
        r_sph_cols = ['color_vision_r_b', 'color_vision_r_g', 'color_vision_r_r', 
                      'headache', 'near_reading_problems', 'far_reading_problems', 
                      'watering_eyes', 'dizziness', 'long_range_vision_r']
        x_sph_right = rd[r_sph_cols].values
        x_sph_right = np.expand_dims(x_sph_right, axis=1)  # Add middle dimension for 3D input
        sph_right_pred = model_sph.predict(x_sph_right)
        r_sph = str(round(float(sph_right_pred[0][0]), 2)) if sph_right_pred.ndim > 1 else str(round(float(sph_right_pred[0]), 2))
        
        # LEFT EYE
        l_sph_cols = ['color_vision_l_b', 'color_vision_l_g', 'color_vision_l_r', 
                      'headache', 'near_reading_problems', 'far_reading_problems', 
                      'watering_eyes', 'dizziness', 'long_range_vision_l']
        x_sph_left = ld[l_sph_cols].values
        x_sph_left = np.expand_dims(x_sph_left, axis=1)  # Add middle dimension for 3D input
        sph_left_pred = model_sph.predict(x_sph_left)
        l_sph = str(round(float(sph_left_pred[0][0]), 2)) if sph_left_pred.ndim > 1 else str(round(float(sph_left_pred[0]), 2))
        
        # CYLINDRICAL PREDICTION
        # RIGHT EYE
        x_cyl_right = rd[r_sph_cols].values  # Using same columns as spherical
        x_cyl_right = np.expand_dims(x_cyl_right, axis=1)  # Add middle dimension for 3D input
        cyl_right_pred = model_cyl.predict(x_cyl_right)
        r_cyl = str(round(float(cyl_right_pred[0][0]), 2)) if cyl_right_pred.ndim > 1 else str(round(float(cyl_right_pred[0]), 2))
        
        # LEFT EYE
        x_cyl_left = ld[l_sph_cols].values  # Using same columns as spherical
        x_cyl_left = np.expand_dims(x_cyl_left, axis=1)  # Add middle dimension for 3D input
        cyl_left_pred = model_cyl.predict(x_cyl_left)
        l_cyl = str(round(float(cyl_left_pred[0][0]), 2)) if cyl_left_pred.ndim > 1 else str(round(float(cyl_left_pred[0]), 2))
        
        # ADDITION PREDICTION
        # RIGHT EYE
        r_add_cols = ['color_vision_r_b', 'color_vision_r_g', 'color_vision_r_r', 
                      'headache', 'near_reading_problems', 'far_reading_problems', 
                      'watering_eyes', 'dizziness', 'eye_strain', 'age', 
                      'gender', 'short_range_vision_r']
        x_add_right = rd[r_add_cols].values
        x_add_right = np.expand_dims(x_add_right, axis=1)  # Add middle dimension for 3D input
        add_right_pred = model_add.predict(x_add_right)
        r_add = str(round(float(add_right_pred[0][0]), 2)) if add_right_pred.ndim > 1 else str(round(float(add_right_pred[0]), 2))
        
        # LEFT EYE
        l_add_cols = ['color_vision_l_b', 'color_vision_l_g', 'color_vision_l_r', 
                      'headache', 'near_reading_problems', 'far_reading_problems', 
                      'watering_eyes', 'dizziness', 'eye_strain', 'age', 
                      'gender', 'short_range_vision_l']
        x_add_left = ld[l_add_cols].values
        x_add_left = np.expand_dims(x_add_left, axis=1)  # Add middle dimension for 3D input
        add_left_pred = model_add.predict(x_add_left)
        l_add = str(round(float(add_left_pred[0][0]), 2)) if add_left_pred.ndim > 1 else str(round(float(add_left_pred[0]), 2))
        
        # Print results for debugging
        print("====================================================Axis====================================================")
        print('Right Eye Axis: ' + r_axis)
        print("Left Eye Axis: " + l_axis)
        print("==================================================Spherical=================================================")
        print("Right Eye Spherical: " + r_sph)
        print("Left Eye Spherical: " + l_sph)
        print("=================================================Cylindrical================================================")
        print("Right Eye Cylindrical: " + r_cyl)
        print("Left Eye Cylindrical: " + l_cyl)
        print("===================================================Addition=================================================")
        print("Right Eye Addition: " + r_add)
        print("Left Eye Addition: " + l_add)
        
        # Create result dictionary
        res = {
            'Axis': [r_axis, l_axis],
            'Spherical': [r_sph, l_sph],
            'Cylindrical': [r_cyl, l_cyl],
            'Addition': [r_add, l_add]
        }
        
        # Return result
        return render_template("test.html", result=res)
    
    except Exception as e:
        print(f"Error in data route: {e}")
        import traceback
        traceback.print_exc()
        return render_template("test.html", result={
            'Axis': ["Error", "Error"],
            'Spherical': ["Error", "Error"],
            'Cylindrical': ["Error", "Error"],
            'Addition': ["Error", "Error"]
        })

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)