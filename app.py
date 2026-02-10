import streamlit as st
import joblib
import pandas as pd

# load the trained model (gradient boosting tree) 
model = joblib.load('weather_model.pkl')

# app interface
st.set_page_config(page_title="SkyCast")
st.title("SkyCast: Rain Frequency Predictor")

# sidebar (user input)
st.sidebar.header("Observation Controls")
st.sidebar.markdown("Adjust 9am atmospheric data to see the prediction change.")

humidity = st.sidebar.slider("9am Relative Humidity (%)", 0, 100, 75)
humidity_3pm = st.sidebar.slider("3pm Relative Humidity (%)", 0, 100, 70)
wind_speed = st.sidebar.slider("9am Max Wind Speed (mph)", 0.0, 25.0, 12.0)

pressure = st.sidebar.number_input(
    "9am Air Pressure (hPa)", 
    min_value=908.0, 
    max_value=930.0, 
    value=918.0
)

# predict button
predict_btn = st.sidebar.button("Predict Weather")

# only run the prediction if the button is clicked
if predict_btn:
    # preduction
    input_df = pd.DataFrame([[
        pressure, 20.0, 180.0, wind_speed * 0.7, 180.0, wind_speed, humidity, humidity_3pm
    ]], columns=[
        'air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am', 
        'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am', 
        'relative_humidity_9am', 'relative_humidity_3pm'
    ])

    prediction = model.predict(input_df)

    # peak rain intensity
    hours = [f"{h:02d}:00" for h in range(24)]
    timeline_values = []

    # if it predicts rain, intensity peaks at 0.8, else it stays low at 0.1
    base_level = 0.8 if prediction[0] == 1 else 0.1

    for h in range(24):
        dist_from_peak = abs(h - 15)
        intensity = base_level * (1 - (dist_from_peak / 15))
        timeline_values.append(max(0, intensity)) # Ensure no negative numbers

    chart_data = pd.DataFrame({
        'Time of Day': hours,
        'Rain Intensity': timeline_values,
        'Scale_Max': [1.0] * 24  
    }).set_index('Time of Day')

    # display results
    if prediction[0] == 1:
        st.error("### ☔ Rain is Likely Today")
        st.write("Potential disruptions in outdoor activities.")
    else:
        st.success("### ☀️ Clear Skies Expected")
        st.write("Conditions are stable. Low probability of precipitation.")

    # line chart for rain intensity
    st.markdown("#### Predicted Rain Intensity Timeline")
    st.line_chart(chart_data[['Rain Intensity', 'Scale_Max']])

else:
    st.info("Please adjust the atmospheric controls and click 'Predict Weather' to generate a forecast.")

# tips section
st.divider()
st.subheader("Tips")
st.info("""
- **Humidity:** High humidity is the strongest indicator of rain.
- **Air Pressure:** Lower pressure often signals an approaching storm.
- **Wind Speed:** High wind gusts can increase the chance of rain.
""")
