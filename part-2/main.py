import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, asin, log2
import json
import pickle
import numpy as np
import util as ut
from dotenv import load_dotenv
from groq import Groq
import os
    
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

xgboost_model = load_model("../part-1/models/xgb_model.pkl")

dt_model = load_model("../part-1/models/dt_model.pkl")

random_forest_model = load_model("../part-1/models/rf_model.pkl")

def get_data():

    return pd.read_csv("../part-1/fraud.csv")

def load_encoding(path, key):

    with open(path, "r") as f:
        return json.load(f)[key]

def haversine_distance(lat1, lon1, lat2, lon2):

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def display_map(selected_transaction):

    # Create map centered between customer and merchant locations
    center_lat = (selected_transaction["lat"] + selected_transaction["merch_lat"]) / 2
    center_long = (selected_transaction["long"] + selected_transaction["merch_long"]) / 2

    # Calculate zoom level based on distance between points
    lat_diff = abs(selected_transaction["lat"] - selected_transaction["merch_lat"])
    long_diff = abs(selected_transaction["long"] - selected_transaction["merch_long"])
    max_diff = max(lat_diff, long_diff)
    zoom = int(log2(360/max_diff))  # Adjust zoom to fit points with padding

    m = folium.Map(location=[center_lat, center_long], zoom_start=zoom, dragging=False, 
                  scrollWheelZoom=False, touchZoom=False, doubleClickZoom=False, 
                  boxZoom=False, keyboard=False, zoomControl=False)

    # Add customer marker
    folium.Marker(
        [selected_transaction["lat"], selected_transaction["long"]],
        tooltip=f"{selected_transaction['first']} {selected_transaction['last']}", 
        icon=folium.Icon(color='blue', icon='user', prefix='fa')
    ).add_to(m)

    # Add merchant marker 
    folium.Marker(
        [selected_transaction["merch_lat"], selected_transaction["merch_long"]], 
        tooltip=selected_transaction["merchant"].replace("fraud_", ""),
        icon=folium.Icon(color='red', icon='shop', prefix='fa')
    ).add_to(m)

     # Add line between customer and merchant
    distance = haversine_distance(selected_transaction['lat'], selected_transaction['long'], 
                                selected_transaction['merch_lat'], selected_transaction['merch_long'])

    # Add line between customer and merchant
    folium.PolyLine(
        locations=[[selected_transaction["lat"], selected_transaction["long"]], 
                  [selected_transaction["merch_lat"], selected_transaction["merch_long"]]],
        tooltip=f"Distance: {distance:.2f} km",  # Changed popup to tooltip for permanent display
        color="orange",
        weight=2.5,
        opacity=0.8,
        dash_array='10'  # This makes the line dotted
    ).add_to(m)

    # Display map with full width and height, disable page reloading on map interaction
    st_folium(m, width=700, height=400, key="map", returned_objects=[])

    return distance

def prepare_input(amt, city_pop, gender, dob, job, state, trans_date, trans_time, distance, category):

    return {
        "amt": amt,
        "city_pop": city_pop,
        "gender_F": 1 if gender == "Female" else 0,
        "gender_M": 1 if gender == "Male" else 0,
        "age_at_transaction": (trans_date.year - dob.year) - ((trans_date.month < dob.month) | ((trans_date.month == dob.month) & (trans_date.day < dob.day))),
        "job_encoding": load_encoding("../part-1/encodings/job_encodings.json", job),
        "state_encoding": load_encoding("../part-1/encodings/state_encodings.json", state),
        "hour_of_day": trans_time.hour,
        "day_of_week": trans_date.weekday(),
        "havensine_distance": distance,
        "category_encoding": load_encoding("../part-1/encodings/category_encodings.json", category),
    }

def predict(input_dict):

    input_df = pd.DataFrame([input_dict])

    probabilities = {
        "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
        "Decision Tree": dt_model.predict_proba(input_df)[0][1],
        "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The transaction has a {avg_probability:.2%} probability of being fraudulent.")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)


    return avg_probability

def explain_prediction(probability, input_dict, customer):

    system_prompt = f"""You are an expert data scientist at a Stripe, where you specialize in interpreting and explaining predictions of machine learning models.
    """

    prompt = f"""Don't mention the probability of the transaction being fraudulent, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.

    If you use $ make sure to escape it by writing \$

    Your machine learning model has predicted that a transaction by {customer} has a {round(probability * 100, 1)}% probability of being fraudulent.

    If the transaction has less than a 30% risk of being fraudulent, generate a 3 sentence explanation of why it is at low risk of being fraudulent.

    If the transaction has between 30% and 60% risk of being fraudulent, generate a 3 sentence explanation of why it is at moderate risk of being fraudulent.

    If the transaction has over 60% risk of being fraudulent, generate a 3 sentence explanation of why it is at high risk of being fraudulent.
    
    Your explanation should be based on the transaction's information, the summary statistics of fraudulent and non-fraudulent transactions, and the feature importances provided.
 
    Respond based on the information provided below:

    Here is the customer's information:
    {input_dict}

    Here are the machine learning model's top 10 most important features for predicting fraud:

    Feature             | Importance
    --------------------------------
    hour_of_day         | 0.2831990718841553
    amt                 | 0.24604500830173492
    category_encoding   | 0.23827065527439117
    gender_F            | 0.08312086015939713
    age_at_transaction  | 0.0671813115477562
    job_encoding        | 0.027180925011634827
    city_pop            | 0.019962375983595848
    day_of_week         | 0.012735710479319096
    state_encoding      | 0.012727400287985802
    havensine_distance  | 0.009576713666319847

    Here are summary statistics for fraudulent transactions:
    {df[df["is_fraud"] == 1].describe()}

    Here are summary statistics for non-fraudulent transactions:
    {df[df["is_fraud"] == 0].describe()}

    Note the all encodings are target encoded with is_fraud as the target.
    """

    raw_response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile", # llama-3.1-70b-versatile
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, customer, trans_date, trans_time):

    system_prompt = f"""You are a manager at Stripe. You are responsible for ensuring customers are not defrauded.
    """

    prompt = f"""You noticed a customer named {customer} has a {round(probability * 100, 1)}% probability of being defrauded.

    Generate an email to the customer based on their information, to inform them that a transaction they made has been flagged as potentially fraudulent.

    Use Mr. or Ms. followed by the customer's surname to address the customer. Use Mr. if the customer is male and Ms. if the customer is female. Don't mention the resoning of your choice. 

    Here is the customer's information:
    {input_dict}

    Note that the transaction date and time are {trans_date} and {trans_time} respectively.

    Here is some explanation as to why the transaction might be fraudulent:
    {explanation}
    """

    raw_response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile", # llama-3.1-70b-versatile
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return raw_response.choices[0].message.content

st.title("Credit Card Fraud Detection")

df = get_data()

transactions = [f"{row['trans_num']} - {row['first']} {row['last']}" for _, row in df.iterrows()]

selected_transaction_option = st.selectbox("Select a transaction", transactions)

if selected_transaction_option:
    
    trans_num = selected_transaction_option.split(" - ")[0]
    customer = selected_transaction_option.split(" - ")[1]

    selected_transaction = df.loc[df["trans_num"] == trans_num].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:

    state = st.selectbox(
        "State",
        options=df["state"].unique(),
        index=list(df["state"].unique()).index(selected_transaction["state"])
    )

    city_pop = st.number_input(
        "City Population",
        min_value=0,
        step=1,
        value=int(selected_transaction["city_pop"])
    )

with col2:

    dob = st.date_input(
        "Date of Birth",
        value=pd.to_datetime(selected_transaction["dob"]).date()
    )

    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        index=0 if selected_transaction["gender"] == "Male" else 1
    )

    job = st.selectbox(
        "Job",
        options=df["job"].unique(),
        index=list(df["job"].unique()).index(selected_transaction["job"])
    )

with col3:
   
    amt = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        step=0.01,
        value=float(selected_transaction["amt"]),
        format="%.2f"
    )

    category = st.selectbox(
        "Category",
        options=df["category"].unique(),
        index=list(df["category"].unique()).index(selected_transaction["category"])
    )

    trans_date = st.date_input(
        "Transaction Date",
        value=pd.to_datetime(selected_transaction["trans_date_trans_time"]).date()
    )
    
    trans_time = st.time_input(
        "Transaction Time", 
        value=pd.to_datetime(selected_transaction["trans_date_trans_time"]).time(),
        step=60
    )

distance = display_map(selected_transaction)

input_dict = prepare_input(amt, city_pop, gender, dob, job, state, trans_date, trans_time, distance, category)

avg_probability = predict(input_dict)

explanation = explain_prediction(avg_probability, input_dict, customer)

st.markdown("---")

st.subheader("Explanation of Prediction")

st.markdown(explanation)

if avg_probability > 0.30:

    email = generate_email(avg_probability, input_dict, explanation, customer, trans_date, trans_time)

    st.markdown("---")

    st.subheader("Personalized Email")

    st.markdown(email)