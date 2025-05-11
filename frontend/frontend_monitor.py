import sys
from pathlib import Path

# Dynamically set the Python path to include the project root directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.inference import fetch_hourly_rides, fetch_predictions
except ModuleNotFoundError:
    st.error("Failed to import 'src.inference'. Make sure the 'src' directory exists and is in the correct location.")
    sys.exit(1)

st.title("Mean Absolute Error (MAE) by Pickup Hour")

# Sidebar for user input
st.sidebar.header("Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Plot",
    min_value=12,  # Minimum allowed value
    max_value=24 * 28,  # (Optional) Maximum allowed value
    value=12,  # Initial/default value
    step=1,  # Step size for increment/decrement
)

# Fetch data with error handling
try:
    st.write("Fetching data for the past", past_hours, "hours...")
    df1 = fetch_hourly_rides(past_hours)
    df2 = fetch_predictions(past_hours)
except Exception as e:
    st.error(f"Error fetching data: {str(e)}")
    sys.exit(1)

# Debugging print statements (optional)
print("Fetched Hourly Rides Data:", df1.head())
print("Fetched Predictions Data:", df2.head())

# Check if DataFrames are empty
if df1.empty or df2.empty:
    st.warning("No data fetched for the selected period. Please try a different range.")
    sys.exit(1)

# Merge the DataFrames on 'pickup_location_id' and 'pickup_hour'
try:
    merged_df = pd.merge(df1, df2, on=["pickup_location_id", "pickup_hour"])
except KeyError:
    st.error("Data merging failed. Ensure both dataframes have 'pickup_location_id' and 'pickup_hour' columns.")
    sys.exit(1)

# Calculate the absolute error
merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])

# Group by 'pickup_hour' and calculate the mean absolute error (MAE)
mae_by_hour = merged_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Debugging print statements
print("MAE DataFrame:", mae_by_hour.head())

# Check if MAE DataFrame is empty
if mae_by_hour.empty:
    st.warning("No MAE data to display. Please try again with different settings.")
else:
    # Create a Plotly plot
    fig = px.line(
        mae_by_hour,
        x="pickup_hour",
        y="MAE",
        title=f"Mean Absolute Error (MAE) for the Past {past_hours} Hours",
        labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
        markers=True,
    )
    # Display the plot
    st.plotly_chart(fig)
    st.write(f'Average MAE: {mae_by_hour["MAE"].mean()}')
