# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:36:20 2024

@author: cavus
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:03:55 2024
@author: cavus
"""

import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap

# Load the Excel file
#file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Crime_data_2.xlsx'
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
excel_data = pd.ExcelFile(file_path)
data = excel_data.parse('Sheet1')

# Convert date columns to datetime
data['Date Rptd'] = pd.to_datetime(data['Date Rptd'], errors='coerce')
data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], errors='coerce')

# Drop columns with high missing values
data_cleaned = data.drop(columns=[
    'Mocodes', 'Vict Sex', 'Vict Descent',
    'Weapon Used Cd', 'Weapon Desc', 'Cross Street'
])

# Fill missing 'Premis Desc' values with 'Unknown'
data_cleaned['Premis Desc'].fillna('Unknown', inplace=True)

# Drop rows with missing values in essential columns
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd', 'LAT', 'LON'], inplace=True)

# Add columns for Year, Month, and Day of the Week
data_cleaned['Year'] = data_cleaned['DATE OCC'].dt.year
data_cleaned['Month'] = data_cleaned['DATE OCC'].dt.month
data_cleaned['DayOfWeek'] = data_cleaned['DATE OCC'].dt.day_name()

# Format crime descriptions to title case
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Analyze by Crime Type
crime_type_counts = data_cleaned['Crm Cd Desc'].value_counts().reset_index()
crime_type_counts.columns = ['Crime_Type', 'Count']

# Display top 10 crime types
top_10_crime_types = crime_type_counts.head(10)

# Shorten the labels for the y-axis as requested
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Battery - Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}
top_10_crime_types['Crime_Type'] = top_10_crime_types['Crime_Type'].replace(label_mapping)

# Use the 'tab10' color map for unique colors in the top 10 crime types
colors = plt.get_cmap('tab10').colors[:10]

# Visualize top 10 crime types with unique colors
plt.figure(figsize=(12,6))
plt.barh(top_10_crime_types['Crime_Type'], top_10_crime_types['Count'], color=colors)
plt.xlabel('Number of occurrences', fontsize=16)
plt.ylabel('Crime type', fontsize=16)
plt.title('Top 10 Crime Types', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gca().invert_yaxis()

# Save the figure at 600 DPI
save_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\top_10_crime_types.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')
print(f"Bar chart saved at 600 DPI to '{save_path}'")

# Display the plot
plt.show()

# Mapping Crime Occurrences: Heatmap
latitude = data_cleaned['LAT'].mean()
longitude = data_cleaned['LON'].mean()
crime_map = folium.Map(location=[latitude, longitude], zoom_start=10)
heat_data = data_cleaned[['LAT', 'LON']].dropna().values.tolist()
HeatMap(heat_data, radius=10).add_to(crime_map)

# Save the heatmap to the specified location
heatmap_save_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\crime_heatmap.html'
crime_map.save(heatmap_save_path)
print(f"Crime heatmap has been saved to '{heatmap_save_path}'")

# Mapping Crime Occurrences: Point Map with Synchronized Colors and Legend
point_map = folium.Map(location=[latitude, longitude], zoom_start=10)

# Add circle markers with colors based on crime type and smaller radius
for index, row in data_cleaned.iterrows():
    crime_type = row['Crm Cd Desc']
    color = mcolors.to_hex(colors[index % 10])  # Rotate colors if >10
    folium.CircleMarker(
        location=(row['LAT'], row['LON']),
        radius=1,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['Crm Cd Desc']} - {row['AREA NAME']}"
    ).add_to(point_map)

# Create a legend for the map
legend_html = """
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 250px; height: 300px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white; padding: 10px;">
<b>Crime Type Legend</b><br>
&emsp;<i>Vehicle - Stolen</i> &emsp; <span style="color:#d62728;">&#9679;</span><br>
&emsp;<i>Battery - Simple assault</i> &emsp; <span style="color:#1f77b4;">&#9679;</span><br>
&emsp;<i>Burglary from vehicle</i> &emsp; <span style="color:#2ca02c;">&#9679;</span><br>
&emsp;<i>Vandalism - Felony</i> &emsp; <span style="color:#ff7f0e;">&#9679;</span><br>
&emsp;<i>Burglary</i> &emsp; <span style="color:#8c564b;">&#9679;</span><br>
&emsp;<i>Aggravated assault</i> &emsp; <span style="color:#9467bd;">&#9679;</span><br>
&emsp;<i>Partner assault</i> &emsp; <span style="color:#e377c2;">&#9679;</span><br>
&emsp;<i>Theft plain</i> &emsp; <span style="color:#bcbd22;">&#9679;</span><br>
&emsp;<i>Theft from motor vehicle</i> &emsp; <span style="color:#7f7f7f;">&#9679;</span><br>
&emsp;<i>Theft of identity</i> &emsp; <span style="color:#17becf;">&#9679;</span><br>
</div>
"""
point_map.get_root().html.add_child(folium.Element(legend_html))

# Save the point map with the legend to the specified location
point_map_save_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\crime_point_map_colored.html'
point_map.save(point_map_save_path)
print(f"Point map with crime colors and legend has been saved to '{point_map_save_path}'")

# Secure Region Clustering and Visualization

# Step 1: Clustering to Identify Secure Regions
crime_locations = data_cleaned[['LAT', 'LON']].dropna().copy()

# Apply KMeans Clustering
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
crime_locations['Cluster'] = kmeans.fit_predict(crime_locations[['LAT', 'LON']])

# Calculate crime density per cluster and identify low-crime clusters
crime_density = crime_locations['Cluster'].value_counts().sort_index()
low_crime_clusters = crime_density[crime_density < crime_density.mean()].index

# Step 2: Visualize Secure Regions on Map with only two colors for low-crime and other clusters
secure_region_map = folium.Map(location=[latitude, longitude], zoom_start=10)

# Define colors for low-crime and other clusters
low_crime_color = "#2ca02c"  # Green color for low-crime clusters
other_cluster_color = "#1f77b4"  # Blue color for other clusters

for cluster_num in range(num_clusters):
    cluster_data = crime_locations[crime_locations['Cluster'] == cluster_num]
    color = low_crime_color if cluster_num in low_crime_clusters else other_cluster_color

    # Set opacity for low-crime and other clusters
    fill_opacity = 0.8 if cluster_num in low_crime_clusters else 0.3
    for _, row in cluster_data.iterrows():
        folium.CircleMarker(
            location=(row['LAT'], row['LON']),
            radius=3,
            color=color,
            fill=True,
            fill_opacity=fill_opacity
        ).add_to(secure_region_map)

# Update the legend to match the two-color scheme
legend_html = """
<div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 120px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color:white; padding: 10px;">
<b>Cluster Legend</b><br>
Low-Crime Clusters: <span style="color:#2ca02c;">&#9679;</span><br>
Other Clusters: <span style="color:#1f77b4;">&#9679;</span><br>
</div>
"""
secure_region_map.get_root().html.add_child(folium.Element(legend_html))

# Save the secure region map
secure_map_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\secure_region_map.html'
secure_region_map.save(secure_map_path)
print(f"Secure region map has been saved to '{secure_map_path}'")

# Random Forest Model for Crime Type Prediction
data_cleaned['Crm_Cd_Encoded'] = data_cleaned['Crm Cd Desc'].astype('category').cat.codes
X = data_cleaned[['Month', 'LAT', 'LON', 'Premis Cd']]
y = data_cleaned['Crm_Cd_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("ELOP Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# SHAP for Model Interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[1], X_test, plot_type="bar", feature_names=X_test.columns)

# SHAP Dependence Plot
plt.figure(figsize=(8, 6))
shap.dependence_plot('LAT', shap_values[1], X_test)


import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the data from the specified file path
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')

# Drop columns with high missing values
data_cleaned = crime_data.drop(columns=[
    'Mocodes', 'Vict Sex', 'Vict Descent',
    'Weapon Used Cd', 'Weapon Desc', 'Cross Street'
])

# Fill missing 'Premis Desc' values with 'Unknown'
data_cleaned['Premis Desc'].fillna('Unknown', inplace=True)

# Drop rows with missing values in essential columns
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd', 'LAT', 'LON'], inplace=True)

# Select zones for analysis
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
crime_data_filtered = data_cleaned[data_cleaned['AREA NAME'].isin(selected_zones)]

# Initialize map centered on the mean latitude and longitude of the filtered data
map_center = [crime_data_filtered['LAT'].mean(), crime_data_filtered['LON'].mean()]
crime_map = folium.Map(location=map_center, zoom_start=10)

# Prepare data for heatmap (latitude and longitude points)
heat_data = [[row['LAT'], row['LON']] for index, row in crime_data_filtered.iterrows()]

# Add HeatMap layer to the map
HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(crime_map)

# Save the map as an HTML file
crime_map.save("crime_density_heatmap.html")
crime_map





# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:36:20 2024

@author: cavus
"""

import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import shap

# Load the Excel file
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
excel_data = pd.ExcelFile(file_path)
data = excel_data.parse('Sheet1')

# Convert date columns to datetime
data['Date Rptd'] = pd.to_datetime(data['Date Rptd'], errors='coerce')
data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], errors='coerce')

# Drop columns with high missing values
data_cleaned = data.drop(columns=[
    'Mocodes', 'Vict Sex', 'Vict Descent',
    'Weapon Used Cd', 'Weapon Desc', 'Cross Street'
])

# Fill missing 'Premis Desc' values with 'Unknown'
data_cleaned['Premis Desc'].fillna('Unknown', inplace=True)

# Drop rows with missing values in essential columns
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd', 'LAT', 'LON'], inplace=True)

# Add columns for Year, Month, and Day of the Week
data_cleaned['Year'] = data_cleaned['DATE OCC'].dt.year
data_cleaned['Month'] = data_cleaned['DATE OCC'].dt.month
data_cleaned['DayOfWeek'] = data_cleaned['DATE OCC'].dt.day_name()

# Format crime descriptions to title case
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Analyze by Crime Type
crime_type_counts = data_cleaned['Crm Cd Desc'].value_counts().reset_index()
crime_type_counts.columns = ['Crime_Type', 'Count']

# Display top 10 crime types
top_10_crime_types = crime_type_counts.head(10)

# Shorten the labels for the y-axis as requested
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Battery - Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}
top_10_crime_types['Crime_Type'] = top_10_crime_types['Crime_Type'].replace(label_mapping)

# Use the 'tab10' color map for unique colors in the top 10 crime types
colors = plt.get_cmap('tab10').colors[:10]

# Visualize top 10 crime types with unique colors
plt.figure(figsize=(12, 6))
plt.barh(top_10_crime_types['Crime_Type'], top_10_crime_types['Count'], color=colors)
plt.xlabel('Number of occurrences', fontsize=16)
plt.ylabel('Crime type', fontsize=16)
plt.title('Top 10 Crime Types', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add gridlines for the x-axis
plt.gca().invert_yaxis()

# Save the figure at 600 DPI
save_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\top_10_crime_types.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')
print(f"Bar chart saved at 600 DPI to '{save_path}'")

# Display the plot
plt.show()

# Random Forest Model for Crime Type Prediction
data_cleaned['Crm_Cd_Encoded'] = data_cleaned['Crm Cd Desc'].astype('category').cat.codes
X = data_cleaned[['Month', 'LAT', 'LON', 'Premis Cd']]
y = data_cleaned['Crm_Cd_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("ELOP Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# SHAP for Model Interpretability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot with 600 DPI and proper y-axis label
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values[1], 
    X_test, 
    plot_type="bar", 
    feature_names=['Month', 'LAT', 'LON', 'Premis Cd']
)
plt.xlabel('Mean SHAP Value', fontsize=16)
plt.ylabel('Features (Month, LAT, LON, Premis Cd)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add gridlines for better readability

# Save the SHAP summary plot at 600 DPI
shap_summary_save_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\shap_summary_plot_labeled.png'
plt.savefig(shap_summary_save_path, dpi=600, bbox_inches='tight')
print(f"SHAP summary plot saved at 600 DPI to '{shap_summary_save_path}'")

# Display the plot
plt.show()

# SHAP Dependence Plot with 600 DPI and y-axis label
plt.figure(figsize=(8, 6))
shap.dependence_plot('LAT', shap_values[1], X_test)
plt.xlabel('Latitude (LAT)', fontsize=16)
plt.ylabel('SHAP Value for LAT', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='both', linestyle='--', alpha=0.7)  # Gridlines for both axes

# Save the SHAP dependence plot at 600 DPI
shap_dependence_save_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\shap_dependence_plot_labeled.png'
plt.savefig(shap_dependence_save_path, dpi=600, bbox_inches='tight')
print(f"SHAP dependence plot saved at 600 DPI to '{shap_dependence_save_path}'")

# Display the plot
plt.show()



import os
import matplotlib.pyplot as plt

# Define the save path
save_directory = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures'
os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
save_path = os.path.join(save_directory, 'SHAP_Feature_Importance_600dpi.png')

# Create the plot
features = ["Premis Cd", "LAT", "LON", "Month"]
shap_values = [0.00200, 0.00175, 0.00150, 0.00100]

# Adjust figure size to increase length
plt.figure(figsize=(10, 6))  # Width increased to 10 inches
plt.barh(features, shap_values, color="blue")
plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.title("SHAP Feature Importance", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add grid to the chart
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()

# Save the figure
plt.savefig(save_path, dpi=600)
plt.show()

print(f"Figure saved at: {save_path}")




import os
import matplotlib.pyplot as plt

# Define the save path
save_directory = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures'
os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
save_path = os.path.join(save_directory, 'SHAP_Feature_Importance_600dpi_NoAnnotations.png')

# Create the plot
features = ["Premis Cd", "LAT", "LON", "Month"]
shap_values = [0.00200, 0.00175, 0.00150, 0.00100]

# Adjust figure size for better visualization
plt.figure(figsize=(12, 7))  # Increased size for better readability

# Create a bar chart with color gradient
colors = plt.cm.viridis([0.2, 0.4, 0.6, 0.8])
plt.barh(features, shap_values, color=colors)

# Labels and Title
plt.xlabel("Mean(|SHAP value|)\n(Average impact on model output magnitude)", fontsize=24)
plt.ylabel("Features", fontsize=24)
#plt.title("SHAP Feature Importance\nFeature Contribution to Model Predictions", fontsize=18, pad=15)

# Customize ticks and grid
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Tight layout for better fit
plt.tight_layout()

# Save the figure
plt.savefig(save_path, dpi=600)
plt.show()

print(f"SHAP feature importance figure saved at: {save_path}")

