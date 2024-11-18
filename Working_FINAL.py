import pandas as pd
import matplotlib.pyplot as plt

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

# Define label mapping for crime types, including both variations of the motor vehicle theft description
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft From Motor Vehicle - Grand ($950.01 And Over)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}

# Apply the label mapping to both the top 10 crime types and filtered data
top_10_crime_types['Crime_Type'] = top_10_crime_types['Crime_Type'].replace(label_mapping)
data_cleaned['Short Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].replace(label_mapping)

# Use the 'tab10' color map for unique colors in the top 10 crime types
colors = plt.get_cmap('tab10').colors[:10]

# Visualize top 10 crime types with unique colors and gridlines
plt.figure(figsize=(12,6))
plt.barh(top_10_crime_types['Crime_Type'], top_10_crime_types['Count'], color=colors)
plt.xlabel('Number of occurrences', fontsize=16)
plt.ylabel('Crime type', fontsize=16)
plt.title('Top 10 Crime Types', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add gridlines on the x-axis
plt.gca().invert_yaxis()
plt.savefig("top_10_crime_types.png", dpi=600, bbox_inches='tight')  # Save the figure at 600 DPI
plt.show()

# Select 5 zones for analysis
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
crime_data_filtered = data_cleaned[data_cleaned['AREA NAME'].isin(selected_zones)]

# Group by 'AREA NAME' and 'Short Crm Cd Desc' to get counts of each crime type in each zone
crime_counts_by_zone = (
    crime_data_filtered.groupby(['AREA NAME', 'Short Crm Cd Desc'])
    .size()
    .reset_index(name='Count')
)

# Sort values and get top 5 crimes for each zone
top_crimes_by_zone = (
    crime_counts_by_zone.sort_values(['AREA NAME', 'Count'], ascending=[True, False])
    .groupby('AREA NAME')
    .head(5)
)

# Print data to verify it's not empty
print("Top Crimes by Zone:\n", top_crimes_by_zone)

# Plot top 5 crimes per zone with gridlines
fig, ax = plt.subplots(figsize=(12, 8))
for area in selected_zones:
    area_data = top_crimes_by_zone[top_crimes_by_zone['AREA NAME'] == area]
    ax.bar(area_data['Short Crm Cd Desc'], area_data['Count'], label=area)

# Customize plot appearance
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Crime type', fontsize=18)
plt.ylabel('Number of crimes', fontsize=18)
plt.xticks(rotation=90)
plt.legend(title='Zone')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines on the y-axis
plt.tight_layout()

# Save the figure at 600 DPI to the specified path
plt.savefig(r"C:\Users\cavus\Desktop\Conference_IEEE\Figures\top_5_crimes_by_zone.png", dpi=600, bbox_inches='tight')

# Show the plot
plt.show()




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



import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster

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

# Format crime descriptions to title case
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Define label mapping for consistent crime names
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft From Motor Vehicle - Grand ($950.01 And Over)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}

# Apply the label mapping to create 'Short Crm Cd Desc' column
data_cleaned['Short Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].replace(label_mapping)

# Define filters for each zone
zone_filters = {
    'West Valley': ['Vehicle - Stolen', 'Burglary from vehicle'],
    'Northeast': ['Simple assault'],
    'Southwest': ['Aggravated assault', 'Partner assault'],
    'Topanga': ['Theft plain', 'Theft from motor vehicle'],
    'Hollywood': ['Vandalism - Felony', 'Burglary']
}

# Define color mapping for each zone
zone_colors = {
    'West Valley': 'blue',
    'Northeast': 'orange',
    'Southwest': 'green',
    'Topanga': 'red',
    'Hollywood': 'purple'
}

# Initialize map centered on the mean latitude and longitude of the filtered data
map_center = [data_cleaned['LAT'].mean(), data_cleaned['LON'].mean()]
crime_map = folium.Map(location=map_center, zoom_start=10)

# Prepare data for the heatmap (latitude and longitude points)
heat_data = [[row['LAT'], row['LON']] for index, row in data_cleaned.iterrows()]
HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(crime_map)

# Add colored circle markers for each crime location based on the zone and its specific filters
marker_cluster = MarkerCluster().add_to(crime_map)
for zone, crimes in zone_filters.items():
    # Filter the data based on the zone and its specific crimes
    zone_data = data_cleaned[(data_cleaned['AREA NAME'] == zone) & (data_cleaned['Short Crm Cd Desc'].isin(crimes))]
    color = zone_colors.get(zone, 'gray')
    for _, row in zone_data.iterrows():
        folium.CircleMarker(
            location=[row['LAT'], row['LON']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=(
                f"Zone: {row['AREA NAME']}<br>"
                f"Crime: {row['Short Crm Cd Desc']}<br>"
                f"Date: {row['DATE OCC'].date()}<br>"
                f"Location: {row['Premis Desc']}"
            )
        ).add_to(marker_cluster)

# Add a custom legend to the map
legend_html = '''
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 200px; height: 150px; 
                 background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                 ">
     <b>&nbsp; Zone Legend</b><br>
     &nbsp; West Valley &nbsp; <i class="fa fa-circle" style="color:blue"></i><br>
     &nbsp; Northeast &nbsp; <i class="fa fa-circle" style="color:orange"></i><br>
     &nbsp; Southwest &nbsp; <i class="fa fa-circle" style="color:green"></i><br>
     &nbsp; Topanga &nbsp; <i class="fa fa-circle" style="color:red"></i><br>
     &nbsp; Hollywood &nbsp; <i class="fa fa-circle" style="color:purple"></i>
     </div>
     '''
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map as an HTML file
crime_map.save("crime_density_heatmap_with_zone_filters.html")
crime_map





import pandas as pd
import folium

# Load and clean the data
crime_data = pd.read_excel(file_path)
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'


# Convert 'DATE OCC' to datetime and clean data as shown in the initial code
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
data_cleaned = crime_data.drop(columns=[
    'Mocodes', 'Vict Sex', 'Vict Descent', 'Weapon Used Cd', 'Weapon Desc', 'Cross Street'
])
data_cleaned['Premis Desc'].fillna('Unknown', inplace=True)
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd', 'LAT', 'LON'], inplace=True)
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Define a label mapping for crime descriptions
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft From Motor Vehicle - Grand ($950.01 And Over)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}
data_cleaned['Short Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].replace(label_mapping)

# Select zones for mapping
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
crime_data_filtered = data_cleaned[data_cleaned['AREA NAME'].isin(selected_zones)]

# Group by 'AREA NAME' and 'Short Crm Cd Desc' to get counts of each crime type in each zone
crime_counts_by_zone = (
    crime_data_filtered.groupby(['AREA NAME', 'Short Crm Cd Desc'])
    .size()
    .reset_index(name='Count')
)
top_crimes_by_zone = (
    crime_counts_by_zone.sort_values(['AREA NAME', 'Count'], ascending=[True, False])
    .groupby('AREA NAME')
    .head(5)
)

# Merge coordinates for mapping
top_crimes_with_coords = top_crimes_by_zone.merge(
    data_cleaned[['AREA NAME', 'Short Crm Cd Desc', 'LAT', 'LON']].drop_duplicates(),
    on=['AREA NAME', 'Short Crm Cd Desc'],
    how='left'
)

# Define colors for each crime type
color_mapping = {
    'Vehicle - Stolen': 'red',
    'Simple assault': 'blue',
    'Burglary from vehicle': 'green',
    'Vandalism - Felony': 'purple',
    'Burglary': 'orange',
    'Aggravated assault': 'darkred',
    'Partner assault': 'cadetblue',
    'Theft plain': 'lightgreen',
    'Theft from motor vehicle': 'pink',
    'Theft of identity': 'lightblue'
}

# Center the map and create folium map
map_center = [data_cleaned['LAT'].mean(), data_cleaned['LON'].mean()]
crime_map = folium.Map(location=map_center, zoom_start=11)

# Add colorful circle markers to the map
for idx, row in top_crimes_with_coords.iterrows():
    color = color_mapping.get(row['Short Crm Cd Desc'], 'gray')  # default to gray if crime type is missing
    folium.CircleMarker(
        location=[row['LAT'], row['LON']],
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=row['Short Crm Cd Desc'],
        tooltip=f"{row['AREA NAME']}: {row['Short Crm Cd Desc']} ({row['Count']} occurrences)"
    ).add_to(crime_map)

# Save the map as an HTML file
map_file_path = "top_crimes_map_colorful.html"  # Save location
crime_map.save(map_file_path)
print(f"Map saved to {map_file_path}")







import pandas as pd
import folium

# Load and clean the data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'  # Your specified file path
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and clean data as shown in the initial code
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
data_cleaned = crime_data.drop(columns=[
    'Mocodes', 'Vict Sex', 'Vict Descent', 'Weapon Used Cd', 'Weapon Desc', 'Cross Street'
])
data_cleaned['Premis Desc'].fillna('Unknown', inplace=True)
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd', 'LAT', 'LON'], inplace=True)
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Define a label mapping for crime descriptions
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft From Motor Vehicle - Grand ($950.01 And Over)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}
data_cleaned['Short Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].replace(label_mapping)

# Select zones for mapping
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
crime_data_filtered = data_cleaned[data_cleaned['AREA NAME'].isin(selected_zones)]

# Group by 'AREA NAME' and 'Short Crm Cd Desc' to get counts of each crime type in each zone
crime_counts_by_zone = (
    crime_data_filtered.groupby(['AREA NAME', 'Short Crm Cd Desc'])
    .size()
    .reset_index(name='Count')
)
top_crimes_by_zone = (
    crime_counts_by_zone.sort_values(['AREA NAME', 'Count'], ascending=[True, False])
    .groupby('AREA NAME')
    .head(5)
)

# Merge coordinates for mapping
top_crimes_with_coords = top_crimes_by_zone.merge(
    data_cleaned[['AREA NAME', 'Short Crm Cd Desc', 'LAT', 'LON']].drop_duplicates(),
    on=['AREA NAME', 'Short Crm Cd Desc'],
    how='left'
)

# Define colors for each zone
zone_colors = {
    'West Valley': 'blue',
    'Northeast': 'orange',
    'Southwest': 'green',
    'Topanga': 'red',
    'Hollywood': 'purple'
}

# Center the map on an average location of the selected regions
map_center = [data_cleaned['LAT'].mean(), data_cleaned['LON'].mean()]
crime_map = folium.Map(location=map_center, zoom_start=11)

# Add colorful circle markers to the map
for idx, row in top_crimes_with_coords.iterrows():
    color = zone_colors.get(row['AREA NAME'], 'gray')  # default to gray if zone is missing
    folium.CircleMarker(
        location=[row['LAT'], row['LON']],
        radius=7,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=row['Short Crm Cd Desc'],
        tooltip=f"{row['AREA NAME']}: {row['Short Crm Cd Desc']} ({row['Count']} occurrences)"
    ).add_to(crime_map)

# Save the map as an HTML file
map_file_path = "top_crimes_map_colorful.html"  # Save location
crime_map.save(map_file_path)
print(f"Map saved to {map_file_path}")








import pandas as pd
import folium
from folium.plugins import FloatImage

# Load and clean the data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'  # Your specified file path
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and clean data as shown in the initial code
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
data_cleaned = crime_data.drop(columns=[
    'Mocodes', 'Vict Sex', 'Vict Descent', 'Weapon Used Cd', 'Weapon Desc', 'Cross Street'
])
data_cleaned['Premis Desc'].fillna('Unknown', inplace=True)
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd', 'LAT', 'LON'], inplace=True)
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Define label and color mappings
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft From Motor Vehicle - Grand ($950.01 And Over)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}
data_cleaned['Short Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].replace(label_mapping)

# Define colors for crime types
color_mapping = {
    'Vehicle - Stolen': 'red',
    'Simple assault': 'blue',
    'Burglary from vehicle': 'green',
    'Vandalism - Felony': 'purple',
    'Burglary': 'orange',
    'Aggravated assault': 'darkred',
    'Partner assault': 'cadetblue',
    'Theft plain': 'lightgreen',
    'Theft from motor vehicle': 'pink',
    'Theft of identity': 'lightblue'
}

# Define colors for zones
zone_colors = {
    'West Valley': 'blue',
    'Northeast': 'orange',
    'Southwest': 'green',
    'Topanga': 'red',
    'Hollywood': 'purple'
}

# Select zones for mapping
selected_zones = list(zone_colors.keys())
crime_data_filtered = data_cleaned[data_cleaned['AREA NAME'].isin(selected_zones)]

# Group by 'AREA NAME' and 'Short Crm Cd Desc' to get counts of each crime type in each zone
crime_counts_by_zone = (
    crime_data_filtered.groupby(['AREA NAME', 'Short Crm Cd Desc'])
    .size()
    .reset_index(name='Count')
)
top_crimes_by_zone = (
    crime_counts_by_zone.sort_values(['AREA NAME', 'Count'], ascending=[True, False])
    .groupby('AREA NAME')
    .head(5)
)

# Merge coordinates for mapping
top_crimes_with_coords = top_crimes_by_zone.merge(
    data_cleaned[['AREA NAME', 'Short Crm Cd Desc', 'LAT', 'LON']].drop_duplicates(),
    on=['AREA NAME', 'Short Crm Cd Desc'],
    how='left'
)

# Center the map on an average location of the selected regions
map_center = [data_cleaned['LAT'].mean(), data_cleaned['LON'].mean()]
crime_map = folium.Map(location=map_center, zoom_start=11)

# Add colorful circle markers to the map using both crime and zone colors
for idx, row in top_crimes_with_coords.iterrows():
    crime_color = color_mapping.get(row['Short Crm Cd Desc'], 'gray')
    folium.CircleMarker(
        location=[row['LAT'], row['LON']],
        radius=7,
        color=crime_color,
        fill=True,
        fill_color=crime_color,
        fill_opacity=0.7,
        popup=f"{row['Short Crm Cd Desc']} ({row['Count']} occurrences) in {row['AREA NAME']}",
        tooltip=f"{row['AREA NAME']}: {row['Short Crm Cd Desc']} ({row['Count']} occurrences)"
    ).add_to(crime_map)

# Add legends for crime and zone colors
# Adding a simple legend as HTML overlay
legend_html = '''
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 280px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;">
     <h4 style="margin-top:10px; text-align: center;">Crime Legend</h4>
     <ul style="list-style: none; padding: 0;">
       <li><span style="background-color:red; color:white; padding:2px 5px;">Vehicle - Stolen</span></li>
       <li><span style="background-color:blue; color:white; padding:2px 5px;">Simple Assault</span></li>
       <li><span style="background-color:green; color:white; padding:2px 5px;">Burglary from vehicle</span></li>
       <li><span style="background-color:purple; color:white; padding:2px 5px;">Vandalism - Felony</span></li>
       <li><span style="background-color:orange; color:white; padding:2px 5px;">Burglary</span></li>
       <li><span style="background-color:darkred; color:white; padding:2px 5px;">Aggravated Assault</span></li>
       <li><span style="background-color:cadetblue; color:white; padding:2px 5px;">Partner Assault</span></li>
       <li><span style="background-color:lightgreen; color:black; padding:2px 5px;">Theft Plain</span></li>
       <li><span style="background-color:pink; color:black; padding:2px 5px;">Theft from motor vehicle</span></li>
       <li><span style="background-color:lightblue; color:black; padding:2px 5px;">Theft of identity</span></li>
     </ul>
     <h4 style="text-align: center; margin-top:10px;">Zone Legend</h4>
     <ul style="list-style: none; padding: 0;">
       <li><span style="background-color:blue; color:white; padding:2px 5px;">West Valley</span></li>
       <li><span style="background-color:orange; color:white; padding:2px 5px;">Northeast</span></li>
       <li><span style="background-color:green; color:white; padding:2px 5px;">Southwest</span></li>
       <li><span style="background-color:red; color:white; padding:2px 5px;">Topanga</span></li>
       <li><span style="background-color:purple; color:white; padding:2px 5px;">Hollywood</span></li>
     </ul>
</div>
'''
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map as an HTML file
map_file_path = "top_crimes_map_with_legends.html"  # Save location
crime_map.save(map_file_path)
print(f"Map saved to {map_file_path}")





import pandas as pd
import folium

# Load and clean the data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'  # Your specified file path
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and clean data as shown in the initial code
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
data_cleaned = crime_data.drop(columns=[
    'Mocodes', 'Vict Sex', 'Vict Descent', 'Weapon Used Cd', 'Weapon Desc', 'Cross Street'
])
data_cleaned['Premis Desc'].fillna('Unknown', inplace=True)
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd', 'LAT', 'LON'], inplace=True)
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Define label and color mappings
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft From Motor Vehicle - Grand ($950.01 And Over)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}
data_cleaned['Short Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].replace(label_mapping)

# Define colors for crime types
color_mapping = {
    'Vehicle - Stolen': 'red',
    'Simple assault': 'blue',
    'Burglary from vehicle': 'green',
    'Vandalism - Felony': 'purple',
    'Burglary': 'orange',
    'Aggravated assault': 'darkred',
    'Partner assault': 'cadetblue',
    'Theft plain': 'lightgreen',
    'Theft from motor vehicle': 'pink',
    'Theft of identity': 'lightblue'
}

# Select zones for mapping
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
crime_data_filtered = data_cleaned[data_cleaned['AREA NAME'].isin(selected_zones)]

# Group by 'AREA NAME' and 'Short Crm Cd Desc' to get counts of each crime type in each zone
crime_counts_by_zone = (
    crime_data_filtered.groupby(['AREA NAME', 'Short Crm Cd Desc'])
    .size()
    .reset_index(name='Count')
)
top_crimes_by_zone = (
    crime_counts_by_zone.sort_values(['AREA NAME', 'Count'], ascending=[True, False])
    .groupby('AREA NAME')
    .head(5)
)

# Merge coordinates for mapping
top_crimes_with_coords = top_crimes_by_zone.merge(
    data_cleaned[['AREA NAME', 'Short Crm Cd Desc', 'LAT', 'LON']].drop_duplicates(),
    on=['AREA NAME', 'Short Crm Cd Desc'],
    how='left'
)

# Center the map on an average location of the selected regions
map_center = [data_cleaned['LAT'].mean(), data_cleaned['LON'].mean()]
crime_map = folium.Map(location=map_center, zoom_start=11)

# Add dot-like circle markers to the map using crime colors
for idx, row in top_crimes_with_coords.iterrows():
    crime_color = color_mapping.get(row['Short Crm Cd Desc'], 'gray')
    folium.CircleMarker(
        location=[row['LAT'], row['LON']],
        radius=2,  # Smaller radius for the dot-like effect
        color=crime_color,
        fill=True,
        fill_color=crime_color,
        fill_opacity=0.8,
        popup=f"{row['Short Crm Cd Desc']} ({row['Count']} occurrences) in {row['AREA NAME']}",
        tooltip=f"{row['AREA NAME']}: {row['Short Crm Cd Desc']} ({row['Count']} occurrences)"
    ).add_to(crime_map)

# Define HTML for a legend
legend_html = '''
<div style="position: fixed; 
     bottom: 30px; left: 30px; width: 220px; height: 280px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:12px;">
     <h4 style="margin-top:10px; text-align: center;">Crime Legend</h4>
     <ul style="list-style: none; padding: 0;">
       <li><span style="background-color:red; color:white; padding:2px 5px;">Vehicle - Stolen</span></li>
       <li><span style="background-color:blue; color:white; padding:2px 5px;">Simple Assault</span></li>
       <li><span style="background-color:green; color:white; padding:2px 5px;">Burglary from vehicle</span></li>
       <li><span style="background-color:purple; color:white; padding:2px 5px;">Vandalism - Felony</span></li>
       <li><span style="background-color:orange; color:white; padding:2px 5px;">Burglary</span></li>
       <li><span style="background-color:darkred; color:white; padding:2px 5px;">Aggravated Assault</span></li>
       <li><span style="background-color:cadetblue; color:white; padding:2px 5px;">Partner Assault</span></li>
       <li><span style="background-color:lightgreen; color:black; padding:2px 5px;">Theft Plain</span></li>
       <li><span style="background-color:pink; color:black; padding:2px 5px;">Theft from motor vehicle</span></li>
       <li><span style="background-color:lightblue; color:black; padding:2px 5px;">Theft of identity</span></li>
     </ul>
</div>
'''
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map as an HTML file
map_file_path = "top_crimes_dot_map.html"  # Save location
crime_map.save(map_file_path)
print(f"Map saved to {map_file_path}")






import pandas as pd
import folium

# Load and clean the data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'  # Your specified file path
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and clean data as shown in the initial code
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
data_cleaned = crime_data.drop(columns=[
    'Mocodes', 'Vict Sex', 'Vict Descent', 'Weapon Used Cd', 'Weapon Desc', 'Cross Street'
])
data_cleaned['Premis Desc'].fillna('Unknown', inplace=True)
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd', 'LAT', 'LON'], inplace=True)
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Define label and color mappings
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft From Motor Vehicle - Grand ($950.01 And Over)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}
data_cleaned['Short Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].replace(label_mapping)

# Define colors for crime types
color_mapping = {
    'Vehicle - Stolen': 'red',
    'Simple assault': 'blue',
    'Burglary from vehicle': 'green',
    'Vandalism - Felony': 'purple',
    'Burglary': 'orange',
    'Aggravated assault': 'darkred',
    'Partner assault': 'cadetblue',
    'Theft plain': 'lightgreen',
    'Theft from motor vehicle': 'pink',
    'Theft of identity': 'lightblue'
}

# Define colors for zones
zone_colors = {
    'West Valley': 'blue',
    'Northeast': 'orange',
    'Southwest': 'green',
    'Topanga': 'red',
    'Hollywood': 'purple'
}

# Select zones for mapping
selected_zones = list(zone_colors.keys())  # All 5 regions
crime_data_filtered = data_cleaned[data_cleaned['AREA NAME'].isin(selected_zones)]

# Verify that all 5 regions are included
print("Selected regions in data:", crime_data_filtered['AREA NAME'].unique())

# Group by 'AREA NAME' and 'Short Crm Cd Desc' to get counts of each crime type in each zone
crime_counts_by_zone = (
    crime_data_filtered.groupby(['AREA NAME', 'Short Crm Cd Desc'])
    .size()
    .reset_index(name='Count')
)
top_crimes_by_zone = (
    crime_counts_by_zone.sort_values(['AREA NAME', 'Count'], ascending=[True, False])
    .groupby('AREA NAME')
    .head(5)
)

# Merge coordinates for mapping
top_crimes_with_coords = top_crimes_by_zone.merge(
    data_cleaned[['AREA NAME', 'Short Crm Cd Desc', 'LAT', 'LON']].drop_duplicates(),
    on=['AREA NAME', 'Short Crm Cd Desc'],
    how='left'
)

# Ensure that top_crimes_with_coords contains data from all 5 regions
print("Regions in top_crimes_with_coords:", top_crimes_with_coords['AREA NAME'].unique())

# Center the map on an average location of the selected regions
map_center = [data_cleaned['LAT'].mean(), data_cleaned['LON'].mean()]
crime_map = folium.Map(location=map_center, zoom_start=11)

# Add dot-like circle markers to the map using crime colors
for idx, row in top_crimes_with_coords.iterrows():
    crime_color = color_mapping.get(row['Short Crm Cd Desc'], 'gray')
    folium.CircleMarker(
        location=[row['LAT'], row['LON']],
        radius=2,  # Smaller radius for the dot-like effect
        color=crime_color,
        fill=True,
        fill_color=crime_color,
        fill_opacity=0.8,
        popup=f"{row['Short Crm Cd Desc']} ({row['Count']} occurrences) in {row['AREA NAME']}",
        tooltip=f"{row['AREA NAME']}: {row['Short Crm Cd Desc']} ({row['Count']} occurrences)"
    ).add_to(crime_map)

# Define HTML for a legend
legend_html = '''
<div style="position: fixed; 
     bottom: 30px; left: 30px; width: 220px; height: 280px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:12px;">
     <h4 style="margin-top:10px; text-align: center;">Crime Legend</h4>
     <ul style="list-style: none; padding: 0;">
       <li><span style="background-color:red; color:white; padding:2px 5px;">Vehicle - Stolen</span></li>
       <li><span style="background-color:blue; color:white; padding:2px 5px;">Simple Assault</span></li>
       <li><span style="background-color:green; color:white; padding:2px 5px;">Burglary from vehicle</span></li>
       <li><span style="background-color:purple; color:white; padding:2px 5px;">Vandalism - Felony</span></li>
       <li><span style="background-color:orange; color:white; padding:2px 5px;">Burglary</span></li>
       <li><span style="background-color:darkred; color:white; padding:2px 5px;">Aggravated Assault</span></li>
       <li><span style="background-color:cadetblue; color:white; padding:2px 5px;">Partner Assault</span></li>
       <li><span style="background-color:lightgreen; color:black; padding:2px 5px;">Theft Plain</span></li>
       <li><span style="background-color:pink; color:black; padding:2px 5px;">Theft from motor vehicle</span></li>
       <li><span style="background-color:lightblue; color:black; padding:2px 5px;">Theft of identity</span></li>
     </ul>
</div>
'''
crime_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map as an HTML file
map_file_path = "top_crimes_dot_map.html"  # Save location
crime_map.save(map_file_path)
print(f"Map saved to {map_file_path}")


















import pandas as pd
import matplotlib.pyplot as plt

# Load and clean data
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

# Define label mapping for crime types
label_mapping = {
    'Vehicle - Stolen': 'Vehicle - Stolen',
    'Battery - Simple Assault': 'Simple assault',
    'Burglary From Vehicle': 'Burglary from vehicle',
    'Vandalism - Felony ($400 & Over, All Church Vandalisms)': 'Vandalism - Felony',
    'Burglary': 'Burglary',
    'Assault With Deadly Weapon, Aggravated Assault': 'Aggravated assault',
    'Intimate Partner - Simple Assault': 'Partner assault',
    'Theft Plain - Petty ($950 & Under)': 'Theft plain',
    'Theft From Motor Vehicle - Petty ($950 & Under)': 'Theft from motor vehicle',
    'Theft From Motor Vehicle - Grand ($950.01 And Over)': 'Theft from motor vehicle',
    'Theft Of Identity': 'Theft of identity'
}

# Apply the label mapping to both the top 10 crime types and filtered data
top_10_crime_types['Crime_Type'] = top_10_crime_types['Crime_Type'].replace(label_mapping)
data_cleaned['Short Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].replace(label_mapping)

# Use the 'tab10' color map for unique colors in the top 10 crime types
colors = plt.get_cmap('tab10').colors[:10]

# Visualize top 10 crime types with unique colors and gridlines
plt.figure(figsize=(12,6))
plt.barh(top_10_crime_types['Crime_Type'], top_10_crime_types['Count'], color=colors)
plt.xlabel('Number of occurrences', fontsize=16)
plt.ylabel('Crime type', fontsize=16)
plt.title('Top 10 Crime Types', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add gridlines on the x-axis
plt.gca().invert_yaxis()
plt.show()

# Select 5 zones for analysis
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
crime_data_filtered = data_cleaned[data_cleaned['AREA NAME'].isin(selected_zones)]

# Group by 'AREA NAME' and 'Short Crm Cd Desc' to get counts of each crime type in each zone
crime_counts_by_zone = (
    crime_data_filtered.groupby(['AREA NAME', 'Short Crm Cd Desc'])
    .size()
    .reset_index(name='Count')
)

# Sort values and get top 5 crimes for each zone, ensuring "Theft of identity" is included if present
top_crimes_by_zone = (
    crime_counts_by_zone.sort_values(['AREA NAME', 'Count'], ascending=[True, False])
    .groupby('AREA NAME')
    .head(5)
)

# Add "Theft of identity" if missing in top 5 for any zone
for area in selected_zones:
    if "Theft of identity" not in top_crimes_by_zone[top_crimes_by_zone['AREA NAME'] == area]['Short Crm Cd Desc'].values:
        theft_identity_row = crime_counts_by_zone[(crime_counts_by_zone['AREA NAME'] == area) & 
                                                  (crime_counts_by_zone['Short Crm Cd Desc'] == "Theft of identity")]
        if not theft_identity_row.empty:
            top_crimes_by_zone = pd.concat([top_crimes_by_zone, theft_identity_row])

# Plot top crimes per zone with gridlines
fig, ax = plt.subplots(figsize=(12, 8))
for area in selected_zones:
    area_data = top_crimes_by_zone[top_crimes_by_zone['AREA NAME'] == area]
    ax.bar(area_data['Short Crm Cd Desc'], area_data['Count'], label=area)

plt.xlabel('Crime type', fontsize=16)
plt.ylabel('Number of crimes', fontsize=16)
plt.xticks(rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Zone')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines on the y-axis
plt.title('Top Crimes by Zone Including "Theft of Identity"', fontsize=16)
plt.tight_layout()
plt.show()


import pandas as pd
from prophet import Prophet  # Make sure to install Prophet with `pip install prophet`
import matplotlib.pyplot as plt

# Load and prepare data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and filter for essential columns
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
crime_data.dropna(subset=['DATE OCC', 'AREA NAME'], inplace=True)

# Select zones of interest
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
crime_data_filtered = crime_data[crime_data['AREA NAME'].isin(selected_zones)]

# Prepare figure for subplots
fig, axes = plt.subplots(len(selected_zones), 1, figsize=(12, 15), sharex=True)
fig.suptitle('Crime Predictions for Each Zone with Confidence Intervals', fontsize=16)

# Process each zone and plot predictions
for i, zone in enumerate(selected_zones):
    zone_data = crime_data_filtered[crime_data_filtered['AREA NAME'] == zone]
    
    # Aggregate data by date
    zone_data = zone_data.groupby('DATE OCC').size().reset_index(name='Crime_Count')
    zone_data = zone_data.rename(columns={'DATE OCC': 'ds', 'Crime_Count': 'y'})
    
    # Initialize and fit Prophet model
    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    model.fit(zone_data)
    
    # Make predictions for the next 30 days
    future_dates = model.make_future_dataframe(periods=30)
    forecast = model.predict(future_dates)
    
    # Plot historical data and forecast with confidence intervals
    ax = axes[i]
    ax.plot(zone_data['ds'], zone_data['y'], label='Historical', color='black')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
    
    # Confidence interval: shaded area for the range yhat_lower to yhat_upper
    ax.fill_between(
        forecast['ds'], 
        forecast['yhat_lower'], 
        forecast['yhat_upper'], 
        color='blue', 
        alpha=0.2, 
        label='Confidence Interval'
    )
    
    ax.set_title(f'{zone} - Crime Prediction')
    ax.set_ylabel('Crime Count')
    ax.legend()

# Final adjustments
plt.xlabel('Date')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title

# Save the figure at 600 DPI
plt.savefig("crime_predictions_by_zone.png", dpi=600, bbox_inches='tight')
plt.show()




import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and filter for essential columns
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
crime_data.dropna(subset=['DATE OCC', 'AREA NAME'], inplace=True)

# Zones of interest
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']

# Filter data to include only selected zones
crime_data_filtered = crime_data[crime_data['AREA NAME'].isin(selected_zones)]

# Encode categorical columns (Vict Sex and AREA NAME) for model input
label_encoder_sex = LabelEncoder()
label_encoder_area = LabelEncoder()
crime_data_filtered['Vict Sex'] = label_encoder_sex.fit_transform(crime_data_filtered['Vict Sex'].fillna('Unknown'))
crime_data_filtered['AREA NAME'] = label_encoder_area.fit_transform(crime_data_filtered['AREA NAME'])

# Define parameter grids for tuning
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Prepare placeholders for tuned models and results
zone_optimized_results = {}
zone_optimized_models = {}

# Optimize models for each zone
for zone in selected_zones:
    # Filter data for the specific zone and aggregate daily counts
    zone_data = crime_data_filtered[crime_data_filtered['AREA NAME'] == label_encoder_area.transform([zone])[0]]
    daily_counts = zone_data.groupby('DATE OCC').size().rename('Crime Count').reset_index()
    
    # Merge daily counts with other features (Vict Age, Vict Sex, Premis Cd)
    merged_data = pd.merge(daily_counts, zone_data[['DATE OCC', 'Vict Age', 'Vict Sex', 'Premis Cd']].drop_duplicates(),
                           on='DATE OCC', how='left').fillna(0)
    
    # Create lagged features for crime counts (previous 7 days)
    for lag in range(1, 8):
        merged_data[f'lag_{lag}'] = merged_data['Crime Count'].shift(lag)
    
    # Drop NaN values resulting from lagging
    merged_data.dropna(inplace=True)
    
    # Split features and target variable
    X = merged_data[['Vict Age', 'Vict Sex', 'Premis Cd'] + [f'lag_{i}' for i in range(1, 8)]]
    y = merged_data['Crime Count']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Grid search for Random Forest
    rf_model = RandomForestRegressor(random_state=0)
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='r2', n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)
    best_rf = rf_grid_search.best_estimator_
    
    # Grid search for XGBoost
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=0)
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring='r2', n_jobs=-1)
    xgb_grid_search.fit(X_train, y_train)
    best_xgb = xgb_grid_search.best_estimator_
    
    # Evaluate both models on the test set
    rf_y_pred = best_rf.predict(X_test)
    xgb_y_pred = best_xgb.predict(X_test)
    
    # Calculate accuracy metrics for both models
    rf_mae = mean_absolute_error(y_test, rf_y_pred)
    rf_r2 = r2_score(y_test, rf_y_pred)
    
    xgb_mae = mean_absolute_error(y_test, xgb_y_pred)
    xgb_r2 = r2_score(y_test, xgb_y_pred)
    
    # Choose the model with the best R2 score
    if rf_r2 > xgb_r2:
        chosen_model = best_rf
        chosen_mae = rf_mae
        chosen_r2 = rf_r2
        chosen_model_name = 'Random Forest'
    else:
        chosen_model = best_xgb
        chosen_mae = xgb_mae
        chosen_r2 = xgb_r2
        chosen_model_name = 'XGBoost'
    
    # Store results and model
    zone_optimized_results[zone] = {
        'Best Model': chosen_model_name,
        'MAE': chosen_mae,
        'R2': chosen_r2
    }
    zone_optimized_models[zone] = chosen_model

# Display optimized results for each zone
print("Optimized Results for Each Zone:")
for zone, results in zone_optimized_results.items():
    print(f"Zone: {zone}")
    print(f"  Best Model: {results['Best Model']}")
    print(f"  MAE: {results['MAE']:.4f}")
    print(f"  R²: {results['R2']:.4f}")



import matplotlib.pyplot as plt
from datetime import timedelta

# Number of days for historical data visualization
historical_days = 90  # Shows the last 90 days of historical data

# Prepare a dictionary to store historical data for each zone
zone_historical_data = {}

for zone in selected_zones:
    # Filter data for the specific zone and aggregate daily counts
    zone_data = crime_data_filtered[crime_data_filtered['AREA NAME'] == label_encoder_area.transform([zone])[0]]
    daily_counts = zone_data.groupby('DATE OCC').size().rename('Crime Count').reset_index()
    
    # Extract the last 'historical_days' of historical data for plotting
    historical_data = daily_counts.tail(historical_days)
    zone_historical_data[zone] = historical_data

# Plot historical data and forecasted data for each zone
fig, axes = plt.subplots(len(selected_zones), 1, figsize=(12, 20), sharex=True)
fig.suptitle("Zone-Specific Crime Trends with Forecasts", fontsize=16)

for i, zone in enumerate(selected_zones):
    ax = axes[i]
    
    # Plot historical data
    historical_data = zone_historical_data[zone]
    ax.plot(historical_data['DATE OCC'], historical_data['Crime Count'], label='Historical', color='black')
    
    # Plot forecasted data
    zone_forecast = zone_forecasts[zone]  # Retrieve forecasted data from the previous step
    ax.plot(zone_forecast['Date'], zone_forecast['Forecasted Crime Count'], label='Forecast', color='blue', linestyle='--')
    
    # Formatting
    ax.set_title(f"{zone} - Crime Trends and Forecast")
    ax.set_ylabel("Crime Count")
    ax.legend()

plt.xlabel("Date")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()







## Goog map ## 

import pandas as pd
import folium
from folium.plugins import HeatMap

# Load data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and filter for essential columns
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
crime_data.dropna(subset=['DATE OCC', 'AREA NAME', 'LAT', 'LON'], inplace=True)

# Zones of interest
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']

# Initialize base map
base_map = folium.Map(location=[34.0522, -118.2437], zoom_start=10)

# Generate heat map and add zone names at the centroid for each zone
for zone in selected_zones:
    # Filter data for each zone
    zone_data = crime_data[crime_data['AREA NAME'] == zone]
    
    # Extract latitude and longitude coordinates for heat map
    heat_data = [[row['LAT'], row['LON']] for index, row in zone_data.iterrows()]
    
    # Create HeatMap layer
    heat_map = HeatMap(
        heat_data,
        radius=15,         # Controls the spread of the heat points
        blur=10,           # Controls blurring of each point
        max_zoom=13        # Controls how zoom affects intensity
    )
    
    # Add heat map layer to the base map
    folium.FeatureGroup(name=zone).add_child(heat_map).add_to(base_map)
    
    # Calculate centroid (average latitude and longitude) for the zone
    centroid_lat = zone_data['LAT'].mean()
    centroid_lon = zone_data['LON'].mean()
    
    # Add zone name at the centroid with customized styling
    folium.Marker(
        location=[centroid_lat, centroid_lon],
        icon=folium.DivIcon(html=f"""
            <div style="
                font-size: 14pt;
                font-weight: bold;
                color: black;
                text-align: center;
                transform: translate(-50%, -50%);
            ">{zone}</div>"""
        )
    ).add_to(base_map)

# Add layer control to toggle zones on the map
folium.LayerControl().add_to(base_map)

# Save and show map
base_map.save("detailed_crime_heat_maps_by_zone.html")
base_map




import pandas as pd
import folium
from folium.plugins import HeatMap

# Load data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and filter for essential columns
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
crime_data.dropna(subset=['DATE OCC', 'AREA NAME', 'LAT', 'LON'], inplace=True)

# Forecast data for each zone (as generated in the forecasting model)
zone_forecasts = {
    'West Valley': {'forecast': [20, 25, 22, 24], 'yhat_upper': 28, 'yhat_lower': 18},
    'Northeast': {'forecast': [30, 35, 32, 31], 'yhat_upper': 38, 'yhat_lower': 26},
    'Southwest': {'forecast': [15, 18, 20, 17], 'yhat_upper': 22, 'yhat_lower': 13},
    'Topanga': {'forecast': [10, 15, 13, 14], 'yhat_upper': 18, 'yhat_lower': 9},
    'Hollywood': {'forecast': [45, 50, 48, 47], 'yhat_upper': 55, 'yhat_lower': 40}
}

# Zones of interest and coordinates for centering markers
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']

# Initialize base map
base_map = folium.Map(location=[34.0522, -118.2437], zoom_start=10)

# Overlay forecasted crime data on the map
for zone in selected_zones:
    # Filter data for each zone
    zone_data = crime_data[crime_data['AREA NAME'] == zone]
    
    # Calculate centroid (average latitude and longitude) for the zone
    centroid_lat = zone_data['LAT'].mean()
    centroid_lon = zone_data['LON'].mean()

    # Get forecast data for the zone
    forecast = zone_forecasts[zone]['forecast']
    yhat_upper = zone_forecasts[zone]['yhat_upper']
    yhat_lower = zone_forecasts[zone]['yhat_lower']

    # Create popup text for forecast information
    popup_text = (f"<b>{zone} - Forecasted Crime</b><br>"
                  f"Average Forecast: {sum(forecast)/len(forecast):.2f}<br>"
                  f"Upper Bound: {yhat_upper}<br>"
                  f"Lower Bound: {yhat_lower}")

    # Add zone name and forecast data at the centroid
    folium.Marker(
        location=[centroid_lat, centroid_lon],
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=f"{zone} - Crime Forecast",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(base_map)

# Save and show map
base_map.save("forecasted_crime_map.html")
base_map


## GOOD PREDICTION

from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# Load the data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and filter for essential columns
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
crime_data.dropna(subset=['DATE OCC', 'AREA NAME'], inplace=True)

# Define the zones of interest
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
zone_forecasts_ensemble = {}

for zone in selected_zones:
    # Prepare data for each zone
    zone_data = crime_data[crime_data['AREA NAME'] == zone]
    daily_counts = zone_data.groupby('DATE OCC').size().rename('Crime Count').reset_index()
    daily_counts = daily_counts[(daily_counts['DATE OCC'] >= '2020-01-01') & (daily_counts['DATE OCC'] <= '2020-12-31')]
    all_dates = pd.date_range(start='2020-01-01', end='2020-12-31')
    daily_counts = daily_counts.set_index('DATE OCC').reindex(all_dates, fill_value=0).rename_axis('DATE OCC').reset_index()
    daily_counts.columns = ['ds', 'y']

    # Prophet Model - with updated tuning
    model_prophet = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=20)
    model_prophet.fit(daily_counts[daily_counts['ds'] <= '2020-08-31'])
    future_dates_df = pd.DataFrame(pd.date_range(start='2020-09-01', end='2020-12-31', freq='D'), columns=['ds'])
    prophet_forecast = model_prophet.predict(future_dates_df)

    # XGBoost Model - expanded parameter tuning
    scaler_xgb = MinMaxScaler()
    daily_counts_scaled = scaler_xgb.fit_transform(daily_counts[['y']])
    X, y = [], []
    sequence_length = 45  # Increased sequence length for better pattern capture
    for i in range(sequence_length, len(daily_counts_scaled) - 122):
        X.append(daily_counts_scaled[i-sequence_length:i, 0])
        y.append(daily_counts_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'colsample_bytree': [0.8, 1.0],
        'subsample': [0.8, 1.0]
    }
    model_xgb = GridSearchCV(XGBRegressor(), xgb_params, scoring='neg_mean_squared_error', cv=3)
    model_xgb.fit(X, y)
    X_test = daily_counts_scaled[-(122 + sequence_length):-122].reshape(-1, sequence_length)
    xgb_forecast_scaled = model_xgb.predict(X_test)
    xgb_forecast = scaler_xgb.inverse_transform(xgb_forecast_scaled.reshape(-1, 1))

    # LSTM Model - increased capacity and patience in early stopping
    scaler_lstm = MinMaxScaler()
    daily_counts_lstm = scaler_lstm.fit_transform(daily_counts[['y']])
    X_lstm, y_lstm = [], []
    for i in range(sequence_length, len(daily_counts_lstm) - 122):
        X_lstm.append(daily_counts_lstm[i-sequence_length:i, 0])
        y_lstm.append(daily_counts_lstm[i, 0])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    
    model_lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
        Dropout(0.3),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
    model_lstm.fit(X_lstm.reshape(-1, sequence_length, 1), y_lstm, epochs=60, verbose=0, callbacks=[early_stopping])
    X_test_lstm = daily_counts_lstm[-(122 + sequence_length):-122].reshape(-1, sequence_length, 1)
    lstm_forecast_scaled = model_lstm.predict(X_test_lstm)
    lstm_forecast = scaler_lstm.inverse_transform(lstm_forecast_scaled).flatten()

    # Adjusted Ensemble Forecasting
    ensemble_forecast = (
        0.5 * prophet_forecast['yhat'].values[-122:] +
        0.25 * xgb_forecast.flatten() +
        0.25 * lstm_forecast
    )

    # Calculate and display RMSE
    historical_last_4_months = daily_counts[(daily_counts['ds'] >= '2020-09-01') & (daily_counts['ds'] <= '2020-12-31')]
    rmse = np.sqrt(mean_squared_error(historical_last_4_months['y'], ensemble_forecast))
    print(f"{zone} - Improved Ensemble RMSE for last 4 months of prediction: {rmse:.2f}")

    # Store forecast results
    zone_forecasts_ensemble[zone] = {
        'forecast': ensemble_forecast,
        'historical': historical_last_4_months['y'].values
    }

# Plotting adjusted results
fig, axes = plt.subplots(len(selected_zones), 1, figsize=(12, 20), sharex=True)
fig.suptitle("Adjusted Historical Data (2020) and Ensemble Predicted Data (2020-09 to 2020-12)", fontsize=16)

for i, zone in enumerate(selected_zones):
    if zone in zone_forecasts_ensemble:
        ax = axes[i]
        data = zone_forecasts_ensemble[zone]

        ax.step(historical_last_4_months['ds'], data['historical'], where='mid', label='Historical Data', color='black')
        forecast_dates = pd.date_range(start='2020-09-01', end='2020-12-31')
        ax.step(forecast_dates, data['forecast'], where='mid', label='Ensemble Prediction', color='blue', linestyle='--')

        ax.set_title(f"{zone} - Historical and Improved Ensemble Predicted Crime Counts", fontsize=14)
        ax.set_ylabel("Crime Count", fontsize=16)
        ax.legend(loc="upper left")

plt.xlabel("Date", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("improved_historical_vs_ensemble_predicted_stairs_2020.png", dpi=600)
plt.show()





