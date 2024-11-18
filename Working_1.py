import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib.colors as mcolors

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
data_cleaned.dropna(subset=['Crm Cd', 'Premis Cd'], inplace=True)

# Add columns for Year, Month, and Day of the Week
data_cleaned['Year'] = data_cleaned['DATE OCC'].dt.year
data_cleaned['Month'] = data_cleaned['DATE OCC'].dt.month
data_cleaned['DayOfWeek'] = data_cleaned['DATE OCC'].dt.day_name()

# Format crime descriptions to title case
data_cleaned['Crm Cd Desc'] = data_cleaned['Crm Cd Desc'].str.title()

# Analyze by Crime Type
crime_type_counts = data_cleaned['Crm Cd Desc'].value_counts().reset_index()
crime_type_counts.columns = ['Crime_Type', 'Count']

# Display top 10 crime types for verification
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
    color = colors[index % 10]  # Rotate colors if >10
    folium.CircleMarker(
        location=(row['LAT'], row['LON']),
        radius=1,
        color=mcolors.to_hex(color),  # Convert RGB to hex for folium
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
&emsp;<b>Crime Type Legend</b><br>
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



# Mapping Crime Occurrences: Point Map with Draggable Legend
point_map = folium.Map(location=[latitude, longitude], zoom_start=10)

# Add circle markers with colors based on crime type
for index, row in data_cleaned.iterrows():
    crime_type = row['Crm Cd Desc']
    color = colors[index % 10]  # Rotate colors if >10
    folium.CircleMarker(
        location=(row['LAT'], row['LON']),
        radius=1,
        color=mcolors.to_hex(color),  # Convert RGB to hex for folium
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['Crm Cd Desc']} - {row['AREA NAME']}"
    ).add_to(point_map)

# Create a draggable legend with font size 16
legend_html = """
<div id="legend" style="position: absolute; 
                         bottom: 50px; left: 50px; 
                         width: 250px; height: 300px; 
                         border:2px solid grey; 
                         z-index:9999; 
                         font-size:16px; 
                         background-color:white; 
                         padding: 10px; 
                         border-radius: 5px; 
                         cursor: move;">
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

<script>
    const legend = document.getElementById('legend');
    let isMouseDown = false, offsetX, offsetY;

    legend.addEventListener('mousedown', function(e) {
        isMouseDown = true;
        offsetX = e.clientX - legend.offsetLeft;
        offsetY = e.clientY - legend.offsetTop;
    });

    document.addEventListener('mouseup', function() {
        isMouseDown = false;
    });

    document.addEventListener('mousemove', function(e) {
        if (isMouseDown) {
            legend.style.left = (e.clientX - offsetX) + 'px';
            legend.style.top = (e.clientY - offsetY) + 'px';
        }
    });
</script>
"""
point_map.get_root().html.add_child(folium.Element(legend_html))

# Save the point map with the draggable legend
point_map_save_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\crime_point_map_colored_draggable.html'
point_map.save(point_map_save_path)
print(f"Point map with draggable crime legend has been saved to '{point_map_save_path}'")









# Mapping Crime Occurrences: Point Map with Draggable Legend
point_map = folium.Map(location=[latitude, longitude], zoom_start=10)

# Add circle markers with colors based on crime type
for index, row in data_cleaned.iterrows():
    crime_type = row['Crm Cd Desc']
    color = colors[index % 10]  # Rotate colors if >10
    folium.CircleMarker(
        location=(row['LAT'], row['LON']),
        radius=1,
        color=mcolors.to_hex(color),  # Convert RGB to hex for folium
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['Crm Cd Desc']} - {row['AREA NAME']}"
    ).add_to(point_map)

legend_html = """
<div id="legend" style="position: absolute; 
                         bottom: 50px; left: 50px; 
                         width: 300px; height: 350px; 
                         border:2px solid grey; 
                         z-index:9999; 
                         font-size:18px; 
                         background-color:white; 
                         padding: 15px; 
                         border-radius: 5px; 
                         cursor: move;">
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

<script>
    const legend = document.getElementById('legend');
    let isMouseDown = false, offsetX, offsetY;

    legend.addEventListener('mousedown', function(e) {
        isMouseDown = true;
        offsetX = e.clientX - legend.offsetLeft;
        offsetY = e.clientY - legend.offsetTop;
    });

    document.addEventListener('mouseup', function() {
        isMouseDown = false;
    });

    document.addEventListener('mousemove', function(e) {
        if (isMouseDown) {
            legend.style.left = (e.clientX - offsetX) + 'px';
            legend.style.top = (e.clientY - offsetY) + 'px';
        }
    });
</script>
"""

point_map.get_root().html.add_child(folium.Element(legend_html))

# Save the point map with the draggable legend
point_map_save_path = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures\crime_point_map_colored_draggable.html'
point_map.save(point_map_save_path)
print(f"Point map with draggable crime legend has been saved to '{point_map_save_path}'")










