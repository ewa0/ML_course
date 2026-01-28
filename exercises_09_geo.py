import pandas as pd
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import iplot

cf.go_offline()
# ***** World Power Consumption ***** #
path = "/home/ewa/Documents/Course/course_materials/Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting"
wpc = pd.read_csv(path + "/2014_World_Power_Consumption")
print("World Power Consumption:\n", wpc.head())
data_wpc = dict(type="choropleth",
                colorscale="viridis",
                locations=wpc["Country"],
                z=wpc["Power Consumption KWH"],
                locationmode="country names",
                text=wpc["Text"],
                marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
                colorbar={"title": "Power Consumption KWH"})
choromap_wpc = go.Figure(data=[data_wpc], layout=dict(geo={"scope": "world", }))
iplot(choromap_wpc, validate=False)

# ***** Election Data ***** #
ed = pd.read_csv(path + "/2012_Election_Data")
print("\nElection Data:\n", ed.head())
data_ed = dict(type="choropleth",
               colorscale="algae",
               locations=ed["State Abv"],
               z=ed["Voting-Age Population (VAP)"],
               locationmode="USA-states",
               text=ed["Voting-Age Population (VAP)"],
               marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
               colorbar={"title": "Voting-Age Population"})
choromap_ed = go.Figure(data=[data_ed], layout=dict(geo={"scope": "usa", }))
iplot(choromap_ed, validate=False)
