#####################################################################
# A script that generates csv-based gantt chart into graphic files
#####################################################################
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import datetime as dt
import logging
import argparse
import numpy as np
import plotly.express as px

#####################################################################
# Initialize logger 
#####################################################################
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO);

#####################################################################
# Constants
#####################################################################
time_unit_map = {
  "day": 'D',
  "3day": '3D',
  "week": 'W',
  "month": 'M',
  "year": 'Y'
}

xtick_interval_map = {
  "day": 1,
  "3day": 3,
  "week": 7,
  "month": 30,
  "year": 365
}

# Color map
c_dict = {}

#####################################################################
# Create a column with the color for each department
#####################################################################
def color(row):
  return c_dict[row['Department']]

#####################################################################
# Create legend elements for each color
#####################################################################
def color_legends():
  return [Patch(facecolor=c_dict[i], label=i)  for i in c_dict]

#####################################################################
# Get the difference between two dates in different time units
#####################################################################
def diff_date(start, end, unit):
  x = pd.to_datetime(end) - pd.to_datetime(start)
  return int (x / np.timedelta64(1, unit))

#####################################################################
# Read color map from a csv file
#####################################################################
def read_color(csv_file):
  color_dict = pd.read_csv(csv_file, header=None, index_col=0, skiprows=1).squeeze("columns").to_dict()
  return color_dict

#####################################################################
# Read Data from a csv file
#####################################################################
def read_data(csv_file):
  df = pd.read_csv(csv_file)
  df.head()
  return df

#####################################################################
# Preprocessing data
#####################################################################
def process_data(df, sort_start_date, sort_department):
  # Convert dates to datetime format
  df.Start = pd.to_datetime(df.Start)

  # If either 'duration' or 'end' can be accepted
  # - if 'duration' is defined while 'end' is not, 'end' will be caculated based on 'start' and 'duration'
  # Note: duration is always counted in days
  # - As long as 'end' is defined, we will calculate 'duration' based on 'start' and 'end' 
  if ('Duration' in df):
    df.Duration = pd.to_timedelta(df.Duration, unit='D')

  if ('Finish' in df):
    df.Finish = pd.to_datetime(df.Finish)
  
  # Sort in ascending order of start date
  if (sort_start_date):
    df = df.sort_values(by='Start', ascending=True)
  
  # Sort based on Resource
  if (sort_department):
    df = df.sort_values(by='Resource', 
                        ascending=False).reset_index(drop=True)

  # Compute duration for each task in the unit of days
  if ('Duration' in df):
    df['Finish'] = df.Start + df.Duration - pd.DateOffset(days=1)
  else:
    df['Duration'] = df.Finish - df.Start

  df.Duration = df.Duration.apply(lambda x: x.days)

  # Force milestone duration to be 0
  df.loc[df.Task.str.startswith("M"), 'Duration'] = 0 

  df.head()

  return df

#####################################################################
# Plot data into figures
#####################################################################
def plot_gantt(df, time_unit, show_progress_bar):
  fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource", text="Description")
  fig.update_yaxes(autorange="reversed")
  fig.update_layout(yaxis={"tickmode": "array", "tickvals": df.index, "ticktext": df.Task})

  return fig

#####################################################################
# draw an arrow from the end of the first job to the start of the second job
#####################################################################
def draw_arrow_between_jobs(fig, first_job_dict, second_job_dict):
    ## retrieve tick text and tick vals
    print(fig.layout.yaxis)
    print(fig.layout.yaxis)
    job_yaxis_mapping = dict(zip(fig.layout.yaxis.ticktext, fig.layout.yaxis.tickvals))
    jobs_delta = second_job_dict['Start'] - first_job_dict['Finish']
    ## horizontal line segment
    fig.add_shape(
        x0=first_job_dict['Finish'], y0=job_yaxis_mapping[first_job_dict['Task']], 
        x1=first_job_dict['Finish'] + jobs_delta/2, y1=job_yaxis_mapping[first_job_dict['Task']],
        line=dict(color="blue", width=2)
    )
    ## vertical line segment
    fig.add_shape(
        x0=first_job_dict['Finish'] + jobs_delta/2, y0=job_yaxis_mapping[first_job_dict['Task']], 
        x1=first_job_dict['Finish'] + jobs_delta/2, y1=job_yaxis_mapping[second_job_dict['Task']],
        line=dict(color="blue", width=2)
    )
    ## horizontal line segment
    fig.add_shape(
        x0=first_job_dict['Finish'] + jobs_delta/2, y0=job_yaxis_mapping[second_job_dict['Task']], 
        x1=second_job_dict['Start'], y1=job_yaxis_mapping[second_job_dict['Task']],
        line=dict(color="blue", width=2)
    )
    ## draw an arrow
    fig.add_annotation(
        x=second_job_dict['Start'], y=job_yaxis_mapping[second_job_dict['Task']],
        xref="x",yref="y",
        showarrow=True,
        ax=-10,
        ay=0,
        arrowwidth=2,
        arrowcolor="blue",
        arrowhead=2,
    )
    return fig

#####################################################################
# Plot data
#####################################################################
def save_figure(fig, image_name, image_format, image_dpi):
  fig.write_image(image_name, format=image_format) 

#####################################################################
# Main function
#####################################################################
if __name__ == '__main__':
  # Execute when the module is not initialized from an import statement

  # Parse the options and apply sanity checks
  parser = argparse.ArgumentParser(description='Gantt chart render')
  parser.add_argument('--input_csv',
                      required=True,
                      help='CSV file which contains gantt data')
  parser.add_argument('--colormap_csv',
                      required=True,
                      help='CSV file which contains the colormap for different owners in the gantt')
  parser.add_argument('--output_figure',
                      required=True,
                      help='Figure to be outputted')
  parser.add_argument('--time_unit',
                      default='week',
                      help='Time unit in gantt chart: [day|3day|week|month|year]')
  parser.add_argument('--sort_start_date',
                      action='store_true',
                      help='Sort the start date of each task in gantt chart')
  parser.add_argument('--sort_department',
                      action='store_true',
                      help='Sort the department of each task in gantt chart')
  parser.add_argument('--progress_bar',
                      action='store_true',
                      help='Plot progress bar for each task in gantt chart')
  args = parser.parse_args()

  c_dict = read_color(args.colormap_csv)
  df = read_data(args.input_csv)
  df = process_data(df, args.sort_start_date, args.sort_department)
  fig = plot_gantt(df, args.time_unit, args.progress_bar)
  # Test: draw dependencies
  fig = draw_arrow_between_jobs(fig, df.to_dict('index')[0], df.to_dict('index')[3])
  save_figure(fig, args.output_figure, 'svg', 1200)
