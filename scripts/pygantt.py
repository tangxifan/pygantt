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
  df.start = pd.to_datetime(df.start)

  # If either 'duration' or 'end' can be accepted
  # - if 'duration' is defined while 'end' is not, 'end' will be caculated based on 'start' and 'duration'
  # Note: duration is always counted in days
  # - As long as 'end' is defined, we will calculate 'duration' based on 'start' and 'end' 
  if ('duration' in df):
    df.duration = pd.to_timedelta(df.duration, unit='D')

  if ('end' in df):
    df.end = pd.to_datetime(df.end)
  
  # Sort in ascending order of start date
  if (sort_start_date):
    df = df.sort_values(by='start', ascending=True)
  
  # Sort based on Department
  if (sort_department):
    df = df.sort_values(by='Department', 
                        ascending=False).reset_index(drop=True)

  # Compute duration for each task in the unit of days
  if ('duration' in df):
    df['end'] = df.start + df.duration - pd.DateOffset(days=1)
  else:
    df['duration'] = df.end - df.start

  df.duration = df.duration.apply(lambda x: x.days)

  # Compute progress bar for each task
  df['current_progress'] = round(df.Completion * df.duration / 100, 2)

  # Force milestone duration to be 0
  df.loc[df.Task.str.startswith("M"), 'duration'] = 0 

  df.head()

  return df

#####################################################################
# Plot data into figures
#####################################################################
def plot_data(df, time_unit, show_progress_bar):
  ################ 
  # Basic concepts
  # Critical variables to be computed when preprocessing the data for each task
  #  
  #      | relative_start  |<--width_completion-->|
  #      |<--------------->|<---------------------+------------->|<--------->|
  #      |                 |             task[i].duration        |           |
  #      |                 |                                     |           |
  #      |                 |                                     |           |
  #      |          task[i].start                          task[i].end       |
  #      |                                                                   |
  # project_start                                                       project_end
  
  # Project level variables
  project_start = df.start.min()
  project_end = df.end.max()
  project_duration = (project_end - project_start).days + 1
  
  # Compute relative date to the project start date
  df['relative_start'] = df.start.apply(lambda x: (x - project_start).days)

  # Create custom x-ticks and x-tick labels
  interval = xtick_interval_map[time_unit]
  xticks = np.arange(0, project_duration - 0, interval)
  xticks_labels = pd.date_range(start=project_start, end=project_end, freq=time_unit_map[time_unit], closed='left').strftime("%m/%d") 
  print(project_start)
  print(project_end)
  print(xticks)
  print(xticks_labels)
  xticks_minor = np.arange(1, project_duration, 1)
  
  # Create custom y-ticks and y-tick labels
  yticks = [i for i in range(len(df.Task))]

  # create a column with the color for each department
  df['color'] = df.apply(color, axis=1) 
  
  ###### PLOTTING GANTT CHART ######
  fig, ax = plt.subplots(1, figsize=(16,6))
  plt.title('Gantt Chart', size=18)

  # Compute color ratio
  default_color_alpha = 1
  progress_bar_color_alpha = 1
  if (show_progress_bar):
    default_color_alpha = 0.4
  
  # Plot bars for tasks
  for idx, row in df.iterrows():
    if df.Task[idx].startswith("T"):
      ax.barh(y=df.Task[idx], left=df.relative_start[idx], width=df.duration[idx],
              color=df.color[idx], edgecolor='black', linewidth='2', alpha=default_color_alpha)
      # Show texts for each task
      if (row.relative_start + row.duration < 0.75 * project_duration):
        ax.text(row.relative_start + row.duration + 0.5, idx, 
                row.description,
                color='black',
                va='center', alpha=0.8,
                horizontalalignment='left')
      else:
        ax.text(row.relative_start - 0.5, idx, 
                row.description,
                color='black',
                va='center', alpha=0.8,
                horizontalalignment='right')
    elif df.Task[idx].startswith("M"):
      ax.barh(y=df.Task[idx], left=df.relative_start[idx], width=df.duration[idx],
              color='white', edgecolor='white', linewidth='2', alpha=default_color_alpha)
      # Show texts for each milestone:
      # - If the relative start + duration is less than 75%, we plot the text on the right side of the marker
      # - If the relative start + duration is more than 75%, we plot the text on the left side of the marker
      if (row.relative_start + row.duration < 0.75 * project_duration):
        ax.text(row.relative_start + row.duration + 0.5, idx, 
                row.description, 
                va='center', alpha=0.8,
                horizontalalignment='left')
      else:
        ax.text(row.relative_start + row.duration - 0.5, idx, 
                row.description, 
                va='center', alpha=0.8,
                horizontalalignment='right')

    # Plot milestone markers
    if df.Task[idx].startswith("M"):
      ax.plot(df.relative_start[idx], idx, color='tab:orange', marker='D', markersize='6')
    # Plot milestone vertical lines which is easier for readers
    if df.Task[idx].startswith("M"):
      cur_ms_line_height = 1 - idx / len(df.Task)
      plt.axvline(df.relative_start[idx], ymax=cur_ms_line_height, color='tab:orange', linestyle='--', linewidth='0.5')

  # Plot progress bars for tasks
  if (show_progress_bar):
    for idx, row in df.iterrows():
      if df.Task[idx].startswith("T"):
        ax.barh(y=df.Task[idx], left=df.relative_start[idx], width=df.current_progress[idx],
                color=df.color[idx], edgecolor='black', linewidth='2', alpha=progress_bar_color_alpha)
        # Show texts on progress for each task
        ax.text(row.relative_start + row.duration / 2, idx, 
                f"{int(row.Completion*100)}%", 
                color='white',
                va='center', alpha=0.8)
      elif df.Task[idx].startswith("M"):
        ax.barh(y=df.Task[idx], left=df.relative_start[idx], width=df.current_progress[idx],
                color='white', edgecolor='white', linewidth='2', alpha=progress_bar_color_alpha)

  plt.gca().invert_yaxis()
  ax.set_xticks(xticks)
  ax.set_xticks(xticks_minor, minor=True)
  #ax.set_xticklabels(xticks_labels,fontsize='8')

  # Show grids
  plt.grid(axis='x')
  
  # Legends
  legend_elements = color_legends()
  plt.legend(handles=legend_elements)

  return plt

#####################################################################
# Plot data
#####################################################################
def save_figure(plt, image_name, image_format, image_dpi):
  plt.savefig(image_name, format=image_format, dpi=image_dpi) 

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
  plt = plot_data(df, args.time_unit, args.progress_bar)
  save_figure(plt, args.output_figure, 'jpg', 1200)
