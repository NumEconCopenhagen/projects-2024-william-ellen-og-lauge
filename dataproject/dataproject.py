# Importing the relevant packages

import ipywidgets as widgets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from IPython.display import display
import matplotlib.ticker as mtick

# Creating a population pyramid for 2023

def plot_population_pyramid(datapop_df):
    # Splitting the DataFrame into one for men and one for women
    men_df = datapop_df[datapop_df['KØN'] == 'Men']
    women_df = datapop_df[datapop_df['KØN'] == 'Women']

    # Since the 'women_df' population counts need to be negative for the pyramid plot, we'll multiply by -1
    women_df['INDHOLD'] = women_df['INDHOLD'] * -1

    # Now, we plot the population pyramid
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the bar plots for men (positive) and women (negative) to create the pyramid effect
    ax.barh(men_df['ALDER'], men_df['INDHOLD'], color='blue', label='Men')
    ax.barh(women_df['ALDER'], women_df['INDHOLD'], color='green', label='Women')

    # Adding labels and title
    ax.set_xlabel('Population Count')
    ax.set_ylabel('Age')
    ax.set_title('Population Pyramid for 2023')

    # Positive values on the x-axis
    ticks = ax.get_xticks()
    ax.set_xticklabels([int(abs(tick)) for tick in ticks])

    # Adding legends
    ax.legend()

    # Showing grid
    ax.grid(True)

    # Showing the plot
    plt.show()


def plot_population_pyramid1(datapop_api):
    # This function will update the plot when the slider is changed
    def update_plot(year):
        # Filter the DataFrame for the selected year
        df_year = datapop_api[datapop_api["TID"] == year]

        # Group by gender and age, then sum the population count
        age_pyramid_data = df_year.groupby(["KØN", "ALDER"])["INDHOLD"].sum().unstack("KØN")

        # Make women's population negative for plotting
        age_pyramid_data['Women'] = -age_pyramid_data['Women']

        # Clear the previous figure and create a new one
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the data
        ax.barh(age_pyramid_data.index, age_pyramid_data['Men'], color='blue', label='Men')
        ax.barh(age_pyramid_data.index, age_pyramid_data['Women'], color='green', label='Women')

        # Set labels and title
        ax.set_xlabel('Population Count')
        ax.set_ylabel('Age')
        ax.set_title(f'Population Pyramid for {year}')
        ax.legend()

        ax.set_ylim(0, age_pyramid_data.index.max())

        # Change y-axis to show only labels for every 5 years.
        ax.set_yticks(np.arange(0, 101, 5))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure that only integer ticks are shown

        # Set the x-axis to show positive values for both sides
        ticks = ax.get_xticks()
        ax.set_xticklabels([int(abs(tick)) for tick in ticks])

        # Show grid
        ax.grid(True)
        

        # Display the plot
        plt.show()
        interactive_plot = widgets.interactive(update_plot, year=year_slider)
    
    # Display the interactive plot
        display(interactive_plot)

    # Create a slider for the year selection
    years = datapop_api['TID'].unique()
    year_slider = widgets.IntSlider(min=min(years), max=max(years), step=1, value=min(years), description='Year')

    # Display the slider and attach the update function
    widgets.interactive(update_plot, year=year_slider)

# Creating plot of age groups towards 2070 in a bar-chart
  
def plot_age_groups(datapop_api, selected_years):
    # Defining the age groups and their labels
    bins = [0, 20, 50, 65, 80, float('inf')]
    labels = ['0-19', '20-49', '50-64', '65-79', '80+']

    # Creating a new column for the age groups
    datapop_api['AgeGroup'] = pd.cut(datapop_api['ALDER'], bins=bins, labels=labels, right=False)

    # Filtering the dataframe for the selected years
    datapop_api_figure2 = datapop_api[datapop_api['TID'].isin(selected_years)]

    # Grouping by year and age group, and calculating the total amount of people
    grouped = datapop_api_figure2.groupby(['TID', 'AgeGroup']).sum().reset_index()

    # Pivoting the dataframe so that each age group is a column
    pivot = grouped.pivot(index='TID', columns='AgeGroup', values='INDHOLD')

    # Creating a new figure with a specific size
    plt.figure(figsize=(10, 6))

    # Plotting the data
    ax = pivot.plot(kind='bar', stacked=True)

    # Formatting y-axis to display in millions
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'{x/1e6:.0f}'))

    plt.title('Total Amount of People by Age Group')
    plt.xlabel('Year')
    plt.ylabel('Total Amount of People (Millions)')

    # Moving the legend to the right side of the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    # Manually reordering legend labels
    handles, labels = ax.get_legend_handles_labels()
    order = [4, 3, 2, 1, 0]  # This will reverse the order of the legend
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    plt.show()

# Creating plot of 'HERKOMST' groups towards 2070 in a bar-chart

def plot_herkomst_groups(datapop_api, selected_years):
    # Ensuring 'INDHOLD' column is of numeric type
    datapop_api['INDHOLD'] = pd.to_numeric(datapop_api['INDHOLD'])

    # Defining the HERKOMST groups
    groups = ['Immigrants from western countries', 'Immigrants from non-western countries', 'Descendants from western countries', 'Descendants from non-western countries']

    # Filtering the dataframe for the selected years and HERKOMST groups
    datapop_api_figure3 = datapop_api[datapop_api['TID'].isin(selected_years) & datapop_api['HERKOMST'].isin(groups)]

    # Grouping by year and HERKOMST, and calculating the total amount of people
    grouped = datapop_api_figure3.groupby(['TID', 'HERKOMST'])['INDHOLD'].sum().reset_index()

    # Pivoting the dataframe so that each HERKOMST group is a column
    pivot = grouped.pivot(index='TID', columns='HERKOMST', values='INDHOLD')

    # Creating a new figure with a specific size
    plt.figure(figsize=(10, 6))

    # Plotting the data
    pivot.plot(kind='bar', stacked=True)

    plt.title('Total Amount of People by HERKOMST')
    plt.xlabel('Year')
    plt.ylabel('Total Amount of People (Millions)')

    # Formatting y-axis to display in millions with one decimal place
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'{x/1e6:.1f}'))

    # Moving the legend to the right side of the plot and make it horizontal
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc='upper left', ncol=1)

    plt.show()

def update_plot(year):
    # Filtering the DataFrame for the selected year
    df_year = datapop_api[datapop_api["TID"] == year]
    
    # Grouping by gender and age, then sum the population count
    age_pyramid_data = df_year.groupby(["KØN", "ALDER"])["INDHOLD"].sum().unstack("KØN")
    
    # Making women's population negative for plotting
    age_pyramid_data['Women'] = -age_pyramid_data['Women']

    # Clearing the previous figure and create a new one
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotting the data
    ax.barh(age_pyramid_data.index, age_pyramid_data['Men'], color='blue', label='Men')
    ax.barh(age_pyramid_data.index, age_pyramid_data['Women'], color='green', label='Women')

    # Setting labels and title
    ax.set_xlabel('Population Count')
    ax.set_ylabel('Age')
    ax.set_title(f'Population Pyramid for {year}')
    ax.legend()


    ax.set_ylim(0, age_pyramid_data.index.max())

    # Changing y-axis to show only labels for every 5 years
    # Finding the max age to set as the limit for the y-axis ticks
    # max_age = age_pyramid_data.index.max() + (5 - age_pyramid_data.index.max() % 5)  # Round up to the nearest multiple of 5
    ax.set_yticks(np.arange(0, 101, 5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure that only integer ticks are shown

    # Setting the x-axis to show positive values for both sides
    ticks = ax.get_xticks()
    ax.set_xticklabels([int(abs(tick)) for tick in ticks])

    # Showing grid
    ax.grid(True)

    # Displaying the plot
    plt.show()