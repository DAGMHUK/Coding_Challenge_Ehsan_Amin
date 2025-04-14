import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def replace_birthdate_with_age(df, birthdate_column, reference_date):
    reference_date = pd.to_datetime(reference_date)
    df['DriverAge'] = reference_date.year - pd.to_datetime(df[birthdate_column]).dt.year
    df.drop(columns=[birthdate_column], inplace=True)
    return df


def plot_attribute_distribution(data, column_name):
    # Count occurrences of each value in the specified column

    if data[column_name].nunique() > 100:
        fig = px.histogram(
            data,
            x=column_name,
            nbins=50,  # Adjust the number of bins as needed
            labels={'x': column_name, 'y': 'Frequency'},
            title=f'Distribution of {column_name} (Histogram)'
        )
        
        # Update layout for better appearance
        fig.update_layout(
            xaxis_title=column_name,
            yaxis_title='Frequency',
            width=800,  # Set figure width
            height=400  # Set figure height
        )
    else:
        value_counts = data[column_name].value_counts().sort_index()
        # Use a bar plot for smaller numbers of unique values
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={'x': column_name, 'y': 'Frequency'},
            title=f'Distribution of {column_name}'
        )
        # Add text on top of each bar
        fig.update_traces(text=value_counts.values, textposition='outside')

        # Update layout for better appearance
        fig.update_layout(
            xaxis_title=column_name,
            yaxis_title='Volume',
            xaxis=dict(tickmode='linear'),
            width= np.clip(50 * len(value_counts), 400, 1200),  # Set figure width with min 400 and max 1200
            height=400  # Set figure height
        )

    # Show the plot
    fig.show()
    

def load_data(file_path, engine='pyarrow'):
    """
    Reads and explores the content of a Parquet file.

    Args:
        file_path (str): Path to the Parquet file.

    Returns:
        pd.DataFrame: The loaded DataFrame for further exploration.
    """
    try:
        # Read the Parquet file
        df = pd.read_parquet(file_path)
        
        # Display basic information about the DataFrame
        print("DataFrame Info:")
        print(df.info())
        print("\n")
        print('##########################################################################')
        # Display the first few rows of the DataFrame
        print("First 5 Rows:")
        print(df.head())
        print("\n")
        print('##########################################################################')
        # Display summary statistics for numerical columns
        print("Summary Statistics:")
        print(df.describe())
        print("\n")
        print('##########################################################################')
        # Display the number of unique values per column
        print("Unique Values Per Column:")
        print(df.nunique())
        print("\n")
        print('##########################################################################')
        
        # Display the total number of missing values in the DataFrame
        print("Total Missing Values in DataFrame:")
        print(df.isna().sum())
        print("\n")
        print('##########################################################################')
        # Display column names and data types
        #print("Columns and Data Types:")
        #print(df.dtypes)
        #print("\n")
        
        # Display the number of missing values per column
       # print("Missing Values:")
        #print(df.isnull().sum())
        #print("\n")
        
        return df
    except Exception as e:
        print(f"An error occurred while reading the Parquet file: {e}")
        return None


def plot_grouped_by_ClaimNb(data, frequeny_column, column_name, max = None):
    # Group by the specified column and calculate claim frequency
    frequency = data.groupby(column_name).ClaimNb.sum() / data.groupby(column_name).Exposure.sum()
    frequency = pd.DataFrame(frequency, columns=[frequeny_column])

    # Generate scatter plot
    fig = px.scatter(
        frequency.reset_index(),
        x=column_name,
        y=frequeny_column,
        title=f"{frequeny_column} by {column_name}",
        labels={column_name: column_name, frequeny_column: "Claim Frequency"},
        template="plotly_white"
    )

    fig.update_traces(marker=dict(size=8, color="blue"))
    fig.update_layout(
        #xaxis=dict(range=[47, 153]),  # Adjust range as needed
        yaxis=dict(range=[0, max]) if max is not None else None,  # Adjust range as needed
        width=1000,
        height=400
    )

    fig.show()
    
    
def merge_files(df1, df2, ID):
    # Ensure the ID column exists in both DataFrames
    if ID not in df1.columns:
        raise KeyError(f"'{ID}' column not found in df1")
    if ID not in df2.columns:
        raise KeyError(f"'{ID}' column not found in df2")

    # Ensure the ID column is of the same type in both DataFrames
    df1[ID] = df1[ID].astype(str)
    df2[ID] = df2[ID].astype(str)

    # Sum ClaimAmount over identical IDs in df2
    df2 = df2.groupby(df2[ID])['ClaimAmount'].sum().reset_index()

    # Merge the two DataFrames
    df = pd.merge(df1, df2, on=ID, how='inner')

    # Fill missing ClaimAmount values with 0
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # Unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")

    return df

def plot_boxplots(features, y_feature, data, ymax=None):
    num_features = len(features)
    num_cols = 2  # Number of columns in the subplot grid
    num_rows = (num_features + num_cols - 1) // num_cols  # Calculate rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.boxplot(x=feature, y=y_feature, data=data, ax=ax)
        mean_value = data.groupby(feature)[y_feature].mean()
        for j, mean in enumerate(mean_value):
            ax.scatter(j, mean, color='green', label='Mean' if j == 0 else "", zorder=5)
        if i == 0:  # Add legend only to the first subplot
            ax.legend()
        ax.set_title(f'Boxplot of {y_feature} by {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel(y_feature)
        ax.tick_params(axis='x', rotation=45)

        # Calculate ymax if not provided
        if ymax is None:
            calculated_ymax = 2 * data.groupby(feature)[y_feature].mean().mean()
            ax.set_ylim(-.05, calculated_ymax)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
