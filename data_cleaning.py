# %% [markdown]
# # Data Cleaning for Lead Exposure Project

# %%
# Packages to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rapidfuzz import process,fuzz
from sklearn.neighbors import BallTree

# %% [markdown]
# ### Combine Service Line Inventory and Franklin County Parcel Data

# %%
# Import data files

# Documentation for the parcel data set is located at https://www.franklincountyauditor.com/AUDR-website/media/Documents/FTP/Parcel-CSV-description-of-fields.pdf
url_parcel = "https://apps.franklincountyauditor.com/Parcel_CSV/2025/05/Parcel.csv"
df_parcel = pd.read_csv(url_parcel)

# Service Line inventory data set
name_sl = "ServiceLineInventory.csv"
df_sl = pd.read_csv(name_sl)

# %%
# Create common Street Address column for both data sets
df_sl["Street Address"] = df_sl["Street Number"].astype(str) + ' ' + df_sl["Street Name"]

# Delete duplicate rows in each data set
df_parcel = df_parcel.drop_duplicates(subset=["STADDR"])
df_sl = df_sl.drop_duplicates(subset=["Street Address"])

# %%
# Find matching addresses in both data sets

common_values = set(df_sl['Street Address']) & set(df_parcel['STADDR'])
num_matches = len(common_values)
num_sl = len(df_sl)
num_parcel = len(df_parcel)

percent_match = num_matches/num_sl*100
percent_nomatch = 100-percent_match

# Data frame with exact matches for addresses
df_match_exact = pd.merge(df_sl, df_parcel, left_on="Street Address", right_on="STADDR", how="inner")

# %%
# Pull out all rows from Service Line Inventory without exact address matches
df_non_matches = df_sl[~df_sl["Street Address"].isin(df_parcel["STADDR"])]

# Consider a small sample of these rows
df_non_matches_sample = df_non_matches.sample(n=50)

def get_best_match(addr, choices, scorer=fuzz.token_sort_ratio):
    '''
    Use fuzzy logic package to match addresses. 
    addr: target address
    choices: range of possible addresses to match
    scorer: scoring method for fuzzy logic algorithm
    returns Pandas series containing the matched address and corresponding score
    '''

    results = process.extract(addr, choices, scorer=scorer,limit=5, score_cutoff=80)

    # If there are no results, return None
    if not results:
        return pd.Series([None, None])
    

     # See if address numbers are identical for any of the potential matches
    address_no = addr.split()[0]
    
    for match, score, _ in results:
        match_no = match.split()[0]
        if match_no == address_no:
            return pd.Series([match, score])
    
    # If no house number matches, return None
    return pd.Series([None, None])

        
## Get matches for sample dataset
"""
# Consider a small sample of these rows
df_non_matches_sample = df_non_matches.sample(n=50)

df_non_matches_sample[["Best Match", "Match Score"]] = df_non_matches_sample["Street Address"].apply(lambda addr: get_best_match(addr, df_parcel['STADDR']))
df_match_near = pd.merge(df_non_matches_sample, df_parcel, left_on="Best Match", right_on="STADDR", how="inner")
df_match_near = df_match_near.dropna(subset=['Best Match'])
"""

## Get matches for entire dataset. Uncomment to run code
"""
df_non_matches[["Best Match", "Match Score"]] = df_non_matches["Street Address"].apply(lambda addr: get_best_match(addr, df_parcel['STADDR']))
df_match_near = pd.merge(df_non_matches,df_parcel, left_on="Best Match", right_on="STADDR", how="inner")
df_match_near = df_match_near.dropna(subset=['Best Match'])

df_match_near.to_csv('data_match_near.csv')
"""

# %%
# Load near matches data
df_match_near = pd.read_csv('data_match_near.csv')

# Clean up the two dataframes so they can be combined
df_match_exact["Match Score"] = 100

columns_filtered = ['PARCEL ID', 'YEARBLT', 'APPRTOT', 'PRICE', 'GRADE',
                    'Street Number', 'Street Name', 'City', 'Zip Code', 'County', 'Latitude', 'Longitude', 
                    'System-Owned Portion Service Line Material Classification ',
                    'If Non-Lead in Column R.. Was Material Ever Previously Lead?',
                    'Customer-Owned Portion Service Line Material Classification']

df_match_exact_filtered = df_match_exact[columns_filtered]
df_match_near_filtered = df_match_near[columns_filtered]

# Dataframe with data for matched addresses in parcel and service line inventory
df_match_combined_filtered = pd.concat([df_match_exact_filtered,df_match_near_filtered])


# %%
# Check for null entries
df_match_combined_filtered.isnull().sum()

# %%
# Drop rows with null entries
df_match_combined_cleaned = df_match_combined_filtered.dropna()
df_match_combined_cleaned.info()

# %%
# Summarize address matching 
common_values = set(df_sl['Street Address']) & set(df_parcel['STADDR'])
num_matches = len(common_values)
num_sl = len(df_sl)
num_parcel = len(df_parcel)

num_near_matches = len(df_match_near)

percent_match = num_matches/num_sl*100
percent_near_match = num_near_matches/num_sl*100
percent_nomatch = 100-percent_match - percent_near_match

num_rows_final = len(df_match_combined_cleaned)
percent_final = num_rows_final/num_sl*100

print(f"Number of unique properties in parcel registry: {num_parcel}")
print(f"Number of unique properties listed in service line inventory: {num_sl}")
print(f"Number of exactly matching addresses: {num_matches} ({percent_match:0.1f}%)")
print(f"Number of nearly matching addresses: {num_near_matches} ({percent_near_match:0.1f}%)")
print(f"Number of non-matching addresses: {num_sl-num_matches} ({percent_nomatch:0.1f}%) \n")

print(f"Total number of properties with matching addresses and non-null entries:")
print(f"{num_rows_final} ({percent_final:0.1f}%)")      



# %%
# Summarize stats on lead vs non-lead customer service line material
lead_counts_customer = df_match_combined_cleaned["Customer-Owned Portion Service Line Material Classification"].eq('PB').sum()
gal_counts_customer = df_match_combined_cleaned["Customer-Owned Portion Service Line Material Classification"].eq('GAL').sum()
nonlead_counts_customer = df_match_combined_cleaned["Customer-Owned Portion Service Line Material Classification"].isin(['CU','OT','PL']).sum()
unknown_counts_customer = df_match_combined_cleaned["Customer-Owned Portion Service Line Material Classification"].eq('MU').sum()

print("Customer Service Line Summary")
print(f"Properties with lead pipes: {lead_counts_customer}")
print(f"Properties with galvanized pipes: {gal_counts_customer}")
print(f"Properties with non-lead pipes: {nonlead_counts_customer}")
print(f"Properties with galvanized pipes: {gal_counts_customer}")
print(f"Properties with unknown service line material: {unknown_counts_customer}")

# Note that since there are no properties listed as having lead pipes, we cannot create a model for the customer-owned service line material

# %%
# Summarize stats on lead vs non-lead city service line material
lead_counts_city = df_match_combined_cleaned["System-Owned Portion Service Line Material Classification "].eq('PB').sum()
gal_counts_city = df_match_combined_cleaned["System-Owned Portion Service Line Material Classification "].eq('GAL').sum()
nonlead_counts_city = df_match_combined_cleaned["System-Owned Portion Service Line Material Classification "].isin(['CU','OT','PL']).sum()
unknown_counts_city = df_match_combined_cleaned["System-Owned Portion Service Line Material Classification "].eq('MU').sum()

p_lead = lead_counts_city/(lead_counts_city+nonlead_counts_city+gal_counts_city)

print("City Service Line Summary")
print(f"Properties with lead pipes: {lead_counts_city}")
print(f"Properties with galvanized pipes: {gal_counts_city}")
print(f"Properties with non-lead pipes: {nonlead_counts_city}")
print(f"Properties with galvanized pipes: {gal_counts_city}")
print(f"Properties with unknown service line material: {unknown_counts_city}")

# %% [markdown]
# ### Remove all data not used for plotting or model building

# %%
# Clip all data before 1920. Years listed before this are not reliable, according to Franklin County Auditor's Website
df_match_combined_cleaned["YEARBLT"] = df_match_combined_cleaned["YEARBLT"].clip(lower=1920) 

# Remove all rows with unknown service line material. These should not contribute to the model
df_match_combined_cleaned = df_match_combined_cleaned[df_match_combined_cleaned["System-Owned Portion Service Line Material Classification "] != "MU"]

# %%
# Remove columns that do not contribute to the model or spatial plotting
columns_filtered_2 = ['PARCEL ID', 'YEARBLT',
                        'Street Number', 'Street Name', 'City', 'Zip Code', 'County', 'Latitude', 'Longitude', 
                        'System-Owned Portion Service Line Material Classification ']

df_match_combined_cleaned = df_match_combined_cleaned[columns_filtered_2]

# %%
df_match_combined_cleaned.info()

# %%
df_match_combined_cleaned["System-Owned Portion Service Line Material Classification "].value_counts()

# %% [markdown]
# ### Encode categorical data for binary classification

# %%
df_match_combined_cleaned["is_lead"] = (df_match_combined_cleaned["System-Owned Portion Service Line Material Classification "]=="PB").astype(int)

# Use +/- 1 for lead/non-lead values. Makes nearest-neighbor analysis clearer
df_match_combined_cleaned['is_lead_pm'] = df_match_combined_cleaned['is_lead'].replace(0,-1)

## Additional code for predicting galvanized pipe presence
#df_match_combined_cleaned["is_galv"] = (df_match_combined_cleaned["System-Owned Portion Service Line Material Classification "]=="GAL").astype(int)
#df_match_combined_cleaned["needs_replaced"] = df_match_combined_cleaned["is_lead"] + df_match_combined_cleaned["is_galv"]

# %% [markdown]
# #### Use k-nearest neighbors to encode nearest neighbor (nn) is_lead values and associated distances

# %%
# Convert degrees to radians
coords_rad = np.radians(df_match_combined_cleaned[['Latitude', 'Longitude']].values)

# Build BallTree with haversine metric
tree = BallTree(coords_rad, metric='haversine')

# Query for k=2 (skip the point itself)
dists, indices = tree.query(coords_rad, k=2)

# Multiply by Earth radius (in km)
earth_radius_km = 6371.0
nearest_dists_km = dists[:, 1] * earth_radius_km
nearest_indices = indices[:, 1]
nearest_is_lead = df_match_combined_cleaned.iloc[nearest_indices]['is_lead_pm'].values

# Add to DataFrame
df_match_combined_cleaned['nn_is_lead'] = nearest_is_lead
df_match_combined_cleaned['nn_distance_km'] = nearest_dists_km

# Create weighted nearest neighbor lead value
epsilon = 1e-3  # to avoid division by zero. Corresponds to 1 meter 
df_match_combined_cleaned['nn_is_lead_weighted'] = df_match_combined_cleaned['nn_is_lead'] / (df_match_combined_cleaned['nn_distance_km'] + epsilon)

# %%
# Generate stats on nearest neighbor distance
df_match_combined_cleaned['nn_distance_km'].describe()

# %%
# take a look at the nearest neighbor distance data
df_match_combined_cleaned.hist(column='nn_distance_km', bins=51)

# %%
# Describe weighted nn lead values
df_match_combined_cleaned['nn_is_lead_weighted'].describe()

# %%
# Describe absolute value of nn lead values
abs(df_match_combined_cleaned['nn_is_lead_weighted']).describe()

# %%
df_match_combined_cleaned.hist(column='nn_is_lead_weighted', bins=51)

# %%
# Save cleaned dataframe as csv.
df_match_combined_cleaned.to_csv('df_cleaned.csv')

# %%
df_match_combined_cleaned.sample(n=10)

# %%



