# Does your Home Contain Lead? Predicting Lead Pipe Exposure for Homes in Columbus, OH 

### Team Members: [Katherine Laliotis](https://github.com/kklaliotis), [Alex Schimmoller](https://github.com/aschimmo), and [Brock Grafstrom](https://github.com/brockgrafstrom)

## Table of Contents
- [Introduction](#introduction)
- [Dataset Creation](#dataset-creation)
- [Preprocessing and Exploratory Analysis](#preprocessing-and-exploratory-analysis)
- [Model Selection](#model-selection)
- [Results](#results)
- [Files](#files)
- [Software Requirements](#software-requirements)
## Introduction
Many Americans take for granted the fact that they have access to safe and clean drinking water within the comfort of their own homes. Yet in recent years, public health concerns have brought the issue of water contamination back into the public consciousness, with the most high profile example being the water emergency encountered by residents in Flint, Michigan. Starting in 2014 in order to minimze costs to the city, the water supply was changed to the Flint river. Although not publicly announced at the time of the switch, residents almost immediately noticed a drastic change in their water quality. After years of complaints, state & federal investigations, and numerous lawsuits it was eventually determined that the water of Flint river had a corrosive effect on the outdated service lines within the city. Those service lines were largely constructed out of lead, which leached out into the water supply due to the caustic nature of the river's water quality. Although minimum testing levels are set by the Enviromental Protection Agency, it is widely acknowledged that there is **no safe level for lead exposure** as the effects from lead exposure are chronic and irreversible. This is especially true for young adolescencents, as lead exposure is directly correlated to stunted development and cognitive impairment for the remainder of their lives. 

Fortunately, the EPA has set new regulation guidelines which require all lead service lines to be replaced by 2035, and the state of Ohio has a new proposed bill that seeks to phase out all lead water lines within the next 15 years. The city of Columbus has already taken a proactive role on this front by cataloging the materials of service lines throughout the city. However, this process is costly and time intensive, and therefore many homes and commerical properties are unaccounted for. Our group hoped to retify this problem by creating a mathematical model which could predict the likelihood that a certain property will be serviced by lead pipes. In doing so we hope to not only extend the scope of the city's service line database, but we aimed to provide key takeaways that city officals could use to prioritize resources for it's [Lead Service Line Replacement](https://www.columbus.gov/Services/Columbus-Water-Power/About-Columbus-Water-Power/The-Division-of-Water/Water-Facts/Water-Health/Lead-Service-Program-Information) program.

## Dataset Creation
The city of Columbus has a [Service Line Material Inventory](https://experience.arcgis.com/experience/1ddfc9ee51ae4eddbdf8003c81eef7e4/) which describes the construction material of pipes which provide water to individual buildings. After contacting city officals we obtained access to the full database which contains data on roughly 288,000 properties. The information was then cleaned to create `ServiceLineInventory.csv`, which contains addresses, lat & long. data, and pipe material classifications. We then crossreferenced the addresses of each building of this dataset with detailed home information (e.g. Build year, property value, etc.) provided by the [Franklin County Auditor's Office](https://www.franklincountyauditor.com/home). This task ended up being more challenging than initially anticipated, as the Address labels did not always perfectly match (e.g. "48 West Duncan Street" versus "48 W Duncan St"). To solve this problem, 'fuzzy logic' tools were employed to assign a match likelihood between every address in each dataset (where the previous example might yield 80%, and a perfect match would yield 100%). We also realized during this time that the Franklin County dataset contained nearly twice as many entries as the Service Line Inventory, again highlighting the need for a more complete database of water service lines. After the fuzzy matching process was completed, we had a final dataset (`df_cleaned.csv`) consisting of ~224,000 unique properties.

In order to demarcate the geographical regions of the Columbus metro area, we used the city's [offical planning area boundaries](https://opendata.columbus.gov/datasets/00b5b47799d546efb13eddee7dad52b5_16/explore) which consists of 27 "neighborhoods". The python package `geopandas` was used to assign each property in `df_cleaned.csv` to a neighborhood region within the greater Columbus area. This enabled us to calculate statistics for not only the entire city, but distinct spatial regions as well.

## Preprocessing and Exploratory Analysis

Roughly 10% of homes in the service line inventory are serviced by lead or galvinized steel pipes (which lead particles can stick to). This 
Timeseries analysis: 

## Model Selection

\item Time series analysis of build year and lead pipes followed by logistic regression: Our data contains homes that were constructed in every year between 2024 and the late-1800s. Preliminary analysis indicates that homes constructed between 1900-1950 have the greatest likelihood of containing lead, with a drastic decrease after 1950. Cross-referencing with Ohio and federal EPA statues we note the following key years...
        \begin{itemize}
            \item 1930s - Copper pipes overtake lead for new construction projects.
            \item 1963 - Ohio passes \href{https://codes.ohio.gov/ohio-administrative-code/rule-3745-81-84}{Administrative Rule 3745-81-84} banning lead for use in the construction.
            \item 1965 - Last year Ohio law allows new construction projects to use lead. 
            \item 1986 - EPA bans lead pipes for use in new construction.
            \item 1991 - EPA enacts \href{https://www.epa.gov/dwreginfo/lead-and-copper-rule}{Lead \& Copper Rule} which requires the removal and replacement of lead lines regardless of build year. Also set stricter water monitoring regulations (drinking water must have less than 15ppb of lead)
            \item 1996 - The city of Columbus starts removing lead service lines.
            \item 2024 - EPA passes final revision of Lead and Copper Rule which mandates that all lead service lines in the US must be replaced by 2034.
            \item 2025 - Ohio state legislature proposes \href{https://greatlakes.org/2025/05/environmental-organizations-celebrate-bipartisan-bill-to-remove-ohios-lead-service-lines-to-improve-drinking-water/}{Lead Line Replacement Act} which mandates that all lead service lines must be replaced within 15 years.
        \end{itemize}
    
        \item \href{https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html}{Stratified K-fold Cross Validation}: Allows us to partition the majority and minority data sets in a balanced fashion. For example, we randomly sample 225,000 homes to create an 80:20 training testing split, we would naively expect 10\% of those homes to have lead pipes in both the training \emph{and} testing data sets. However, through random chance, some folds may overestimate or underestimate the number of homes that contain lead. To ensure this does not happen, we will create a constrained training set of 179,200 homes (80\% of the total homes) with the following breakdown...

Stratified K-fold Cross-Validation:

Tree algorithm: Can be used to aid in pipe material classification, and also allows us to assign different weights for classification. Hyperparameter optimization of these classification weights might aid in mitigating systematic biases in our data.

## Results

## Files 

### Notebooks
◦ [`data_cleaning.ipynb`](data_cleaning.ipynb)\
◦ [`JoiningData.ipynb`](JoiningData.ipynb)\
◦ [`preliminary_analysis.ipynb`](preliminary_analysis.ipynb)\
◦ [`LogisticRegression.ipynb`](LogisticRegression.ipynb)\
◦ [`Neighbors_Trees.ipynb`](Neighbors_Trees.ipynb)\
◦ [`Columbus_CommunityPLanningDistricts.ipynb`](Columbus_CommunityPLanningDistricts.ipynb)\
◦ [`Plotly_Chloropleth.ipynb`](Plotly_Chloropleth.ipynb)
### Python Scripts
◦ [`preliminary_analysis.py`](preliminary_analysis.py)
### CSV
◦ [`ServiceLineInventory.csv`](ServiceLineInventory.csv) Raw service line inventory data and [`ServiceLineKey.txt`](ServiceLineKey.txt)\
◦ [`data_match_near.csv`](data_match_near.csv) Connecting the Service Line Inventory to the Franklin County Auditor's Data\
◦ [`df_cleaned.csv`](df_cleaned.csv) Final dataset used for training and testing

## Software Requirements
All of the above files ultilize python (version 3.12.10) and the following packages...

<ins>Basic Data Manipulation</ins>$~~~~~~$<ins>Geospatial Data Manipulation</ins>$~~~~~~$<ins>Static \& Interactive Plotting</ins>$~~~~~~$<ins>Model Training</ins>
1. `numpy`     $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$    3. `geopandas` $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ 5. `plotly.express`  $~~~~~~~~~~~~~~~~~~~~~~$ 10. `sklearn`
2. `pandas`     $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$     4. `pillow` $~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ 6. `plotly.graph_objects`\
$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ 7. `matplotlib.pyplot`\
$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ 8. `seaborn`\
$~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$ 9. `contextily`

