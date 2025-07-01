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
Many Americans take for granted the fact that they have access to safe and clean drinking water within the comfort of their own homes. Yet in recent years, health concerns have brought the issue of water contamination back into the public consciousness, with the most high profile example being the water emergency encountered by residents in Flint, Michigan. Starting in 2014 in order to minimze costs to the city, the water supply was changed to the Flint river. Although not publicly announced at the time of the switch, residents almost immediately noticed a drastic change in their water quality. After years of complaints, state & federal investigations, and numerous lawsuits it was eventually determined that the water of Flint river had a corrosive effect on the outdated service lines within the city. Those service lines were largely constructed out of lead, which leached out into the water supply due to the caustic nature of the river's water quality. Although minimum testing levels are set by the Enviromental Protection Agency, it is widely acknowledged that there is **no safe level for lead exposure** as the effects from lead exposure are chronic and irreversible. This is especially true for young adolescencents, as lead exposure is directly correlated to stunted development and cognitive impairment for the remainder of their lives. 

Fortunately, the EPA has set new regulation guidelines which require all lead service lines to be replaced by 2035, and the state of Ohio has a new proposed bill that seeks to phase out all lead water lines within the next 15 years. The city of Columbus has already taken a proactive role on this front by cataloging the materials of service lines throughout the city. However, this process is costly and time intensive, and therefore many homes and commerical properties are unaccounted for. Our group hoped to retify this problem by creating a mathematical model which could predict the likelihood that a certain property will be serviced by lead pipes. In doing so we hope to not only extend the scope of the city's service line database, but we aimed to provide key takeaways that city officals could use to prioritize resources for it's [Lead Service Line Replacement](https://www.columbus.gov/Services/Columbus-Water-Power/About-Columbus-Water-Power/The-Division-of-Water/Water-Facts/Water-Health/Lead-Service-Program-Information) program.

## Dataset Creation
The city of Columbus has a [Service Line Material Inventory](https://experience.arcgis.com/experience/1ddfc9ee51ae4eddbdf8003c81eef7e4/) which describes the construction material of pipes which provide water to individual buildings. After contacting city officals we obtained access to the full database which contains roughly 288,000 properties. The information was then cleaned to create `ServiceLineInventory.csv`, so that only pertinent informartion, such as addresses, lat & long. data, and pipe material classifications, were included. We then crossreferenced the addresses of each building of this dataset with detailed home information (e.g. Build year, property value, etc.) provided by the [Franklin County Auditor's Office](https://www.franklincountyauditor.com/home). This task ended up being more challenging than initially anticipated, as the Address labels did not always perfectly align (e.g. "48 West Duncan Street" versus "48 W Duncan St"). To solve this problem, 'fuzzy logic' tools were employed to assign a match likelihood between every address in each dataset (where the previous example might yield 80%, and a perfect match would yield 100%). We also realized during this time that the Franklin County dataset contained nearly twice as many entries as the Service Line Inventory, again highlighting the need for a more complete database of water service lines. After the fuzzy matching process was completed, we had a final dataset (`df_cleaned.csv`) consisting of approximately 224,000 unique properties.

In order to demarcate the geographical regions of the Columbus metro area, we used the city's [offical planning area boundaries](https://opendata.columbus.gov/datasets/00b5b47799d546efb13eddee7dad52b5_16/explore) which consists of 27 "neighborhoods". The python package `geopandas` was used to assign each property in `df_cleaned.csv` to a neighborhood region within the greater Columbus area. This enabled us to calculate statistics for not only the entire city, but distinct spatial regions as well.

## Preprocessing and Exploratory Analysis
Roughly 10% of homes in the service line inventory are serviced by lead or galvinized steel pipes (which lead particles can stick to). One consequence of this fact, is that a typical train-test split will not correctly handle the unbalanced data. These errors are largely caused by the fact that our model might accidentally be trained (or tested) on virtually all non-lead data points during random sampling. For all the models being considered in this project, we rectified the problem of unbalanced data by performing *Stratified* K-Fold Cross-Validation. This method works by requiring that 10% percent of the randomly sampled data points are buildings which test positive for lead, thereby migigating the issue of oversampling non-lead data points.

As a starting point, we calculated a histogram of all the homes built in Columbus between 1803 - 2024 (*note that home build years only became offically reliable after 1920). We did this because we expected a strong correlation between construction year and the prevalence of lead pipes. We also researched historical state records and found that Copper overtook lead in 1930 as the choice material for water pipes, as well as an Ohio [law](https://codes.ohio.gov/ohio-administrative-code/rule-3745-81-84) banning the use of lead pipes for new construction projects starting in 1963. This rule also included a two year grace period for compliance. Likewise, similar rules were enacted by the Environmental Protection Agency in in 1986 and [1991](https://www.epa.gov/dwreginfo/lead-and-copper-rule) which placed minimum lead level standards across the country. These rules gave the impression that 1930, 1963-1965, and 1991 would be key inflection years.

Our preliminary results verified this hypothesis, as homes constructed between 1900-1950 were found to have 50% chance of containing lead pipes if picked by random, while homes between 1960-1990 where found to have virtually 0% chance of having lead pipes when randomly selected.

Stratified K-fold Cross Validation: Allows us to partition the majority and minority data sets in a controlled fashion, such that our results are comparable to different folds of traing and testing. For example, we randomly sampled 225,000 homes to create an 80:20 training testing split, we would naively expect 10\% of those homes to have lead pipes in both the training \emph{and} testing data sets. However, through random chance, some folds may overestimate or underestimate the number of homes that contain lead. To ensure this does not happen, we created a constrained training set of 179,200 homes (80\% of the total homes) where 10% of that 179,200 homes training set was garunteed to be lead positive homes. Unique addresses are shuffled randomly fold-to-fold, but the overall ratios remain fixed.

## Model Selection

Correlation between Home Age adn Lead: Simple and easily interpretable model, which can be used to verify the correlation between build year and likelihood of lead. Assumes a direct correlation and likely does not capture the full picture behind the numerous driving factors for predicting lead likelihood.

Stratified K-fold Cross Validation: Allows us to partition the majority and minority data sets in a controlled fashion, such that our results are comparable to different folds of traing and testing. For example, we randomly sampled 225,000 homes to create an 80:20 training testing split, we would naively expect 10\% of those homes to have lead pipes in both the training \emph{and} testing data sets. However, through random chance, some folds may overestimate or underestimate the number of homes that contain lead. To ensure this does not happen, we created a constrained training set of 179,200 homes (80\% of the total homes) where 10% of that 179,200 homes training set was garunteed to be lead positive homes. Unique addresses are shuffled randomly fold-to-fold, but the overall ratios remain fixed.

Logistic Regression: Simplest regression model, where we only consider the build year and number of postive lead nearest neighbors. Because our dataset consits of ~224,000 unique data points, it is impossible (from a computational standpoint) to calculate all nearest-neighbors, so instead we derived an "effective radius/correlation length" that weights the contribution of a given postive-lead home's influence on the prediction of the predicted address. In other words, the closer the neighbors, the larger the influence those neighbors will have on overall the prediction.

KNN: A strictly nearest neighbor regression method, which isolates the contribution of knowledge about the overall neighborhood. Has the benefit of being easy to interpret and provides a nice way of analizing the spatial regions where nearest neighbors struggles to accuractely predict lead pipes within the city.

## Results
See a strong correlation between build year and likelihood of lead. Useful for extending the service line inventory's scope, but not necessarily the most accurate (see metrics).

Logistic regression works better than random guessing after hyperparameters have been optimized. Over a 50% increase in accuracy over the random guessing case. Interpretibilty is more challenging however.

KNN (nearest-neighbor) works well, but the prevalance of newer homes in traditionally older neighborhoods contaminates the model's ability to accurately predict true negatives in regions where there is a high number of lead-postive homes.

# Extensions for Future Work
While all our models drastically improve upon the simplest case of random guessing, they struggle with lead prediction in regions where there is little to no training data. One focus for the city would be to prioritize reporting in those regions, so the model's could be properly trained over a wider range of geographical neighborhoods. This would liekly help the KNN model the most, as logistic regression seems to be more robust due to the fact that it also considers correlations with build year. 

Our models only focused on the two primary influences for lead prediction, however other influences/correlations such as home appraisal price, or grade rating would likely refine the accuracy and precision of the above models throughout the entire Columbus region. 

Another interesting extension would be to transfer the model's prediction to newer cities that have a narrower range of build years, or to predict average student test scores for statewide assements.


## Files 

### Notebooks
◦ [`data_cleaning.ipynb`](data_cleaning.ipynb)\
◦ [`JoiningData.ipynb`](JoiningData.ipynb)\
◦ [`preliminary_analysis.ipynb`](preliminary_analysis.ipynb)\
◦ [`LogisticRegression.ipynb`](LogisticRegression.ipynb)\
◦ [`KNN_Analysis.ipynb`](KNN_Analysis.ipynb)\
◦ [`Columbus_CommunityPLanningDistricts.ipynb`](Columbus_CommunityPlanningDistricts.ipynb)\
◦ [`Plotly_Chloropleth.ipynb`](Plotly_Chloropleth.ipynb)
### Python Scripts
◦ [`preliminary_analysis.py`](preliminary_analysis.py)\
◦ [`LogisticRegression.py`](LogisticRegression.py)\
◦ [`KNN_Analysis.py`](KNN_Analysis.py)
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

