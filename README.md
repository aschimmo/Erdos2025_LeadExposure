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

## Dataset Creation

The city of Columbus has a [Service Line Material Inventory](https://experience.arcgis.com/experience/1ddfc9ee51ae4eddbdf8003c81eef7e4/) which describes the construction material of pipes which provide water to individual buildings. It contains data on roughly 288,000 properties, and yet there are still ~ 250,000 properties unaccounted for. 

Detailed home information (e.g. Build year, property value, etc.) was provided by the [Franklin Country Auditor's Office](https://www.franklincountyauditor.com/home).

In order to demarcate the geographical regions of the Columbus metro area, we used the city's [offical planning area boundaries](https://opendata.columbus.gov/datasets/00b5b47799d546efb13eddee7dad52b5_16/explore). By using the 
## Preprocessing and Exploratory Analysis

## Model Selection

## Results

## Files 

## Software Requirements
All of the above files listed ultilize python (version 3.12.10) and the following packages...

<ins>Basic Data Manipulation</ins>$~~~~~~$<ins>Geospatial Data Manipulation</ins>
1. `numpy`      $~~~~~~~~~~~~$          3. `geopandas`
2. `pandas`     $~~~~~~~~~~~~$          4. `pillow`

<ins>Geospatial Data Manipulation</ins>
3. `geopandas`
4. `pillow`

<ins>Plotting</ins>
5. `plotly.express`
6. plotly.graph_objects
7. `matplotlib.pyplot`
8. `seaborn`
9. `contextily`

<ins>Model Training</ins>
10. `sklearn`
