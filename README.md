# Water-Demand-Forecasting Characterizing demand non-stationarity in the water system.

Abstract: The assumption of per-capita demand stationarity plagues municipal water systems.
Identifying sources of uncertainty in projecting water demand and representing their influence in quantitative estimates will help improve water system planning and operations. 
To address these needs, this paper has two objectives A) characterize the impacts of demand stationarity on forecast accuracy and B) develop a coherent machine learning framework to improve the accuracy of seasonal demand estimates.
Using the Salt Lake City Department of Public Utilities service area, we benchmark stationary demand forecasting performance and limitations with observed demands from historical periods representing drought, average, and surplus supply conditions.
Using an ensemble of ML tools, the ridge regression-based Salt Lake City Water Demand Model (SLC-WDM) mitigates the stationary limitations to produce climate-driven season demand forecasts, expected to be more accurate.
Performance evaluation of the SLC-WDM forecasts compared with multi-layered perceptron (MLP) and Random Forest regression (RFR) models indicates performance-interpretability trade-offs.
We find the stationary assumption leads to over-estimates of monthly drought-condition demands up to 9\% and seasonally up to 40\%, a result of an extended irrigation season and regional drought awareness.
In these conditions, SLC-WDM forecasts are within 7\%  and 0.2\% of the observed monthly and season values, respectively.
The SLC-WDM also minimized error in all supply scenarios, exceeding those from the stationary,  MLP, and RFR forecasts (\textit{MAPE} = 8.4\%, 31.0\%, 9.0\%, and 10.5\%, respectively).
As a result, we demonstrate stationarity is a critical limitation in demand forecasting and a coherent machine learning framework can capture climate-driven demand non-stationarity.


This repository contains the data (in the Data branch) and modeling scripts (also containing data analysis and figure development) for the Salt Lake City Water Demand Model (SLC-WDM), multi-layered perceptron (MLP), and Random Forest (RFR).
Each of these scripts also contains monthly stationary methods for comparision of forecasting performance and analysis.
