#Author: Ryan C. Johnson PhD
#Date: 6/10/2022


'''
We want to see the trends for the training data,# and will want to have them modeled into the future based
on consevation goals

From plots, There appears to be very little trends in training data. The last 10 years do show decreasing trends
likely due to conservation goals estabished in 2005


Thinking of a stepwise process:
1) identify any trends (training and testing): check, no trends in training. however in 2000, Utah established a 
   conservation goal. To reduce the gpcd water use by 25% by 2025. Evaluating water use from 2005-2017 shows a decline
   in water use.
2) Determine conservation goals (annual reduction /yr, slope), over 25yrs a 75 gpcd reduction is requested (3gpcd/yr), check
3) Separately model indoor and outdoor demands per year. check 
4) Training Outdoor demand (Apr-Oct) will be that month's total demand minus that years Jan-Mar mean indoor demand
5) Final model = modeled indoor demand - conservation reduction + Outdoor demand (if Apr-Oct).
'''



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn. metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_percentage_error
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn import preprocessing
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import seaborn as sns; sns.set()
import joblib
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression
from pathlib import Path
import copy
import pickle

#Trying out recusive feature elimination to compare with step wise regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import sklearn

import statsmodels.api as sm
from time import strptime
import datetime
import calendar
from calendar import monthrange
from progressbar import ProgressBar
import warnings
warnings.filterwarnings('ignore')


#ignoring unnecessary warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
np.warnings.filterwarnings('ignore', category=UserWarning)
from pandas.core.common import SettingWithCopyWarning
np.warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

plt.rcParams["axes.grid"] = False
plt.rcParams["axes.facecolor"] ='white'



class CSDWDM():
    
    def __init__(self, cwd):
        self = self
        self.cwd = cwd
        
    def SLC_Data_Processing(self, slc, snow, O_cons, I_cons, time):

        #rename the gpcd column
        for i in slc:
            slc[i].rename(columns={i+'_gpcd': 'Obs_gpcd'}, inplace=True)
            slc[i]=slc[i].set_index('Year')
        snow=snow.set_index('Year')
        #need to remove certain features

        colrem= ['Dem_AF', 'seven', 'meantemp_days', 'maxtemp_days', 'mean_max', 'mill', 'precip_days', 
                 'Days_abovemax','Days_abovemean', 'red' , 'emig', 'sqmi','max_Days_WO',
                 'mtn','ResHouseDensity', 'Urban_Area_Perc','Residential_Area_Perc', 'IrrPopulationDensity',
                 'Irrigated_Area_Perc','CityCrk_AcFt_WR_Mar', 'LitCotCrk_AcFt_WR_Jun']#, 'AcFt', 'WO', 'days', 'days', 'above' , 'Perc']


        for i in slc:
            for j in colrem:
                slc[i]=slc[i].loc[:,~slc[i].columns.str.contains(j , case=False)] 

        #Create training and testing data, use most recent low, average, and high water years
        self.slc_train=copy.deepcopy(slc)
        self.slc_test=copy.deepcopy(slc)

        #2008 is a high year
        #2011 and 2017 are average years
        #2014 and 2016 are below average years
        #2015 is a very  low year

        IN_WY_Months = ['Jan' , 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sep', 'Oct']
        Prior_YR_WY_Months = ['Nov', 'Dec']
        testWYyrs = [2008,2015,2017]


        for i in slc:
            #Select the training/testing dataframes
            self.slc_train[i]=self.slc_train[i][~self.slc_train[i].index.isin(testWYyrs)]
            self.slc_test[i]=self.slc_test[i][self.slc_test[i].index.isin(testWYyrs)]


        #Determine the indoor mean to subtract from outdoor
        self.I_mean_train=(self.slc_train['Jan']['Obs_gpcd']+
                      self.slc_train['Feb']['Obs_gpcd']+
                      self.slc_train['Mar']['Obs_gpcd']+
                      self.slc_train['Nov']['Obs_gpcd']+
                     self.slc_train['Dec']['Obs_gpcd'])/5
        self.I_mean_test=(self.slc_test['Jan']['Obs_gpcd']+
                     self.slc_test['Feb']['Obs_gpcd']+
                     self.slc_test['Mar']['Obs_gpcd']+
                     self.slc_test['Nov']['Obs_gpcd']+
                     self.slc_test['Dec']['Obs_gpcd'])/5

        for i in self.slc_train:
            self.slc_train[i]['Iave']=self.I_mean_train
            #for now include testing years
            self.slc_test[i]['Iave']=self.I_mean_test

        IrrSeason= ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
        Indoor=['Jan', 'Feb', 'Mar', 'Nov', 'Dec']
        colrem=['Iave', 'Obs_gpcd']
        #set the target gpcd to indoor for indoor and total-indoor for outdoor
        #change to indoor to separate outdoor demands from total
        for i in Indoor:
            self.slc_train[i]['Target_gpcd']=self.slc_train[i]['Obs_gpcd']
            self.slc_train[i]= self.slc_train[i].drop(columns=colrem)
            #for now include testing years
            self.slc_test[i]['Target_gpcd']=self.slc_test[i]['Obs_gpcd']
            self.slc_test[i]= self.slc_test[i].drop(columns=colrem)

        for i in IrrSeason:
            self.slc_train[i]['Target_gpcd']=self.slc_train[i]['Obs_gpcd']-self.slc_train[i]['Iave']
            self.slc_train[i].loc[self.slc_train[i]['Target_gpcd'] < 0, 'Target_gpcd'] = 0

            #add in snow info
            self.slc_train[i]=pd.concat([self.slc_train[i], snow], axis=1, join="inner")

            self.slc_train[i]= self.slc_train[i].drop(columns=colrem)
            #for now include testing years
            self.slc_test[i]=pd.concat([self.slc_test[i], snow], axis=1, join="inner")
            self.slc_test[i]['Target_gpcd']=self.slc_test[i]['Obs_gpcd']-self.slc_test[i]['Iave']
            self.slc_test[i].loc[self.slc_test[i]['Target_gpcd'] < 0, 'Target_gpcd'] = 0

            #create monthly historical mean and conservation trends
            Out_mean = np.mean(self.slc_train[i]['Target_gpcd'].loc[2000:])
            goal = (1-O_cons)*Out_mean
            O_cons_rate = (Out_mean -goal)/time



            self.slc_train[i]['cons_goal'] = Out_mean- ((self.slc_train[i].index-2000)*O_cons_rate)
            self.slc_train[i].loc[ self.slc_train[i].index <2000, ['cons_goal']] = Out_mean

            t=self.slc_train[i]['Target_gpcd'].copy()
            c=self.slc_train[i]['cons_goal'].copy()
            self.slc_train[i] = self.slc_train[i].drop(columns=['Target_gpcd', 'cons_goal'])
            self.slc_train[i]['Target_gpcd'] = t
            self.slc_train[i]['cons_goal'] = c

            self.slc_test[i]['cons_goal'] = Out_mean - ((self.slc_test[i].index-2000)*O_cons_rate)
            self.slc_test[i]= self.slc_test[i].drop(columns=colrem)

            #Determine the historical indoor mean to apply conservation strategies too
        Indmean = np.mean(self.slc_train['Jan']['Target_gpcd'].loc[2000:]+
                          self.slc_train['Feb']['Target_gpcd'].loc[2000:]+
                          self.slc_train['Mar']['Target_gpcd'].loc[2000:]+
                          self.slc_train['Nov']['Target_gpcd'].loc[2000:]+
                         self.slc_train['Dec']['Target_gpcd'].loc[2000:])/5

        goal = (1-I_cons)*Indmean



        cons_rate = (Indmean -goal)/time


        #create feature called cons_goal!
        for i in Indoor:
            self.slc_test[i]['cons_goal'] = Indmean-((self.slc_test[i].index-2000)*cons_rate) 
            self.slc_train[i]['cons_goal'] = Indmean-((self.slc_train[i].index-2000)*cons_rate) 

            self.slc_train[i].loc[self.slc_train[i].index <2000, ['cons_goal']] = Indmean


        self.Cons_mean_test=(self.slc_test['Jan']['cons_goal']+
                        self.slc_test['Feb']['cons_goal']+
                        self.slc_test['Mar']['cons_goal']+
                        self.slc_test['Nov']['cons_goal']+
                        self.slc_test['Dec']['cons_goal'])/5

        #split training and testing data into features and targets
        self.slc_train_target=copy.deepcopy(self.slc_train)
        self.slc_train_features=copy.deepcopy(self.slc_train)

        self.slc_test_target=copy.deepcopy(self.slc_test)
        self.slc_test_features=copy.deepcopy(self.slc_test)


        target=['Target_gpcd','Housing']
        for i in self.slc_train_target:
            self.slc_train_target[i]=self.slc_train_target[i]['Target_gpcd']
            #for now include testing years
            self.slc_test_target[i]=self.slc_test_target[i]['Target_gpcd']


            self.slc_train_features[i]= self.slc_train_features[i].drop(columns=target)
            #for now include testing years
            self.slc_test_features[i]= self.slc_test_features[i].drop(columns=target)

        #need to remove year from the list to run plots below
        for i in self.slc_train:
            self.slc_train[i]=self.slc_train[i].drop(columns=['Housing',  'Population', 'PopulationDensity'])


        
        
    def CSD_WDM_Train(self, p_space, Outdoor_Months, IndoorMonths, scoring):
        self.slc_val=copy.deepcopy(self.slc_test)
        # calibrate and predict with the outdoor model
        pbar = ProgressBar()
        for i in pbar(Outdoor_Months):
            print('The model is automatically selecting features and calibrating the ', i, 'outdoor demand model.' )
            #put the month, use conservation_goal (-1: no, -2: yes) correlation threshold, colineariy threshold, CV, aplpha, model type, tuning method
            #put in the params, month, scoring method (R2, or RMSE for now)
            PerfDF, cv_results, cor, X_test_RFE, coef = self.Demand_Optimization(p_space, i, scoring)

            colrem = self.slc_test[i].columns
            self.slc_val[i] = self.slc_val[i].reset_index(drop=True)
            self.slc_val[i] = pd.concat([self.slc_val[i], PerfDF], axis=1, join="inner")
            self.slc_val[i] = self.slc_val[i].set_index('Date')
            self.slc_val[i] = self.slc_val[i].drop(columns=colrem)

        
        
        

    def Outdoor_Demand_Model(self, TrainDF, month, X_train_features, y_train_target, X_test_features, y_test_target,
                              snowfeatures, conservation, cor_threshold, colinearity_thresh, cv_splits,
                             model_type, scoring ):


    #subset these features out of main DF and put into cute heatmap plot

        DFcor = copy.deepcopy(TrainDF[month])

        #if snowfeatures is True:
         #   print('LCC Snowfeatures are being used')
        Indoor=['Jan', 'Feb', 'Mar', 'Nov', 'Dec']
        if snowfeatures is False:
            if month in Indoor:
                DFcor=DFcor
            else:
                snow=['Nov_snow_in','Dec_snow_in', 'Jan_snow_in','Feb_snow_in', 
                        'Mar_snow_in','Apr_snow_in', 'Total_snow_in', 'Snow_shortage']
                DFcor=DFcor.drop(columns=snow)


        cor=DFcor.copy()
        if conservation is False:
            del cor['cons_goal']
            cor = cor.corr()
            cor =cor.iloc[:,-1:]
        if conservation is True:
            cor = cor.corr()
            cor =cor.iloc[:,-2:]
            del cor['cons_goal']

        cor['Target_gpcd']=np.abs(cor['Target_gpcd'])
        cor=cor.sort_values(by=['Target_gpcd'], ascending=False)
        cor=cor.dropna()

    #Selecting highly correlated features
        relevant_features = cor[cor['Target_gpcd']>cor_threshold]
        CorFeat = list(relevant_features.index)

        CorDF= DFcor[CorFeat]
        cor = np.abs(CorDF.corr())
        cor = cor.mask(np.tril(np.ones(cor.shape)).astype(np.bool))
        #remove colinearity
        cor = cor[cor.columns[cor.max() < colinearity_thresh]]
        CorFeat=cor.columns
        cor = cor.T
        cor = cor[CorFeat]

        #print('Remaining features are', CorFeat)


       #Set up training and testing data 
        X_train = X_train_features[month][CorFeat].copy()
    #X_train = slc_train_features['Jul'][JulF]
        y_train = y_train_target[month].copy()

        X_test = X_test_features[month][CorFeat].copy()
    #X_test = slc_test_features['Jul'][JulF]
        y_test = y_test_target[month].copy()

        # step-1: create a cross-validation scheme
        folds = KFold(n_splits = cv_splits, shuffle = True, random_state = 42)

    # step-2: specify range of hyperparameters to tune
        if len(CorFeat) > 1 :
            hyper_params = [{'n_features_to_select': list(range(1, len(CorFeat)))}]


    # step-3: perform grid search
    # 3.1 specify model, key to set intercept to false
            trainmodel = model_type
            trainmodel.fit(X_train, y_train)
            rfe = RFE(trainmodel)             

    # 3.2 call GridSearchCV()
            model_cv = GridSearchCV(estimator = rfe, 
                            param_grid = hyper_params, 
                            scoring= scoring, 
                            cv = folds, 
                            verbose = 0,
                            return_train_score=True)      

    # fit the model
            model_cv.fit(X_train, y_train)

    # create a KFold object with 5 splits 
            folds = KFold(n_splits = cv_splits, shuffle = True, random_state = 42)
            scores = cross_val_score(trainmodel, X_train, y_train, scoring=scoring, cv=folds)
           # print('CV scores = ', scores) 

    # cv results
            cv_results = pd.DataFrame(model_cv.cv_results_)


         #code to select features for final model, tell how many features
            N_feat=cv_results.loc[cv_results['mean_test_score'].idxmax()]
            N_feat=N_feat['param_n_features_to_select']
            #print('Number of features to select is ', N_feat)
        # intermediate model
            n_features_optimal = N_feat

            Int_model = model_type
            Int_model.fit(X_train, y_train)

            rfe = RFE(Int_model, n_features_to_select=n_features_optimal)             
            rfe = rfe.fit(X_train, y_train)

    #make the final model with rfe features

    # tuples of (feature name, whether selected, ranking)
    # note that the 'rank' is > 1 for non-selected features

            Features =list(zip(X_train.columns,rfe.support_,rfe.ranking_))
            FeaturesDF=pd.DataFrame(Features, columns=['Feature', 'Important', 'Score'])
            FeaturesDF = FeaturesDF[FeaturesDF.Score<=1]
            RFE_features = list(FeaturesDF['Feature'])
           # print('The final features are ', RFE_features)

            #select only RFE features for model training/validation
            X_train = X_train[RFE_features]
            X_test = X_test[RFE_features]


        #plot of selected features to make sure not colinear
            CorDF= X_train.copy()
            CorDF['Target_gpcd']=self.slc_train[month]['Target_gpcd']

            Final_model = model_type
            Final_model.fit(X_train, y_train)

            #grab uncertainty stats

          #  Uncertainty = sm.OLS(y_train, X_train).fit()
           # print(Uncertainty.summary())


        else:

           #Set up training and testing data to have a random non-correlated feature then
            X_train = X_train_features[month]['HousingDensity'].copy()
            X_test = X_test_features[month]['HousingDensity'].copy()
            cv_results = 0
            cor = 0

            len1 = len(X_train)
            len2 = len(X_test)

            X_train = np.array(X_train).reshape(len1, 1)
            X_test = np.array(X_test).reshape(len2, 1)

            Final_model = model_type
            Final_model.fit(X_train, y_train)

        #    Uncertainty = sm.OLS(y_train, X_train).fit()
         #   print(Uncertainty.summary())


        # Get training data model performance to tune hyperparameters
        yt_pred = Final_model.predict(X_train)

        yt_pred = [0 if x < 0 else x for x in yt_pred]
        O_r2_train = sklearn.metrics.r2_score(y_train, yt_pred)
        O_rmse_train = sklearn.metrics.mean_squared_error(y_train, yt_pred, squared = False)

    # predict X_test
        y_pred = Final_model.predict(X_test)

        y_pred = [0 if x < 0 else x for x in y_pred]
        O_r2_test = sklearn.metrics.r2_score(y_test, y_pred)
        O_rmse_test = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False)



    #plot the predictions
        PerfDF=pd.DataFrame(list(zip(y_pred, y_test)), columns=['y_pred', 'y_test'])

    #Add indoor demands
        Indoor=['Jan', 'Feb', 'Mar', 'Nov', 'Dec']
        if month in Indoor:
            PerfDF['y_test_tot']=PerfDF['y_test']
            PerfDF['y_pred_tot']=PerfDF['y_pred']
        else:
            PerfDF['y_test_tot']=PerfDF['y_test']+list(self.I_mean_test)
            PerfDF['y_pred_tot']=PerfDF['y_pred']+list(self.Cons_mean_test)

        T_r2 = sklearn.metrics.r2_score(PerfDF['y_test_tot'], PerfDF['y_pred_tot'])
        T_rmse= sklearn.metrics.mean_squared_error(PerfDF['y_test_tot'], PerfDF['y_pred_tot'], 
                                                   squared = False)

        #print('Total R2 is ', T_r2)
        #print('Total rmse is ', T_rmse)


        PerfDF['Year'] = list(self.slc_test['Jul'].index)
        PerfDF=PerfDF.set_index('Year')


        datetime_object = datetime.datetime.strptime(month, "%b")
        PerfDF['month'] = datetime_object.month
        PerfDF['Year']=PerfDF.index


        #set up dates so all months can be combined and sorted
        day=[]
        for index, row in PerfDF.iterrows():
            day.append(calendar.monthrange(int(row['Year']), int(row['month']))[1])

        PerfDF['Day']=day

        PerfDF['Date'] = pd.to_datetime(PerfDF[['Year', 'month', 'Day']])

        #PerfDF=PerfDF.set_index('Date')
        PerfDF=PerfDF.drop(columns=['Year', 'month', 'Day'])
        PerfDF=PerfDF.reset_index()

        params = [snowfeatures, conservation, cor_threshold, colinearity_thresh]

        return X_test, PerfDF, O_rmse_train, O_r2_train ,O_rmse_test, O_r2_test , params, cv_results, cor , Final_model.coef_
    


    def Outdoor_Demand_ModelFinal(self, TrainDF, month, X_train_features, y_train_target, X_test_features, y_test_target,
                              snowfeatures, conservation, cor_threshold, colinearity_thresh, cv_splits,
                             model_type, scoring ):


    #subset these features out of main DF and put into cute heatmap plot

        DFcor = copy.deepcopy(TrainDF[month])

        #if snowfeatures is True:
         #   print('LCC Snowfeatures are being used')
        Indoor=['Jan', 'Feb', 'Mar', 'Nov', 'Dec']
        if snowfeatures is False:
            if month in Indoor:
                DFcor=DFcor
            else:
                snow=['Nov_snow_in','Dec_snow_in', 'Jan_snow_in','Feb_snow_in', 
                        'Mar_snow_in','Apr_snow_in', 'Total_snow_in', 'Snow_shortage']
                DFcor=DFcor.drop(columns=snow)


        cor=DFcor.copy()
        if conservation is False:
            del cor['cons_goal']
            cor = cor.corr()
            cor =cor.iloc[:,-1:]
        if conservation is True:
            cor = cor.corr()
            cor =cor.iloc[:,-2:]
            del cor['cons_goal']

        cor['Target_gpcd']=np.abs(cor['Target_gpcd'])
        cor=cor.sort_values(by=['Target_gpcd'], ascending=False)
        cor=cor.dropna()

    #Selecting highly correlated features
        relevant_features = cor[cor['Target_gpcd']>cor_threshold]
        CorFeat = list(relevant_features.index)

        CorDF= DFcor[CorFeat]
        cor = np.abs(CorDF.corr())
        cor = cor.mask(np.tril(np.ones(cor.shape)).astype(np.bool))
        #remove colinearity
        cor = cor[cor.columns[cor.max() < colinearity_thresh]]
        CorFeat=cor.columns
        cor = cor.T
        cor = cor[CorFeat]

        #print('Remaining features are', CorFeat)


       #Set up training and testing data 
        X_train = X_train_features[month][CorFeat].copy()
    #X_train = slc_train_features['Jul'][JulF]
        y_train = y_train_target[month].copy()

        X_test = X_test_features[month][CorFeat].copy()
    #X_test = slc_test_features['Jul'][JulF]
        y_test = y_test_target[month].copy()

        # step-1: create a cross-validation scheme
        folds = KFold(n_splits = cv_splits, shuffle = True, random_state = 42)

    # step-2: specify range of hyperparameters to tune
        if len(CorFeat) > 1 :
            hyper_params = [{'n_features_to_select': list(range(1, len(CorFeat)))}]


    # step-3: perform grid search
    # 3.1 specify model, key to set intercept to false
            trainmodel = model_type
            trainmodel.fit(X_train, y_train)
            rfe = RFE(trainmodel)             

    # 3.2 call GridSearchCV()
            model_cv = GridSearchCV(estimator = rfe, 
                            param_grid = hyper_params, 
                            scoring= scoring, 
                            cv = folds, 
                            verbose = 0,
                            return_train_score=True)      

    # fit the model
            model_cv.fit(X_train, y_train)

    # create a KFold object with 5 splits 
            folds = KFold(n_splits = cv_splits, shuffle = True, random_state = 42)
            scores = cross_val_score(trainmodel, X_train, y_train, scoring=scoring, cv=folds)
           # print('CV scores = ', scores) 

    # cv results
            cv_results = pd.DataFrame(model_cv.cv_results_)


         #code to select features for final model, tell how many features
            N_feat=cv_results.loc[cv_results['mean_test_score'].idxmax()]
            N_feat=N_feat['param_n_features_to_select']
            #print('Number of features to select is ', N_feat)
        # intermediate model
            n_features_optimal = N_feat

            Int_model = model_type
            Int_model.fit(X_train, y_train)

            rfe = RFE(Int_model, n_features_to_select=n_features_optimal)             
            rfe = rfe.fit(X_train, y_train)

    #make the final model with rfe features

    # tuples of (feature name, whether selected, ranking)
    # note that the 'rank' is > 1 for non-selected features

            Features =list(zip(X_train.columns,rfe.support_,rfe.ranking_))
            FeaturesDF=pd.DataFrame(Features, columns=['Feature', 'Important', 'Score'])
            FeaturesDF = FeaturesDF[FeaturesDF.Score<=1]
            RFE_features = list(FeaturesDF['Feature'])
            print('The final features are ', RFE_features)
            featurefile = self.cwd + '/Models/Features/' + month + '_features.pkl'
            with open(featurefile, 'wb') as f:
                pickle.dump(RFE_features, f)

            #select only RFE features for model training/validation
            X_train = X_train[RFE_features]
            X_test = X_test[RFE_features]


        #plot of selected features to make sure not colinear
            CorDF= X_train.copy()
            CorDF['Target_gpcd']=self.slc_train[month]['Target_gpcd']

            Final_model = model_type
            Final_model.fit(X_train, y_train)

            #grab uncertainty stats

            Uncertainty = sm.OLS(y_train, X_train).fit()
            print(Uncertainty.summary())


        else:

           #Set up training and testing data to have a random non-correlated feature then
            X_train = X_train_features[month]['HousingDensity'].copy()
            X_test = X_test_features[month]['HousingDensity'].copy()
            cv_results = 0
            cor = 0

            len1 = len(X_train)
            len2 = len(X_test)

            X_train = np.array(X_train).reshape(len1, 1)
            X_test = np.array(X_test).reshape(len2, 1)

            Final_model = model_type
            Final_model.fit(X_train, y_train)

            Uncertainty = sm.OLS(y_train, X_train).fit()
            print(Uncertainty.summary())


         # save the model to disk
        filename = self.cwd+ '/Models/' + month + '_demand_model.sav'
        pickle.dump(Final_model, open(filename, 'wb'))
        
        UNC_filename = self.cwd+ '/Models/' + month + '_demand_model_unc.sav'
        pickle.dump(Uncertainty, open(UNC_filename, 'wb'))
        
        
        
        #load model
        Final_model_loaded = pickle.load(open(filename, 'rb'))
        Final_model_loaded_unc = pickle.load(open(UNC_filename, 'rb'))
          
        # Get training data model performance to tune hyperparameters
        yt_pred = Final_model_loaded.predict(X_train)

        yt_pred = [0 if x < 0 else x for x in yt_pred]
        O_r2_train = sklearn.metrics.r2_score(y_train, yt_pred)
        O_rmse_train = sklearn.metrics.mean_squared_error(y_train, yt_pred, squared = False)

    # predict X_test
        y_pred = Final_model_loaded.predict(X_test)

        y_pred = [0 if x < 0 else x for x in y_pred]
        O_r2_test = sklearn.metrics.r2_score(y_test, y_pred)
        O_rmse_test = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False)


    #Predict using uncertainties
        Uy_pred = Final_model_loaded_unc.get_prediction(X_test)
        Uy_pred = Uy_pred.summary_frame()
        print(Uy_pred)
        lower = np.array(Uy_pred['mean_ci_lower'])
        #cannot have values below zero
        lower = [0 if x < 0 else x for x in lower]

        upper = np.array(Uy_pred['mean_ci_upper'])

        #plot the predictions
        PerfDF=pd.DataFrame(list(zip(y_test, y_pred)), columns=['y_test', 'y_pred'])
        PerfDF['y_pred_lower'] = lower
        PerfDF['y_pred_upper'] = upper




    #Add indoor demands
        Indoor=['Jan', 'Feb', 'Mar', 'Nov', 'Dec']
        if month in Indoor:
            PerfDF['y_test_tot']=PerfDF['y_test']
            PerfDF['y_pred_tot']=PerfDF['y_pred']
        else:
            PerfDF['y_test_tot']=PerfDF['y_test']+list(self.I_mean_test)
            PerfDF['y_pred_tot']=PerfDF['y_pred']+list(self.Cons_mean_test)
            PerfDF['y_pred_lower_tot']=PerfDF['y_pred_lower']+list(self.Cons_mean_test)
            PerfDF['y_pred_upper_tot']=PerfDF['y_pred_upper']+list(self.Cons_mean_test)

        T_r2 = sklearn.metrics.r2_score(PerfDF['y_test_tot'], PerfDF['y_pred_tot'])
        T_rmse= sklearn.metrics.mean_squared_error(PerfDF['y_test_tot'], PerfDF['y_pred_tot'], 
                                                   squared = False)


        PerfDF['Year'] = list(self.slc_test['Jul'].index)
        PerfDF=PerfDF.set_index('Year')


        datetime_object = datetime.datetime.strptime(month, "%b")
        PerfDF['month'] = datetime_object.month
        PerfDF['Year']=PerfDF.index


        #set up dates so all months can be combined and sorted
        day=[]
        for index, row in PerfDF.iterrows():
            day.append(calendar.monthrange(int(row['Year']), int(row['month']))[1])

        PerfDF['Day']=day

        PerfDF['Date'] = pd.to_datetime(PerfDF[['Year', 'month', 'Day']])

        #PerfDF=PerfDF.set_index('Date')
        PerfDF=PerfDF.drop(columns=['Year', 'month', 'Day'])
        PerfDF=PerfDF.reset_index()


        params = [snowfeatures, conservation, cor_threshold, colinearity_thresh]
        
    
        
        
        return X_test, PerfDF, O_rmse_train, O_r2_train ,O_rmse_test, O_r2_test , params, cv_results, cor , Final_model_loaded.coef_




    #make an optimization function
    #put in your parameter dictionary, month of interest, and scoring method (RMSE or R2)
    def Demand_Optimization(self, Param_dict, month, scoring ):
        print('The automated algorithm automatically optimizes the respective model by looping over input parameters within')
        print('the training data. In addiiton, the algorithm checks for colinearity between features, removing the one with')
        print('less correlation to the target.')
        param_list = []
        performance_list=[]
        for i in Param_dict['snowfeatures']:
       #     print('Snowfeatures is ' + str(i))
            for j in Param_dict['conservation']:
        #        print('Conservation is ' + str(j))
                for k in Param_dict['cor_threshold']:
         #           print('Correlation threshold: ', k)
                    #pbar = ProgressBar()
                    for l in Param_dict['colinearity_thresh']:
          #              print('Colinearity threshold: ', l)
                        X_test_RFE, PerfDF, O_rmse_train,O_r2_train, O_rmse_test, O_r2_test, params, cv_results, cor, coef = self.Outdoor_Demand_Model(self.slc_train, month, self.slc_train_features, self.slc_train_target, self.slc_test_features,
                        self.slc_test_target, snowfeatures= i, conservation = j, cor_threshold = k, colinearity_thresh = l, cv_splits = 5,
                                model_type = linear_model.Ridge(fit_intercept = False, alpha=1), 
                                scoring = 'neg_root_mean_squared_error')
                        param_list.append(params)
                        if scoring =='R2':
                            performance_list.append(O_r2_test)
                        if scoring =='RMSE':
                            performance_list.append(O_rmse_test)



        #take model performances and put into DF so they can be joined and sorted                
        ParamDF = pd.DataFrame(param_list, columns =list(Param_dict.keys()))
        PerfDF = pd.DataFrame(performance_list, columns =[scoring])     
        ParamEval = pd.concat([ParamDF, PerfDF], axis=1)  

        if scoring =='R2':
            ParamEval = ParamEval.sort_values(by=[scoring], ascending = False)
        else:
            ParamEval = ParamEval.sort_values(by=[scoring])

        #select the first row of parameters as this is the one that shows the greatest performance
        ParamEval=ParamEval.head(1)

        X_test_RFE, PerfDF, O_rmse_train,O_r2_train,O_rmse_test, O_r2_test, params, cv_results, cor, coef = self.Outdoor_Demand_ModelFinal(self.slc_train, month, self.slc_train_features, self.slc_train_target, self.slc_test_features,             self.slc_test_target,snowfeatures= list(ParamEval['snowfeatures'])[0] , 
                                conservation = list(ParamEval['conservation'])[0],
                                cor_threshold = list(ParamEval['cor_threshold'])[0],
                                colinearity_thresh = list(ParamEval['colinearity_thresh'])[0],
                                cv_splits = 5, model_type = linear_model.Ridge(fit_intercept = False, alpha=1), 
                                scoring = 'neg_root_mean_squared_error')
       # model_plots(PerfDF, cv_results, cor, X_test_RFE, coef,  scoring, month)
    
    
    
        print('The best training parameters are below with their scoring method: ', scoring)
        print(ParamEval)
        return  PerfDF, cv_results, cor, X_test_RFE, coef
    
    
    #Make a function to put all of the predictions together
    def Demand_Forecast(self, prediction_dictionary, pdict, df, pred, test, units, plotname, model, predcol, obscol):
        FinalDF=pd.DataFrame()
        if pdict is True:
            print('yes')
            for i in prediction_dictionary:
                FinalDF=FinalDF.append(prediction_dictionary[i])

            FinalDF=FinalDF.sort_index()
        else:
            print('pdict is not used')
            FinalDF = df

        #adjust date range to improve figure
        FinalDF['Date']= pd.date_range(start=pd.datetime(2015,1,1),end=pd.datetime(2017,12,1), freq='MS')
        #FinalDF['Date'] = Conditions

        FinalDF.index = FinalDF['Date']
        del FinalDF['Date']



        plotmin_tot = FinalDF[[pred, test]].min().min()
        plotmax_tot = FinalDF[[pred, test]].max().max()

        Xplotmin = FinalDF.index[0]-np.timedelta64(20, 'D')
        Xplotmax = FinalDF.index[-1]+np.timedelta64(33, 'D')

        plt.rc_context({ 'xtick.color':'black'})
        fig, ax = plt.subplots(1,6, constrained_layout=True)
        fig.set_size_inches(9,3.5)

        gs2 = ax[0].get_gridspec()
        # remove the underlying axes
        ax[0].remove()
        ax[1].remove()
        ax[2].remove()
        ax[3].remove()
        axbig = fig.add_subplot(gs2[:4])
        #axbig.set_title('Total demand Timeline Evaluation')
       # axbig.plot(FinalDF[pred], color='orange', label= model)
        #axbig.plot(FinalDF[test],color='blue', label='Observed')
        axbig.bar(FinalDF.index-np.timedelta64(7, 'D'), FinalDF[pred], color=predcol, label= model ,width = 15,  align="center")
        axbig.bar(FinalDF.index+np.timedelta64(8, 'D'), FinalDF[test], color=obscol, label= 'Observed',width = 15,  align="center")
        axbig.set_xlabel('Wet                                       Dry                                      Average \n \n Supply Scenario')
        axbig.set_ylim(plotmin_tot-.9,plotmax_tot*1.3)
        axbig.set_xlim(Xplotmin, Xplotmax)
        axbig.set_ylabel('Demand ('+ units+')')
        axbig.legend(loc = 'upper left', facecolor = 'white')
        axbig.set_facecolor("white")
        axbig.spines['bottom'].set_color('black')
        axbig.spines['left'].set_color('black')
        axbig.tick_params(axis='both', which='both', length=5, color='red')
        axbig.xaxis.set_major_locator(mdates.MonthLocator())
        # Get only the month to show in the x-axis:
        axbig.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        axbig.annotate('A.', (FinalDF.index[-1], plotmax_tot*1.2), size = 14)
        xticks = axbig.xaxis.get_major_ticks()
        months = [0,5,10,12,17,22,24,29,34]

        xticks = axbig.xaxis.get_major_ticks()
        for i,tick in enumerate(xticks):
            #if i%5.5 != 0:
            if i not in months:
                tick.label1.set_visible(False)


        ax[4].remove()
        ax[5].remove()


        axbig2 = fig.add_subplot(gs2[4:])
        axbig2.scatter(FinalDF[test], FinalDF[pred],color=predcol, alpha=0.5)
        axbig2.set_ylabel('Predicted (' + units+')' )
        axbig2.set_xlabel('Observed (' + units+')')
        axbig2.set_ylim(plotmin_tot*.95,plotmax_tot*1.2)
        axbig2.set_xlim(plotmin_tot*.95,plotmax_tot*1.2)
       # axbig2.set_title('Indoor and Outdoor \n Model Performance')
        axbig2.plot([plotmin_tot,plotmax_tot],[plotmin_tot,plotmax_tot], color='black', linestyle='--' )
        #axbig2.set_xticks(np.arange(plotmin_tot, plotmax_tot, 100).round())
        #axbig2.set_yticks(np.arange(plotmin_tot, plotmax_tot, 100).round())
        axbig2.set_facecolor("white")
        axbig2.spines['bottom'].set_color('black')
        axbig2.spines['left'].set_color('black')
        axbig2.annotate('B.', (2050,plotmax_tot*1.1), size = 14)
        axbig2.set_xticks(np.arange(300, 2301, 500))
        axbig2.set_yticks(np.arange(300, 2301, 500))
        fig.tight_layout(pad=0.5)

        fig.savefig(self.cwd+'/Figures/' +str(plotname)+'.png', dpi = 300)
        r2 = sklearn.metrics.r2_score(FinalDF[test], FinalDF[pred])
        MAE= sklearn.metrics.mean_absolute_error(FinalDF[test], FinalDF[pred])
        RMSE= sklearn.metrics.mean_squared_error(FinalDF[test], FinalDF[pred], squared = False)
        MAPE=np.mean(np.abs((FinalDF[test]- FinalDF[pred])/FinalDF[test])*100)

        print('Total R2 is ', r2)
        print('Total MAE is ', MAE)
        print('Total RMSE is ', RMSE)
        print('Total MAPE is ', MAPE)

        FinalDF['Date']= pd.date_range(start=pd.datetime(2015,1,1),end=pd.datetime(2018,1,1), freq='M')
        FinalDF.index = FinalDF['Date']
        del FinalDF['Date']
        return FinalDF

    
    #Make a function to put all of the predictions together
    def Demand_ForecastErr(self, prediction_dictionary, pdict, df, pred, test, units, plotname, model, predcol, obscol):
        FinalDF=pd.DataFrame()
        if pdict is True:
            print('yes')
            for i in prediction_dictionary:
                FinalDF=FinalDF.append(prediction_dictionary[i])

            FinalDF=FinalDF.sort_index()
        else:
            print('pdict is not used')
            FinalDF = df

        #adjust date range to improve figure
        FinalDF['Date']= pd.date_range(start=pd.datetime(2015,1,1),end=pd.datetime(2017,12,1), freq='MS')
        #FinalDF['Date'] = Conditions

        FinalDF.index = FinalDF['Date']
        del FinalDF['Date']

        #Define Error bars for visualization
        FinalDF['UErr'] = FinalDF['y_pred_upper_tot'] - FinalDF['y_pred_tot']
        FinalDF['LErr'] = FinalDF['y_pred_tot'] - FinalDF['y_pred_lower_tot']

        asymmetric_error = [FinalDF['LErr'], FinalDF['UErr'] ]


        plotmin_tot = FinalDF[[pred, test]].min().min()
        plotmax_tot = FinalDF[[pred, test]].max().max()

        Xplotmin = FinalDF.index[0]-np.timedelta64(20, 'D')
        Xplotmax = FinalDF.index[-1]+np.timedelta64(33, 'D')

        plt.rc_context({ 'xtick.color':'black'})
        fig, ax = plt.subplots(1,5, constrained_layout=True)
        fig.set_size_inches(10,3.5)

        gs2 = ax[0].get_gridspec()
        # remove the underlying axes
        ax[0].remove()
        ax[1].remove()
        ax[2].remove()
        axbig = fig.add_subplot(gs2[:3])
        axbig.set_title('Total demand Timeline Evaluation')
        #axbig.plot(FinalDF['y_pred_lower_tot'], color='steelblue', label= model)
        axbig.plot(FinalDF['y_pred_upper_tot'], color='steelblue', label= model)
        axbig.fill_between(FinalDF.index, FinalDF['y_pred_upper_tot'],FinalDF['y_pred_lower_tot'], color = 'steelblue')
        axbig.plot(FinalDF[test],color='orange', label='Observed')
        axbig.scatter(FinalDF.index, FinalDF['y_pred_tot'], color = 'blue', s = 25)
       # axbig.bar(FinalDF.index-np.timedelta64(7, 'D'), FinalDF[pred], color=predcol, 
        #          yerr = asymmetric_error,
         #         label= model ,width = 15,  align="center")
        #axbig.bar(FinalDF.index+np.timedelta64(8, 'D'), FinalDF[test], color=obscol, label= 'Observed',width = 15,  align="center")





        axbig.set_xlabel('Wet                        Dry                       Average \n \n Supply Scenario')
        axbig.set_ylim(plotmin_tot-.9,plotmax_tot*1.3)
        axbig.set_xlim(Xplotmin, Xplotmax)
        axbig.set_ylabel('Demand ('+ units+')')
        axbig.legend(loc = 'upper right', facecolor = 'white')
        axbig.set_facecolor("white")
        axbig.spines['bottom'].set_color('black')
        axbig.spines['left'].set_color('black')
        axbig.tick_params(axis='both', which='both', length=5, color='red')
        axbig.xaxis.set_major_locator(mdates.MonthLocator())
        # Get only the month to show in the x-axis:
        axbig.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        axbig.annotate('A.', (FinalDF.index[-1], plotmax_tot*1.2), size = 14)
        xticks = axbig.xaxis.get_major_ticks()
        months = [0,5,10,12,17,22,24,29,34]

        xticks = axbig.xaxis.get_major_ticks()
        for i,tick in enumerate(xticks):
            #if i%5.5 != 0:
            if i not in months:
                tick.label1.set_visible(False)


        ax[3].remove()
        ax[4].remove()


        axbig2 = fig.add_subplot(gs2[3:])
        axbig2.errorbar(FinalDF[test], FinalDF[pred], yerr = asymmetric_error, fmt='.k', ecolor = 'steelblue', mec = 'blue') 
        axbig2.set_ylabel('Predicted (' + units+')' )
        axbig2.set_xlabel('Observed (' + units+')')
        axbig2.set_ylim(plotmin_tot*.95,plotmax_tot*1.2)
        axbig2.set_xlim(plotmin_tot*.95,plotmax_tot*1.2)
       # axbig2.set_title('Indoor and Outdoor \n Model Performance')
        axbig2.plot([plotmin_tot,plotmax_tot],[plotmin_tot,plotmax_tot], color='black', linestyle='--' )
        #axbig2.set_xticks(np.arange(plotmin_tot, plotmax_tot, 100).round())
        #axbig2.set_yticks(np.arange(plotmin_tot, plotmax_tot, 100).round())
        axbig2.set_facecolor("white")
        axbig2.spines['bottom'].set_color('black')
        axbig2.spines['left'].set_color('black')
        axbig2.annotate('B.', (2050,plotmax_tot*1.1), size = 14)
        axbig2.set_xticks(np.arange(300, 2301, 500))
        axbig2.set_yticks(np.arange(300, 2301, 500))

        fig.tight_layout(pad=0.5)

        fig.savefig(self.cwd + '/Figures/' +str(plotname)+'bar.png', dpi = 300)
        r2 = sklearn.metrics.r2_score(FinalDF[test], FinalDF[pred])
        MAE= sklearn.metrics.mean_absolute_error(FinalDF[test], FinalDF[pred])
        RMSE= sklearn.metrics.mean_squared_error(FinalDF[test], FinalDF[pred], squared = False)
        MAPE=np.mean(np.abs((FinalDF[test]- FinalDF[pred])/FinalDF[test])*100)

        print('Total R2 is ', r2)
        print('Total MAE is ', MAE)
        print('Total RMSE is ', RMSE)
        print('Total MAPE is ', MAPE)

        FinalDF['Date']= pd.date_range(start=pd.datetime(2015,1,1),end=pd.datetime(2018,1,1), freq='M')
        FinalDF.index = FinalDF['Date']
        del FinalDF['Date']
        return FinalDF
    
    
    
    
      
    
    def gradientbars_sliced(self, bars):
        ax = bars[0].axes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor("none")
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            grad = np.linspace(y, y + h, 256).reshape(256, 1)
            ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", zorder=0, origin='lower',
                      vmin= - max(np.abs(ymin), ymax), vmax=max(np.abs(ymin), ymax), cmap='Spectral')
        ax.axis([xmin, xmax, ymin, ymax])

        
        
    def model_plots(self, PerfDF, cv_results,cor, X_test_RFE, coef, scoring, month):
    
        plotmin = PerfDF[['y_pred', 'y_test']].min().min()
        plotmax = PerfDF[['y_pred', 'y_test']].max().max()

        plotmin_tot = PerfDF[['y_pred_tot', 'y_test_tot']].min().min()
        plotmax_tot = PerfDF[['y_pred_tot', 'y_test_tot']].max().max()

        # plotting cv results
        plt.figure(figsize=(12,10))
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.savefig('C:/Users/rjohnson18/Box/Dissertation/Paper1/Figs/' + month + '_corMatrix.pdf')
        plt.show()

        fig, ax = plt.subplots(3,3, constrained_layout=True)
        fig.set_size_inches(9,10)


        ax[0,0].plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
        ax[0,0].plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
        ax[0,0].set_xlabel('number of features')
        ax[0,0].set_ylabel(scoring)
        ax[0,0].set_title("Optimal Number of Features")
        ax[0,0].legend(['test score', 'train score'], loc='upper left')
        ax[0,0].spines['bottom'].set_color('black')
        ax[0,0].spines['left'].set_color('black')

        ax[0,1].scatter(PerfDF['y_test'], PerfDF['y_pred'],color='blue', alpha=0.5)
        ax[0,1].set_ylabel('Predicted')
        ax[0,1].set_xlabel('Observed')
        ax[0,1].set_ylim(plotmin-5,plotmax+5)
        ax[0,1].set_xlim(plotmin-5,plotmax+5)
        ax[0,1].set_title('Outdoor Model Performance')
        ax[0,1].plot([plotmin,plotmax],[plotmin,plotmax], color='red', linestyle='--' )
        ax[0,1].spines['bottom'].set_color('black')
        ax[0,1].spines['left'].set_color('black')

        ax[0,2].scatter(PerfDF['y_test_tot'], PerfDF['y_pred_tot'],color='blue', alpha=0.5)
        ax[0,2].set_ylabel('Predicted')
        ax[0,2].set_xlabel('Observed')
        ax[0,2].set_ylim(plotmin_tot-5,plotmax_tot+5)
        ax[0,2].set_xlim(plotmin_tot-5,plotmax_tot+5)
        ax[0,2].set_title('Indoor and Outdoor \n Model Performance')
        ax[0,2].plot([plotmin_tot,plotmax_tot],[plotmin_tot,plotmax_tot], color='red', linestyle='--' )
        ax[0,2].spines['bottom'].set_color('black')
        ax[0,2].spines['left'].set_color('black')




        gs = ax[1, 1].get_gridspec()
        # remove the underlying axes
        ax[1,0].remove()
        ax[1,1].remove()
        ax[1,2].remove()

        PerfDF['Error'] = (PerfDF['y_pred']-PerfDF['y_test'])
        axbig1 = fig.add_subplot(gs[1, :])
        axbig1.set_title(month+' Outdoor demand Timeline Evaluation')
        axbig1.axhline(y = 0 , color = 'black')
        #axbig1.bar(PerfDF.index, PerfDF['y_pred'], color='orange', label='Predicted')
        Error1 = axbig1.bar(PerfDF.index, PerfDF['Error'],color='blue', label='Prediction Error')
        axbig1.set_xlabel('Year')
        axbig1.set_ylabel('Error (GPCD)')
        axbig1.spines['bottom'].set_color('black')
        axbig1.spines['left'].set_color('black')
        gradientbars_sliced(Error1)


        gs2 = ax[2, 1].get_gridspec()
        # remove the underlying axes
        ax[2,0].remove()
        ax[2,1].remove()
        ax[2,2].remove()

        #create error value
        PerfDF['Error_tot'] = (PerfDF['y_pred_tot']-PerfDF['y_test_tot'])

        axbig2 = fig.add_subplot(gs2[2, :])
        axbig2.set_title(month+' Total Demand Error Timeline Evaluation')
        Error2 = axbig2.bar(PerfDF.index, PerfDF['Error_tot'], color='blue', label='Predicted')
        axbig2.axhline(y = 0 , color = 'black')
        #axbig2.bar(PerfDF.index, PerfDF['y_test_tot'],color='blue', label='Observed')
        axbig2.set_xlabel('Year')
        axbig2.set_ylabel('Error (GPCD)')
        axbig2.spines['bottom'].set_color('black')
        axbig2.spines['left'].set_color('black')
        gradientbars_sliced(Error2)

        fig.suptitle(month+ ' Evaluation', size = 16)
        fig.savefig('C:/Users/rjohnson18/Box/Dissertation/Paper1/Figs/' + month + '_demand.pdf')    

        O_r2 = sklearn.metrics.r2_score(PerfDF['y_test'],PerfDF['y_pred'])
        O_rmse= sklearn.metrics.mean_squared_error(PerfDF['y_test'],PerfDF['y_pred'], squared = False)
        O_mae= sklearn.metrics.mean_absolute_error(PerfDF['y_test'],PerfDF['y_pred'])
        O_mape= sklearn.metrics.mean_absolute_percentage_error(PerfDF['y_test'],PerfDF['y_pred'])

        T_r2 = sklearn.metrics.r2_score(PerfDF['y_test_tot'],PerfDF['y_pred_tot'])
        T_rmse= sklearn.metrics.mean_squared_error(PerfDF['y_test_tot'],PerfDF['y_pred_tot'], squared = False)
        T_mae= sklearn.metrics.mean_absolute_error(PerfDF['y_test_tot'],PerfDF['y_pred_tot'])
        T_mape= sklearn.metrics.mean_absolute_percentage_error(PerfDF['y_test_tot'],PerfDF['y_pred_tot'])

        print('The outdoor Demand prediction RMSE is ', O_rmse)
        print('The outdoor Demand prediction R2 is ', O_r2)

        print('The Total Demand prediction RMSE is ', T_rmse)
        print('The Total Demand prediction R2 is ', T_r2)
        print('The Total Demand prediction MAE is ', T_mae)
        print('The Total Demand prediction MAPE is ', T_mape, '%')                                                       

        print('The final set of features for ' + month + ' are', list(X_test_RFE.columns))
        print('The coefficients for each feature are', coef)
        #set DF up so that all months can be easily combined, basically year-month index
        
        
        
        
    #need to make predictions with model
    def Prediction(self, Forecast_data, IndoorDemand, units, Season, population,pred_name, save_fig):
        self.units = units
        self.Season = Season,
        self.population = population
        self.pred_name = pred_name
        self.save_fig = save_fig
        
        outdoor_months = ['Apr', 'May', 'Jun', 'Jul','Aug', 'Sep','Oct']
        Indoor=['Jan', 'Feb', 'Mar', 'Nov', 'Dec']
        All_months =  ['Jan', 'Feb', 'Mar','Apr', 'May', 'Jun', 'Jul','Aug', 'Sep','Oct', 'Nov', 'Dec']

        self.Forecast=copy.deepcopy(Forecast_data)

        #CSD-WDM has a unique model for each month
        for month in All_months:
            
            if month in outdoor_months:
                #load optimized features and models
                #filepaths
                featurefile = self.cwd + '/Models/Features/' + month + '_features.pkl'
                modelfile = self.cwd+ '/Models/' + month + '_demand_model.sav'
                UNC_filename = self.cwd+ '/Models/' + month + '_demand_model_unc.sav'

                #Loading
                features = pickle.load(open(featurefile, 'rb'))
                model = pickle.load(open(modelfile, 'rb'))
                model_unc = pickle.load(open(UNC_filename, 'rb'))

                #select only RFE features for model prediction
                Forecast_data = self.Forecast[month][features]


                #Predict
                #Deterministic
                y_pred = model.predict(Forecast_data)

                #Using uncertainties
                Uy_pred = model_unc.get_prediction(Forecast_data)
                Uy_pred = Uy_pred.summary_frame()
                #print(Uy_pred)
                lower = np.array(Uy_pred['mean_ci_lower'])

                #cannot have values below zero
                lower = [0 if x < 0 else x for x in lower]

                upper = np.array(Uy_pred['mean_ci_upper'])

                #put in dataframe
                PerfDF=pd.DataFrame(list(y_pred), columns=['y_pred'])
                PerfDF['y_pred_lower'] = lower
                PerfDF['y_pred_upper'] = upper

            #Add indoor demands
            if month in Indoor:
                Forecast_data = self.Forecast[month]
                Ind_dem_list = [IndoorDemand]*len(Forecast_data)
                PerfDF=pd.DataFrame(Ind_dem_list, columns=['y_pred_tot'])
                PerfDF['y_pred_lower_tot'] = PerfDF['y_pred_tot']
                PerfDF['y_pred_upper_tot'] = PerfDF['y_pred_tot']
            else:
                PerfDF['y_pred_tot']=PerfDF['y_pred']+IndoorDemand
                PerfDF['y_pred_lower_tot']=PerfDF['y_pred_lower']+IndoorDemand
                PerfDF['y_pred_upper_tot']=PerfDF['y_pred_upper']+IndoorDemand

            PerfDF['Year'] = list(Forecast_data.index)
            PerfDF=PerfDF.set_index('Year')


            datetime_object = datetime.datetime.strptime(month, "%b")
            PerfDF['month'] = datetime_object.month
            PerfDF['Year']=PerfDF.index


            #set up dates so all months can be combined and sorted
            day=[]
            for index, row in PerfDF.iterrows():
                day.append(calendar.monthrange(int(row['Year']), int(row['month']))[1])

            PerfDF['Day']=day

            PerfDF['Date'] = pd.to_datetime(PerfDF[['Year', 'month', 'Day']])

            #PerfDF=PerfDF.set_index('Date')
            PerfDF=PerfDF.drop(columns=['Year', 'month', 'Day'])
            PerfDF=PerfDF.reset_index()

            colrem = Forecast_data.columns
            self.Forecast[month] = self.Forecast[month].reset_index(drop=True)
            self.Forecast[month] = pd.concat([self.Forecast[month], PerfDF], axis=1, join="inner")
            self.Forecast[month] = self.Forecast[month].set_index('Date')
            self.Forecast[month] = self.Forecast[month].drop(columns=colrem)
            
        print('Predictions complete, displaying results results')
        print('CSDWDM demand estimates based in user input conditions complete, access results with CSDWDM.Forecast and/or CSDWDM.SeasonDF')
        self.Season_Demand()
         #make a plot of predictions
        self.Demand_Plot('y_pred_tot', 'CSD_WDM_Err','CSD-WDM ')
            
             #Make a function to put all of the predictions together
    def Demand_Plot(self, pred, plotname, model):
        FinalDF=pd.DataFrame()
        
        #Extract predictions
        pred_list = ['y_pred_tot', 'y_pred_lower_tot', 'y_pred_upper_tot']
        
        for i in self.Forecast:
            preds = self.Forecast[i][pred_list]
            FinalDF=FinalDF.append(preds)
            FinalDF=FinalDF.sort_index()
      

        #adjust date range to improve figure
        FinalDF['Date']= pd.date_range(start=pd.datetime(2015,1,1),end=pd.datetime(2017,12,1), freq='MS')
        #FinalDF['Date'] = Conditions

        FinalDF.index = FinalDF['Date']
        del FinalDF['Date']

        #Define Error bars for visualization
        FinalDF['UErr'] = FinalDF['y_pred_upper_tot'] - FinalDF['y_pred_tot']
        FinalDF['LErr'] = FinalDF['y_pred_tot'] - FinalDF['y_pred_lower_tot']

        asymmetric_error = [FinalDF['LErr'], FinalDF['UErr'] ]


        plotmin_tot = FinalDF['y_pred_lower_tot'].min()*.3
        plotmax_tot = FinalDF['y_pred_upper_tot'].max()*1.3

        Xplotmin = FinalDF.index[0]-np.timedelta64(20, 'D')
        Xplotmax = FinalDF.index[-1]+np.timedelta64(33, 'D')

        plt.rc_context({ 'xtick.color':'black'})
        fig, ax = plt.subplots(1,4, constrained_layout=True)
        fig.set_size_inches(7,3.5)

        gs2 = ax[0].get_gridspec()
        # remove the underlying axes
        ax[0].remove()
        ax[1].remove()
        ax[2].remove()
        ax[3].remove()
        axbig = fig.add_subplot(gs2[:4])
        #axbig.set_title('Total demand Timeline Evaluation')
        #axbig.plot(FinalDF['y_pred_lower_tot'], color='steelblue', label= model)
        #axbig.plot(FinalDF['y_pred_upper_tot'], color='steelblue', label= model)
        #axbig.fill_between(FinalDF.index, FinalDF['y_pred_upper_tot'],FinalDF['y_pred_lower_tot'], color = 'steelblue')
        #axbig.plot(FinalDF[test],color='orange', label='Observed')
        #axbig.scatter(FinalDF.index, FinalDF['y_pred_tot'], color = 'blue', s = 25)
        axbig.bar(FinalDF.index-np.timedelta64(7, 'D'), FinalDF[pred], 
                  yerr = asymmetric_error, capsize = 4,
                  label= model ,width = 15,  align="center", color='blue' )
       # axbig.bar(FinalDF.index+np.timedelta64(8, 'D'), FinalDF[test], color=obscol, label= 'Observed',width = 15,  align="center")





        axbig.set_xlabel('Wet                        Dry                       Average \n \n Supply Scenario')
        axbig.set_ylim(plotmin_tot-.9,plotmax_tot*1.3)
        axbig.set_xlim(Xplotmin, Xplotmax)
        axbig.set_ylabel('Demand ('+ self.units+')')
        axbig.legend(loc = 'upper left', facecolor = 'white')
        axbig.set_facecolor("white")
        axbig.spines['bottom'].set_color('black')
        axbig.spines['left'].set_color('black')
        axbig.tick_params(axis='both', which='both', length=5, color='red')
        axbig.xaxis.set_major_locator(mdates.MonthLocator())
        # Get only the month to show in the x-axis:
        axbig.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        #axbig.annotate('A.', (FinalDF.index[-1], plotmax_tot*1.2), size = 14)
        xticks = axbig.xaxis.get_major_ticks()
        months = [0,5,10,12,17,22,24,29,34]

        xticks = axbig.xaxis.get_major_ticks()
        for i,tick in enumerate(xticks):
            #if i%5.5 != 0:
            if i not in months:
                tick.label1.set_visible(False)


        FinalDF['Date']= pd.date_range(start=pd.datetime(2015,1,1),end=pd.datetime(2018,1,1), freq='M')
        FinalDF.index = FinalDF['Date']
        del FinalDF['Date']
        self.FinalDF = FinalDF
        
        if self.save_fig == True:
            print('Saving results in prediction folder')
            fig.savefig(self.cwd + '/Predictions/' + self.pred_name+ '.png', dpi=500)


    #create a function to inform cumulative demand
    def Season_Demand(self):
        self.gal_2_AF = 3.06889e-6
        self.SeasonDF = pd.DataFrame()
        for month in self.Season[0]:
            #convert month to number and get days
            month_num = strptime(month,'%b').tm_mon
            days = monthrange(2022,month_num )[1]

            month_lab = month + '_dem'
            self.SeasonDF[month_lab] = self.Forecast[month]['y_pred_tot'].values * days * self.population * self.gal_2_AF 

        #Determine the season sum
        self.SeasonDF['Season_dem'] = self.SeasonDF.sum(axis =1)
        print(self.SeasonDF)
        
      
       
    
    #Need to make a scenario generator to run simulations through CSDWDM
    
    def ScenarioGenerator(self, Testing_df, supply_beg, supply_end, supply_step, climate_beg, climate_end, climate_step, indoordemand, Season, population, predname, save_results):
  
        #make range of values to add to scenarios
        self.supplyrange = list(map(str, list(np.arange(supply_beg, supply_end, supply_step))))
        self.climaterange = list(map(str,list(np.arange(climate_beg, climate_end, climate_step))))
        
        print('Making a total of ', len(self.supplyrange)*len(self.climaterange), ' climate and supply scenarios based in user inputs')
        
        
        self.indoordemand = indoordemand
        self.Season = Season
        self.population = population
        self.predname = predname
        self.save_results = save_results

        #This uses the 'average use as a baseline, 2017'
        self.Average = copy.deepcopy(Testing_df)
        for month in Season:
            self.Average[month] = pd.concat([self.Average[month].iloc[-1:]]*len(self.supplyrange))
            self.Average[month]['Supply_Range_%'] = self.supplyrange


        # initialize dictionary
        self.Scenarios = {}

        # iterating through the elements of list
        for climate in self.climaterange:
            self.Scenarios[climate] = copy.deepcopy(self.Average)

        for climate in self.climaterange:
            for month in self.Season:
                self.Scenarios[climate][month]['climate'] = float(climate)
                #get list of columns with key words
                temp_keyword = 'temp'
                temp_Key_cols = [col for col in self.Scenarios[climate][month].columns if temp_keyword in col]
                for temp_col in temp_Key_cols:
                    self.Scenarios[climate][month][temp_col] = self.Scenarios[climate][month][temp_col]+ self.Scenarios[climate][month]['climate']

                Supply_keyword = 'AcFt'
                Supply_Key_cols = [col for col in self.Scenarios[climate][month].columns if Supply_keyword in col]
                for sup_col in Supply_Key_cols:
                    self.Scenarios[climate][month][sup_col] = self.Scenarios[climate][month][sup_col] * ((self.Scenarios[climate][month]['Supply_Range_%'].astype(float)+100)/100)
        
        print('Scenario generation complete')
        #make a prediction on scenarios
        print('Using the CSD-WDM to make predictions on generated climate and supply scenarios')
        self.ScenarioPrediction()
        print('Predictions complete, now calculating the seasonal demands based as a function of ', ', '.join(self.Season), ', and a population of ', self.population)
        self.Season_Demand_Totals()
        print('Scenario generation and simulation complte, below are the resulting season demand estimates from CSD-WDM. To access this data, use CSDWDM.Demand_Matrix, with the climate scenarios as columns and supply scenarios as rows.')
        #print(self.Demand_Matrix)
        self.Demand_Heatmap()
        
        #save results
        
        if self.save_results == True:
            print('Saving results in prediction folder')
            self.Demand_Matrix.to_excel(self.cwd + '/Predictions/' + self.predname+ '_Demand_Matrix.xlsx')



    #Use the CSD-WDM to predict on a dictionary of input features
    def ScenarioPrediction(self):
        

        self.Simulation=copy.deepcopy(self.Scenarios)

        outdoor_months = ['Apr', 'May', 'Jun', 'Jul','Aug', 'Sep','Oct']
        Indoor=['Jan', 'Feb', 'Mar', 'Nov', 'Dec']
        All_months =  ['Jan', 'Feb', 'Mar','Apr', 'May', 'Jun', 'Jul','Aug', 'Sep','Oct', 'Nov', 'Dec']
        
        for scenario in self.Simulation.keys():

            #CSD-WDM has a unique model for each month
            for month in All_months:

                if month in outdoor_months:
                    #load optimized features and models
                    #filepaths
                    featurefile = self.cwd + '/Models/Features/' + month + '_features.pkl'
                    modelfile = self.cwd+ '/Models/' + month + '_demand_model.sav'
                    UNC_filename = self.cwd+ '/Models/' + month + '_demand_model_unc.sav'

                    #Loading
                    features = pickle.load(open(featurefile, 'rb'))
                    model = pickle.load(open(modelfile, 'rb'))
                    model_unc = pickle.load(open(UNC_filename, 'rb'))

                    #select only RFE features for model prediction
                    Forecast_data = self.Simulation[scenario][month][features]


                    #Predict
                    #Deterministic
                    y_pred = model.predict(Forecast_data)

                    #Using uncertainties
                    Uy_pred = model_unc.get_prediction(Forecast_data)
                    Uy_pred = Uy_pred.summary_frame()
                    #print(Uy_pred)
                    lower = np.array(Uy_pred['mean_ci_lower'])

                    #cannot have values below zero
                    lower = [0 if x < 0 else x for x in lower]

                    upper = np.array(Uy_pred['mean_ci_upper'])

                    #put in dataframe
                    PerfDF=pd.DataFrame(list(y_pred), columns=['y_pred'])
                    PerfDF['y_pred_lower'] = lower
                    PerfDF['y_pred_upper'] = upper

                #Add indoor demands
                if month in Indoor:
                    Forecast_data = self.Simulation[scenario][month]
                    Ind_dem_list = [self.indoordemand]*len(Forecast_data)
                    PerfDF=pd.DataFrame(Ind_dem_list, columns=['y_pred_tot'])
                    PerfDF['y_pred_lower_tot'] = PerfDF['y_pred_tot']
                    PerfDF['y_pred_upper_tot'] = PerfDF['y_pred_tot']
                else:
                    PerfDF['y_pred_tot']=PerfDF['y_pred']+self.indoordemand
                    PerfDF['y_pred_lower_tot']=PerfDF['y_pred_lower']+self.indoordemand
                    PerfDF['y_pred_upper_tot']=PerfDF['y_pred_upper']+self.indoordemand

                PerfDF['Year'] = list(Forecast_data.index)
                PerfDF=PerfDF.set_index('Year')


                datetime_object = datetime.datetime.strptime(month, "%b")
                PerfDF['month'] = datetime_object.month
                PerfDF['Year']=PerfDF.index


                #set up dates so all months can be combined and sorted
                day=[]
                for index, row in PerfDF.iterrows():
                    day.append(calendar.monthrange(int(row['Year']), int(row['month']))[1])

                PerfDF['Day']=day

                PerfDF['Date'] = pd.to_datetime(PerfDF[['Year', 'month', 'Day']])

                #PerfDF=PerfDF.set_index('Date')
                PerfDF=PerfDF.drop(columns=['Year', 'month', 'Day'])
                PerfDF=PerfDF.reset_index()

                colrem = Forecast_data.columns
                self.Simulation[scenario][month] = self.Simulation[scenario][month].reset_index(drop=True)
                self.Simulation[scenario][month] = pd.concat([self.Simulation[scenario][month], PerfDF], axis=1, join="inner")
                self.Simulation[scenario][month] = self.Simulation[scenario][month].set_index('Date')
                self.Simulation[scenario][month] = self.Simulation[scenario][month].drop(columns=colrem)

                
                
   #create a function to inform cumulative demand
    def Season_Demand_Totals(self):
        self.gal_2_AF = 3.06889e-6
        self.SeasonSimDF = {}
        for scenario in self.Simulation.keys():
            self.SeasonSimDF[scenario] = pd.DataFrame()
            for month in self.Season:
                #convert month to number and get days
                month_num = strptime(month,'%b').tm_mon
                days = monthrange(2022,month_num )[1]

                month_lab = month + '_dem'
                self.SeasonSimDF[scenario][month_lab] = self.Simulation[scenario][month]['y_pred_tot'].values * days * self.population * self.gal_2_AF 

            #Determine the season sum
            self.SeasonSimDF[scenario]['Season_dem'] = self.SeasonSimDF[scenario].sum(axis =1)
            
       #put into a matrix for easy lookup/reference
        self.DemandMatrix()
            
            
            
    def DemandMatrix(self):
        self.Demand_Matrix = pd.DataFrame(columns=[self.climaterange], index=[self.supplyrange])
        for climate in self.climaterange:
            self.Demand_Matrix[climate] = np.around(self.SeasonSimDF[climate]['Season_dem'].values,-1)
            
            
            
    def Demand_Heatmap(self):
    
        fontsize = 25
        ticksize = 20

        sns.set(rc = {'figure.figsize':(17,14)})
        sns.color_palette("Spectral", as_cmap=True)

        plt = sns.heatmap(self.Demand_Matrix, annot=True, fmt = 'g',cmap = 'Spectral_r', cbar_kws ={'label':'Demand (AcFt)'},
                          annot_kws={"size": 12})
        plt.set_xlabel('Climate Scenario ($^{o}C$)', fontsize = fontsize)
        plt.set_ylabel('Supply Scenario (% of average)', fontsize = fontsize)
        plt.set_yticklabels(plt.get_ymajorticklabels(),size = ticksize)
        plt.set_xticklabels(plt.get_xmajorticklabels(),size = ticksize)
        plt.figure.axes[-1].yaxis.label.set_size(fontsize)

        if self.save_results == True:
            print('Saving results in prediction folder')
            fig = plt.get_figure()
            fig.savefig(self.cwd + '/Predictions/' + self.predname + '_Demand_Matrix.png', dpi=500)


            
            
     

