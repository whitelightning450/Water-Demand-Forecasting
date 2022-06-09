#!/usr/bin/env python
# coding: utf-8

#Load neccesary modules
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from sklearn.metrics import r2_score
#from sklearn.metrics import mean_absolute_percentage_error
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from sklearn import preprocessing
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import seaborn as sns; sns.set()
import joblib
from sklearn.datasets import make_regression
from pathlib import Path
import copy
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import sklearn
from progressbar import ProgressBar
import datetime
import calendar

get_ipython().run_line_magic('matplotlib', 'inline')


class CSD_WDM(object):

    
    def __init__(self, filename):
        self.file_name = filename
        
        
    #Step 1, convert create features and targets
    def Process_Data(self, forecast, droughtYR, surplusYR, avgYR):

        historical = self.file_name
        predYR = surplusYR
        
        #setting year as index
        for i in historical:
            historical[i].index = historical[i]['Year']
            del historical[i]['Year']
            
        # set year as index and fill in Target_gpcd with any number as a placeholder
        for i in forecast: #######################
            forecast[i]['Year'] = predYR
            forecast[i].index = forecast[i]['Year']
            del forecast[i]['Year']
            forecast[i]['Target_gpcd'] = 100
        
        #split into testing and training dataframes
        test = {'Jan': pd.DataFrame() , 'Feb': pd.DataFrame() , 'Mar': pd.DataFrame() , 'Apr': pd.DataFrame() ,
                'May': pd.DataFrame() , 'Jun': pd.DataFrame() , 'Jul': pd.DataFrame() ,'Aug': pd.DataFrame() ,
                'Sep': pd.DataFrame() , 'Oct': pd.DataFrame() , 'Nov': pd.DataFrame() , 'Dec': pd.DataFrame() }
        train = copy.deepcopy(historical)

        TestYears = [droughtYR, surplusYR, avgYR]
        for i in test:
            for year in TestYears:
                test[i] = test[i].append(historical[i].loc[year])

        for i in historical:
            train[i] = train[i].drop(labels=[surplusYR, droughtYR, avgYR], axis=0)

        #split training and testing data into features and targets
        test_target=copy.deepcopy(test)
        test_features=copy.deepcopy(test)
        train_target=copy.deepcopy(train)
        train_features=copy.deepcopy(train)

        targetR=['Target_gpcd']
        for i in test:
            test_target[i]= test_target[i][targetR]
            test_features[i]= test_features[i].drop(columns=targetR)
            
        for i in train:
            train_target[i]= train_target[i][targetR]
            train_features[i]= train_features[i].drop(columns=targetR) 
             
        #store variables in the class structure
        self.TestTarget = test_target
        self.TestFeatures = test_features
        self.TrainTarget = train_target
        self.TrainFeatures = train_features
        self.Train = train
        self.Test = test
        self.DroughtYR = droughtYR
        self.SurplusYR = surplusYR
        self.AvgYR = avgYR
        self.PredYR = predYR #######################
        self.ForecastDF = forecast #######################
        self.Historical = historical
        
        
    def Demand_Optimization(self, Param_dict, scoring, figpath):   
        
        #load in all necessary variables from class structure
        ValDF = copy.deepcopy(self.Test)
        ForecastDF = self.ForecastDF #######################
        droughtYR = self.DroughtYR
        surplusYR = self.SurplusYR
        avgYR = self.AvgYR
        predYR = self.PredYR
       
        #calibrate model for the outdoor months using the outdoor demand model
        Months=['Jan','Feb','Mar','Apr', 'May' , 'Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov','Dec']
        pbar = ProgressBar()
        for month in pbar(Months):
            print('The model is automatically selecting features and calibrating the ', month, 'outdoor demand model.' )
            print('The automated algorithm optimizes the respective model by looping over input parameters within')
            print('the training data. In addititon, the algorithm checks for collinearity between features, removing the one with')
            print('the lessor correlation to the target.')
            param_list = []
            performance_list=[]
            for i in Param_dict['snowfeatures']:
                for j in Param_dict['conservation']:
                    for k in Param_dict['cor_threshold']:
                        for l in Param_dict['colinearity_thresh']:
                            X_test_RFE, PerfDF, O_rmse_train,O_r2_train, O_rmse_test, O_r2_test, params, cv_results, cor, coef = self.Outdoor_Demand_Model(month, snowfeatures= i, conservation = j, cor_threshold = k, colinearity_thresh = l, cv_splits = 5, 
                            model_type = linear_model.Ridge(fit_intercept = False, alpha=1), scoring = 'neg_root_mean_squared_error')
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

            X_test_RFE, PerfDF, O_rmse_train,O_r2_train,O_rmse_test, O_r2_test, params, cv_results, cor, coef = self.Outdoor_Demand_Model(month, snowfeatures= list(ParamEval['snowfeatures'])[0] , 
                                    conservation = list(ParamEval['conservation'])[0],
                                    cor_threshold = list(ParamEval['cor_threshold'])[0],
                                    colinearity_thresh = list(ParamEval['colinearity_thresh'])[0],
                                    cv_splits = 5, model_type = linear_model.Ridge(fit_intercept = False, alpha=1), 
                                    scoring = 'neg_root_mean_squared_error')
            #create plots to show what is happening
            self.model_plots(PerfDF, cv_results, cor, X_test_RFE, coef, scoring, month, figpath)
            
            print('The best training parameters are below with their scoring method: ', scoring)
            print(ParamEval)

            #add the outcome to the val dataframe
            colrem = ValDF[month].columns
            ValDF[month] = ValDF[month].reset_index(drop=True)
            ValDF[month] = pd.concat([ValDF[month], PerfDF], axis=1, join="inner")
            ValDF[month] = ValDF[month].set_index('Date')
            ValDF[month] = ValDF[month].drop(columns=colrem)
            #add the outcome to the forecast dataframe ##################
            colrem = ForecastDF[month].columns
            ForecastDF[month] = ForecastDF[month].reset_index(drop=True)
            ForecastDF[month] = pd.concat([ForecastDF[month], PerfDF], axis=1, join="inner")
            ForecastDF[month] = ForecastDF[month].set_index('Date')
            ForecastDF[month] = ForecastDF[month].drop(columns=colrem)
        
        #seperate val dataframe into one for the droguht year, the surplus year, and the average year
        ValDrought, ValSurplus, ValAvg = self.Seperate_Data(ValDF) 
        
        #add the historical average values to the dfs          dates get thrown off here btw
        ValDrought = self.Mean_gpcd(ValDrought, droughtYR)
        ValSurplus = self.Mean_gpcd(ValSurplus, surplusYR)   
        ValAvg = self.Mean_gpcd(ValAvg, avgYR)  
        ForecastDF = self.Mean_gpcd(ValAvg, predYR)
     
        return ValSurplus, ValDrought, ValAvg, ForecastDF

    
    def Outdoor_Demand_Model(self, month, snowfeatures, conservation, cor_threshold, colinearity_thresh,
                             cv_splits, model_type, scoring):
        
        #load in all necessary variables from class structure
        TestDF= copy.deepcopy(self.Test)
        y_test_target = self.TestTarget
        X_test_features = self.TestFeatures
        TrainDF = copy.deepcopy(self.Train)
        y_train_target = self.TrainTarget
        X_train_features = self.TrainFeatures
        
        #calculate average gpcds and gpcd goals
        I_mean=(TestDF['Jan']['Target_gpcd']+
              TestDF['Feb']['Target_gpcd']+
              TestDF['Mar']['Target_gpcd']+
              TestDF['Nov']['Target_gpcd']+
              TestDF['Dec']['Target_gpcd'])/5
        
        Cons_mean=(TestDF['Jan']['cons_goal']+ #################### will be a problem is conservation goals are not used
                TestDF['Feb']['cons_goal']+
                TestDF['Mar']['cons_goal']+
                TestDF['Nov']['cons_goal']+
                TestDF['Dec']['cons_goal'])/5

        DFcor = copy.deepcopy(TrainDF[month])

        """if snowfeatures is True:
            #print('LCC Snowfeatures are being used')"""
        #adjusting for whether snowfeatures is true or false
        if snowfeatures is False:
            snow=['Nov_snow_in','Dec_snow_in', 'Jan_snow_in','Feb_snow_in', 
                  'Mar_snow_in','Apr_snow_in', 'Total_snow_in', 'Snow_shortage']
            if 'Total_snow_in' in DFcor.columns:
                DFcor=DFcor.drop(columns=snow)
                
        #adjusting for whether conservation goals is true or false
        cor=DFcor.copy()
        if conservation is False:
            if 'cons_goal' in cor.columns:
                del cor['cons_goal']
            cor = cor.corr()
            cor = cor.loc[:, ['Target_gpcd']]
        if conservation is True:
            cor = cor.corr()
            cor = cor.loc[:, ['Target_gpcd','cons_goal']]
            del cor['cons_goal']

        cor['Target_gpcd']=np.abs(cor['Target_gpcd'])
        cor=cor.sort_values(by=['Target_gpcd'], ascending=False)
        cor=cor.dropna()

        #Selecting highly correlated features
        relevant_features = cor[cor['Target_gpcd'] > cor_threshold]
        CorFeat = list(relevant_features.index)

        CorDF= DFcor[CorFeat]
        cor = np.abs(CorDF.corr())
        cor = cor.mask(np.tril(np.ones(cor.shape)).astype(np.bool))
        #remove colinearity
        cor = cor[cor.columns[cor.max() < colinearity_thresh]]
        CorFeat= cor.columns
        cor = cor.T
        cor = cor[CorFeat]
        
        """print('Remaining features are', CorFeat)"""

        #Set up training and testing data 
        X_train = X_train_features[month][CorFeat].copy()
        y_train = y_train_target[month].copy()

        X_test = X_test_features[month][CorFeat].copy()
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
            
            """print('CV scores = ', scores) """

            # cv results
            cv_results = pd.DataFrame(model_cv.cv_results_)

            #code to select features for final model, tell how many features
            N_feat=cv_results.loc[cv_results['mean_test_score'].idxmax()]
            N_feat=N_feat['param_n_features_to_select']
            
            """print('Number of features to select is ', N_feat)"""
            
            # intermediate model
            n_features_optimal = N_feat

            Int_model = model_type
            Int_model.fit(X_train, y_train)

            rfe = RFE(Int_model, n_features_to_select=n_features_optimal)             
            rfe = rfe.fit(X_train, y_train)

            # make the final model with rfe features

            # tuples of (feature name, whether selected, ranking)
            # note that the 'rank' is > 1 for non-selected features

            Features =list(zip(X_train.columns,rfe.support_,rfe.ranking_))
            FeaturesDF=pd.DataFrame(Features, columns=['Feature', 'Important', 'Score'])
            FeaturesDF = FeaturesDF[FeaturesDF.Score<=1]
            RFE_features = list(FeaturesDF['Feature'])
            
            """print('The final features are ', RFE_features)"""

            #select only RFE features for model training/validation
            X_train = X_train[RFE_features]
            X_test = X_test[RFE_features]

            #plot of selected features to make sure not colinear
            CorDF= X_train.copy()
            CorDF['Target_gpcd']=TrainDF[month]['Target_gpcd']

            Final_model = model_type
            Final_model.fit(X_train, y_train)

        else:

           #Set up training and testing data to have a random non-correlated feature then
            X_train = X_train_features[month].iloc[:, 1].copy() # was [month]['HousingDensity'].copy()
            X_test = X_test_features[month].iloc[:, 1].copy()
            cv_results = 0
            cor = 0

            len1 = len(X_train)
            len2 = len(X_test)

            X_train = np.array(X_train).reshape(len1, 1)
            X_test = np.array(X_test).reshape(len2, 1)

            Final_model = model_type
            Final_model.fit(X_train, y_train)

        # Get training data model performance to tune hyperparameters
        yt_pred = Final_model.predict(X_train)
 
        yt_pred = [0 if x < 0 else x for x in yt_pred]

        O_r2_train = r2_score(y_train, yt_pred)
        O_rmse_train = mean_squared_error(y_train, yt_pred, squared = False)

        # predict X_test
        y_pred = Final_model.predict(X_test)
        y_pred = [0 if x < 0 else x for x in y_pred]
        O_r2_test = r2_score(y_test, y_pred)
        O_rmse_test = mean_squared_error(y_test, y_pred, squared = False)

        #plot the predictions
        PerfDF=pd.DataFrame(list(zip(y_pred, y_test['Target_gpcd'])), columns=['y_pred', 'y_test'])
        PerfDF['y_pred'] = PerfDF['y_pred'].astype(float)

        #Add indoor demands 
        Indoor=['Jan', 'Feb', 'Mar', 'Nov', 'Dec']

        if month in Indoor: ################################################# not running through indoor months in loop- so?
            PerfDF['y_test_tot']=PerfDF['y_test']
            PerfDF['y_pred_tot']=PerfDF['y_pred']
        else:
            PerfDF['y_test_tot']=PerfDF['y_test']+list(I_mean)
            PerfDF['y_pred_tot']=PerfDF['y_pred']+list(Cons_mean)############ problem if cons goal not used- if statement?

        """T_r2 = r2_score(PerfDF['y_test_tot'], PerfDF['y_pred_tot'])
        T_rmse= mean_squared_error(PerfDF['y_test_tot'], PerfDF['y_pred_tot'], 
                                                   #squared = False)

        print('Total R2 is ', T_r2)
        print('Total rmse is ', T_rmse)"""
        
        PerfDF['Year'] = list(TestDF['Jul'].index)
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

        PerfDF=PerfDF.drop(columns=['Year', 'month', 'Day'])
        PerfDF=PerfDF.reset_index()

        params = [snowfeatures, conservation, cor_threshold, colinearity_thresh]
        EstParams = Final_model.get_params()

        return X_test, PerfDF, O_rmse_train, O_r2_train ,O_rmse_test, O_r2_test , params, cv_results, cor , Final_model.coef_

    
    def model_plots(self, PerfDF, cv_results, cor, X_test_RFE, coef, scoring, month, figpath):

        plotmin = PerfDF[['y_pred', 'y_test']].min().min()
        plotmax = PerfDF[['y_pred', 'y_test']].max().max()

        plotmin_tot = PerfDF[['y_pred_tot', 'y_test_tot']].min().min()
        plotmax_tot = PerfDF[['y_pred_tot', 'y_test_tot']].max().max()

        # plotting cv results
        plt.figure(figsize=(12,10))
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

        plt.savefig(figpath + month + '_corMatrix.pdf') ###figpath was 'C:/Users/Ryan/Box/Dissertation/Paper1/Figs/'
        plt.show()

        fig, ax = plt.subplots(3,2, constrained_layout=True) ###########################plt.subplots(3,3, constrained_layout=True)
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
        ax[0,1].set_title('Model Performance') ##'Outdoor Model Performance'
        ax[0,1].plot([plotmin,plotmax],[plotmin,plotmax], color='red', linestyle='--' )
        ax[0,1].spines['bottom'].set_color('black')
        ax[0,1].spines['left'].set_color('black')

        """ax[0,2].scatter(PerfDF['y_test_tot'], PerfDF['y_pred_tot'],color='blue', alpha=0.5)
        ax[0,2].set_ylabel('Predicted')
        ax[0,2].set_xlabel('Observed')
        ax[0,2].set_ylim(plotmin_tot-5,plotmax_tot+5)
        ax[0,2].set_xlim(plotmin_tot-5,plotmax_tot+5)
        ax[0,2].set_title('Indoor and Outdoor \n Model Performance')
        ax[0,2].plot([plotmin_tot,plotmax_tot],[plotmin_tot,plotmax_tot], color='red', linestyle='--' )
        ax[0,2].spines['bottom'].set_color('black')
        ax[0,2].spines['left'].set_color('black')"""

        gs = ax[1, 1].get_gridspec()
        # remove the underlying axes
        ax[1,0].remove()
        ax[1,1].remove()
        #####ax[1,2].remove()

        PerfDF['Error'] = (PerfDF['y_pred']-PerfDF['y_test'])
        """axbig1 = fig.add_subplot(gs[1, :])
        axbig1.set_title(month+' Outdoor demand Timeline Evaluation')
        axbig1.axhline(y = 0 , color = 'black')
        #axbig1.bar(PerfDF.index, PerfDF['y_pred'], color='orange', label='Predicted')
        Error1 = axbig1.bar(PerfDF.index, PerfDF['Error'],color='blue', label='Prediction Error')
        axbig1.set_xlabel('Year')
        axbig1.set_ylabel('Error (GPCD)')
        axbig1.spines['bottom'].set_color('black')
        axbig1.spines['left'].set_color('black')
        self.gradientbars_sliced(Error1)"""

        gs2 = ax[2, 1].get_gridspec()
        # remove the underlying axes
        ax[2,0].remove()
        ax[2,1].remove()
        #####ax[2,2].remove()

        #create error value
        PerfDF['Error_tot'] = (PerfDF['y_pred_tot']-PerfDF['y_test_tot'])

        axbig2 = fig.add_subplot(gs2[2, :])
        axbig2.set_title(month+' Total Demand Error Timeline Evaluation')
        Error2 = axbig2.bar(PerfDF.index, PerfDF['Error_tot'], color='blue', label='Predicted')
        axbig2.axhline(y = 0 , color = 'black')
        axbig2.set_xlabel('Year')
        axbig2.set_ylabel('Error (GPCD)')
        axbig2.spines['bottom'].set_color('black')
        axbig2.spines['left'].set_color('black')
        self.gradientbars_sliced(Error2)

        fig.suptitle(month+ ' Evaluation', size = 16)
        fig.savefig(figpath + month + '_demand.pdf')    

        """O_r2 = r2_score(PerfDF['y_test'],PerfDF['y_pred'])
        O_mae= mean_absolute_error(PerfDF['y_test'],PerfDF['y_pred'])
        #O_mape= mean_absolute_percentage_error(PerfDF['y_test'],PerfDF['y_pred'])

        T_r2 = r2_score(PerfDF['y_test_tot'],PerfDF['y_pred_tot'])
        T_mae= mean_absolute_error(PerfDF['y_test_tot'],PerfDF['y_pred_tot'])
        #T_mape= mean_absolute_percentage_error(PerfDF['y_test_tot'],PerfDF['y_pred_tot'])

        print('The outdoor Demand prediction R2 is ', O_r2)
        print('The Total Demand prediction R2 is ', T_r2)
        print('The Total Demand prediction MAE is ', T_mae)
        #print('The Total Demand prediction MAPE is ', T_mape, '%')  """
        
        O_rmse= mean_squared_error(PerfDF['y_test'],PerfDF['y_pred'], squared = False)
        T_rmse= mean_squared_error(PerfDF['y_test_tot'],PerfDF['y_pred_tot'], squared = False)
        print('The outdoor Demand prediction RMSE is ', O_rmse)
        print('The Total Demand prediction RMSE is ', T_rmse)        
        
        print('The final set of features for ' + month + ' are', list(X_test_RFE.columns))
        print('The coefficients for each feature are', coef)   
        
        
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
    
    def Mean_gpcd(self, df, YR):
       
        #load necessary variables from class structure
        HistoricalDF = self.Historical
        TrainDF = self.Train
        
        """for i in (TrainDF):
            TrainDF[i]['Population'] = HistoricalDF[i]['Population']"""
        
        #create df for storing calculated data
        Pred_Obs=pd.DataFrame()
       
        for i in df:
            Pred_Obs=Pred_Obs.append(df[i])
            
        Pred_Obs=Pred_Obs.sort_index()
            
        #making one row for each month throughout a year
        Pred_Obs['Date']= pd.date_range(start=datetime.datetime(YR,1,1),end=datetime.datetime(YR + 1,1,1), freq='M') #16 was 18
        Pred_Obs.index = Pred_Obs['Date']
        del Pred_Obs['Date']
        
        cols = ['y_test_tot', 'y_pred_tot']
        monthorder = ['Jan', 'Feb' , 'Mar', 'Apr', 'May', 'Jun' , 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        Pred_Obs = Pred_Obs[cols]

        """#input population into DF to calculate total water demands.
        Population = []"""
        UR_gpcd = []
        
        PredDF = copy.deepcopy(HistoricalDF)

        #Calculating the average gpcd for each month based on historical data
        for i in monthorder:
            TrainDF[i] = copy.deepcopy(HistoricalDF[i].loc[:2010])
            PredDF[i]['UR_gpcd'] = np.mean(TrainDF[i]['Target_gpcd'])
            """Population.append(np.round(PredDF[i].loc[2015]['Population'],0)) #added .loc[2015]"""

        # adding mean values to storage dataframe correctly
        UR=pd.DataFrame()
        for i in PredDF:
            PredDF[i]= pd.DataFrame(PredDF[i]['UR_gpcd'])
            PredDF[i]=PredDF[i].reset_index()
            PredDF[i]['M'] = datetime.datetime.strptime(i, "%b").month
            PredDF[i]['D'] = 1
            PredDF[i]['Date'] = pd.to_datetime(PredDF[i].Year*10000+PredDF[i].M*100+PredDF[i].D,format='%Y%m%d')+MonthEnd(1)
            PredDF[i].index = PredDF[i].Date
            PredDF[i]= PredDF[i].drop(columns = ['M', 'D', 'Date', 'Year'])
            UR = UR.append(PredDF[i])

        UR=UR.sort_index()
        Pred_Obs['UR_gpcd'] = UR['UR_gpcd']
        
        """Population = np.sort(np.array(Population).reshape(12,)) #changed 36 to 12

        #place in to prediction df
        Pred_Obs['Population'] = Population  

        Now we can form some acre-feet predictions.
        gpcd=['y_test_tot','y_pred_tot','UR_gpcd']
        for i in gpcd:
            Pred_Obs[i+str('_AF')] = Pred_Obs[i]*Pred_Obs['Population']*9.33454e-5 


        remcol=['y_test_tot','y_pred_tot','UR_gpcd'] # 'Population'
        Ann_Eval = Pred_Obs.drop(columns = remcol).copy()
        Ann_Eval = Ann_Eval.resample('Y').sum()"""
        
        #adding average gpcds to the val dataframes using month index        
        Pred_Obs['Month'] = monthorder
        Pred_Obs['Date'] = Pred_Obs.index
        Pred_Obs.index = Pred_Obs['Month']

        for i in df:
            df[i]['Avg_gpcd'] = Pred_Obs.loc[i,'UR_gpcd']

        return df
    
    
    def Seperate_Data(self, df):

        #Index by year for slicing
        for i in df:
            df[i]['Date'] = df[i].index
            df[i].index = df[i]['Year']
        
        drought = {}
        surplus = {}
        avg = {}

        #slice by year to split dataframe
        for i in df:
            drought[i] = df[i].loc[[self.DroughtYR],:]
            surplus[i] = df[i].loc[[self.SurplusYR],:]
            avg[i] = df[i].loc[[self.AvgYR],:]
        
        #reindex by date
        for df in [drought, surplus, avg]:
            for month in df:
                df[month].index = df[i]['Date']
                
        return drought, surplus, avg    
    
    
    #Puts all of the predictions together
    def Demand_Forecast(self, prediction_dictionary, observed, figpath):
        
        FinalDF=pd.DataFrame()

        """print('yes')"""
        for i in prediction_dictionary:
            FinalDF=FinalDF.append(prediction_dictionary[i])
            
        FinalDF=FinalDF.sort_index()

        #adjust date range to improve figure
        FinalDF['Date']= pd.date_range(start=datetime.datetime(2015,1,1),end=datetime.datetime(2015,12,1), freq='MS')   #was to 2017,12,1
        FinalDF.index = FinalDF['Date']
        del FinalDF['Date']

        pred = 'y_pred_tot'
        test = 'y_test_tot'
        mean = 'Avg_gpcd'
        
        #creating barplot
        if observed is True:
            plotmin_tot = FinalDF[[pred, test]].min().min()
            plotmax_tot = FinalDF[[pred, test]].max().max()
        else:
            plotmin_tot = FinalDF[[pred, mean]].min().min()
            plotmax_tot = FinalDF[[pred, mean]].max().max()

        Xplotmin = FinalDF.index[0]-np.timedelta64(20, 'D')
        Xplotmax = FinalDF.index[-1]+np.timedelta64(33, 'D')

        plt.rc_context({ 'xtick.color':'black'})
        fig, ax = plt.subplots(1,5, constrained_layout=True)
        fig.set_size_inches(9,3.5)

        gs2 = ax[0].get_gridspec()
        # remove the underlying axes
        ax[0].remove()
        ax[1].remove()
        ax[2].remove()
        axbig = fig.add_subplot(gs2[:3])
        axbig.bar(FinalDF.index-np.timedelta64(7, 'D'), FinalDF[pred], color='blue',
                  label= 'CSD-WDM' ,width = 15,  align="center")   
        
        if observed is True:
            axbig.bar(FinalDF.index+np.timedelta64(8, 'D'), FinalDF[test], color='orange',
                  label= 'Observed',width = 15,  align="center")
        else:
            axbig.bar(FinalDF.index+np.timedelta64(8, 'D'), FinalDF[mean], color='red',
                  label= 'Historical Average',width = 15,  align="center")
        
        """axbig.set_xlabel('Surplus                                       Drought                                      Average \n \n Supply Scenario')"""
        axbig.set_ylim(plotmin_tot*.5,plotmax_tot*1.3) # was - .9
        axbig.set_xlim(Xplotmin, Xplotmax)
        axbig.set_ylabel('Demand (GPCD)')
        axbig.legend(loc = 'upper left', facecolor = 'white')
        axbig.set_facecolor("white")
        axbig.spines['bottom'].set_color('black')
        axbig.spines['left'].set_color('black')
        axbig.tick_params(axis='both', which='both', length=5, color='red')
        axbig.xaxis.set_major_locator(mdates.MonthLocator())

        # Get only the month to show in the x-axis:
        axbig.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

        axbig.annotate('A.', (FinalDF.index[-1], 650), size = 14)
        xticks = axbig.xaxis.get_major_ticks()
        months = [0,5,10,12,17,22,24,29,34]

        xticks = axbig.xaxis.get_major_ticks()
        for i,tick in enumerate(xticks):
            if i not in months:
                tick.label1.set_visible(False)

        ax[3].remove()
        ax[4].remove()
        
        #creating subplot if using observed data
        if observed is True:
            axbig2 = fig.add_subplot(gs2[3:])
            fig.tight_layout()
            axbig2.scatter(FinalDF[test], FinalDF[pred],color='blue', alpha=0.5)
            axbig2.set_ylabel('Predicted (GPCD)' )
            axbig2.set_xlabel('Observed (GPCD)')
            axbig2.set_ylim(plotmin_tot*.95,plotmax_tot*1.2)
            axbig2.set_xlim(plotmin_tot*.95,plotmax_tot*1.2)
            axbig2.plot([plotmin_tot,plotmax_tot],[plotmin_tot,plotmax_tot], color='black', linestyle='--' )
            axbig2.set_facecolor("white")
            axbig2.spines['bottom'].set_color('black')
            axbig2.spines['left'].set_color('black')
            axbig2.annotate('B.', (600,600), size = 14)
            axbig2.set_xticks(np.arange(100, 601, 100))
            axbig2.set_yticks(np.arange(100, 601, 100))
        
        #calculating and printing error metrics
        if observed is True:
            fig.savefig(figpath +str('CSD_WDM_prediction_and_observed')+'bar.pdf')
            r2 = sklearn.metrics.r2_score(FinalDF[test], FinalDF[pred])
            MAE= sklearn.metrics.mean_absolute_error(FinalDF[test], FinalDF[pred])
            RMSE= sklearn.metrics.mean_squared_error(FinalDF[test], FinalDF[pred], squared = False)
            MAPE=np.mean(np.abs((FinalDF[test]- FinalDF[pred])/FinalDF[test])*100)
            
            print('Total R2 is ', r2)
            print('Total MAE is ', MAE)
            print('Total RMSE is ', RMSE)
            print('Total MAPE is ', MAPE)
        
        else:
            fig.savefig(figpath +str('CSD_WDM_prediction_and_historical')+'bar.pdf')
            r2 = sklearn.metrics.r2_score(FinalDF[mean], FinalDF[pred])
            MAE= sklearn.metrics.mean_absolute_error(FinalDF[mean], FinalDF[pred])
            RMSE= sklearn.metrics.mean_squared_error(FinalDF[mean], FinalDF[pred], squared = False)
            MAPE=np.mean(np.abs((FinalDF[mean]- FinalDF[pred])/FinalDF[mean])*100)
            
            print('R2 from historical is ', r2)
            print('Mean absolute difference from historical is', MAE)
            print('RMSE from historical is', RMSE)
            print('Mean absolute percentage difference from historical is ', MAPE)
        
        FinalDF['Date']= pd.date_range(start=datetime.datetime(2015,1,1),end=datetime.datetime(2016,1,1), freq='M') #16 was 18
        FinalDF.index = FinalDF['Date']
        del FinalDF['Date']
        return FinalDF
    
 