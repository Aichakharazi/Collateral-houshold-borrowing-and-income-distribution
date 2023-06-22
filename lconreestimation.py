# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:37:18 2020

@author: Aicha-PC
"""







import numpy 
import numpy as np
import pandas

import pandas as pd
#%%

import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

import plotly
import plotly.graph_objs as go
#%%
import addfips


from linearmodels import PanelOLS
from linearmodels import RandomEffects

from linearmodels.panel import PooledOLS
import statsmodels
import statsmodels.api as sm

from linearmodels.panel import compare


#%%

import seaborn 
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api

from pandas import DataFrame
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

                             
                                  
import seaborn 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pandas import DataFrame
import statsmodels.formula.api as smf
#from statsmodels.iolib.summary2 import summary_col

import regex as re
import spacy
import csv


#%%
#import numpy as np
#import pandas as pd
#import string
import linearmodels as lm
#%%


import plotly
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
#%%
import numpy as np

import matplotlib.style as style

from linearmodels.iv import IV2SLS


#%%

convertcountytostate8income = pd.read_csv('pericnomeannualstate.csv', sep=';',error_bad_lines=False)

#%%countyus9218perdebt

convertcountytostate8consumption = pd.read_csv('dataannualconsumptionstate.csv', sep=';',error_bad_lines=False)

#%%


convertcountytostate8consumption['wealthvariationapprox'] = convertcountytostate8income.perincome - convertcountytostate8consumption.consump



convertcountytostate8consumption['perincome'] = convertcountytostate8income.perincome

#%%



us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Palau': 'PW',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}



# thank you to @kinghelix and @trevormarburger for this idea
#abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))

abbrev_us_state = {state: abbrev for abbrev, state in us_state_abbrev.items()}


convertcountytostate8consumption['stalp'] = convertcountytostate8consumption['GeoName'].map(us_state_abbrev)


#%%


convertcountytostate8debt = pd.read_csv('databycountyus9218re.csv', sep=';',error_bad_lines=False)



convertcountytostate8debt['lncon1'] = convertcountytostate8debt.lncon*0.001 + convertcountytostate8debt.lnre*0.001

convertcountytostate8debt['drcon1'] = convertcountytostate8debt.drcon*0.001 + convertcountytostate8debt.drre*0.001

convertcountytostate8debt['ntcon1'] = convertcountytostate8debt.ntcon*0.001 + convertcountytostate8debt.ntre*0.001

convertcountytostate8debt['crcon1'] = convertcountytostate8debt.crcon*0.001 + convertcountytostate8debt.crre*0.001


convertcountytostate8debt['lnre'] = convertcountytostate8debt.lnre*0.001
convertcountytostate8debt['lncon'] = convertcountytostate8debt.lncon*0.001 

convertcountytostate8debt['drcon'] = convertcountytostate8debt.drcon*0.001
 
convertcountytostate8debt['drre'] = convertcountytostate8debt.drre*0.001
 
convertcountytostate8debt['ntcon'] = convertcountytostate8debt.ntcon*0.001  
convertcountytostate8debt['ntre'] = convertcountytostate8debt.ntre*0.001
 
convertcountytostate8debt['crcon'] = convertcountytostate8debt.crcon*0.001  
convertcountytostate8debt['crre'] = convertcountytostate8debt.crre*0.001

subx = convertcountytostate8debt[["A","lncon1", "drcon1","A2","ntcon1","stalp","crcon1","GeoFips","lnre","crre","drre","lncon","drcon","ntcon","crcon" ]]
subx_data = subx.copy()



#%%
subx_data["lncon1"] = pandas.to_numeric(subx_data["lncon1"], errors="coerce")
subx_data["drcon1"] = pandas.to_numeric(subx_data["drcon1"], errors="coerce")
subx_data["A2"] = pandas.to_numeric(subx_data["A2"], errors="coerce")
subx_data["A"] = pandas.to_numeric(subx_data["A"], errors="coerce")
subx_data["ntcon1"] = pandas.to_numeric(subx_data["ntcon1"], errors="coerce")
#subx_data["stalp"] = pandas.to_numeric(subx_data["stalp"], errors="coerce")
subx_data["crcon1"] = pandas.to_numeric(subx_data["crcon1"], errors="coerce")
subx_data["GeoFips"] = pandas.to_numeric(subx_data["GeoFips"], errors="coerce")




# entity and time 
subx_data = subx_data.set_index(['A2'])



#%%


topf = subx_data.groupby(['A2',"stalp"])[["crcon1","ntcon1","drcon1","lncon1","lnre","crre","drre","crcon","ntcon","drcon","lncon"]].sum()

#%%
#topf["Timestate"] = topf.index
topf["Timestate"] = topf.index.to_numpy()


#%%




topf[['years','stalp']] = topf['Timestate'].apply(str).str.split(expand=True)


#%%
topf['years']=topf['years'].str.replace(',','')

topf['years']=topf['years'].str.replace('(','')

topf['stalp']=topf['stalp'].str.replace(',','')

topf['stalp']=topf['stalp'].str.replace(')','')
topf['stalp']=topf['stalp'].str.replace("'","")




topf.reset_index(drop=True, inplace=True)

#%%  topf.dtypes   topf.dtypes  convertcountytostate8consumption.dtypes

topf['years']=topf['years'].astype(int) 
#topf['state']=topf['state'].astype(int) 

#%%

topf=topf.merge(convertcountytostate8consumption, left_on=['years','stalp']
                                , right_on=['years', 'stalp'])

#%%



topf = topf.loc[:, ~topf.columns.str.contains('^Unnamed')]
#%%
topf['stalp2'] = topf['stalp']
topf['years2'] = topf['years']


#%%


#%%


forplotting10 = topf.groupby(['years'])[["lncon1","drcon1","consump","wealthvariationapprox","perincome"]].sum()


forplotting10["Time"] = forplotting10.index
#%%

fig,ax = plt.subplots()
ax.plot(forplotting10.Time, forplotting10.drcon1/1000, color="#88070D", marker="o")
ax.set_xlabel("Time",fontsize=14)
ax.set_ylabel("Uncollectible Individual Loan ($ Millions)",color="#88070D",fontsize=18)
ax2=ax.twinx()
ax2.plot(forplotting10.Time, forplotting10.wealthvariationapprox/1000,color="#081A40", marker="o")
ax2.set_ylabel("Wealth Variation ($ Thousands) ",color="#081A40",fontsize=18)
ax.ticklabel_format(style='plain', scilimits=(0,0), axis='y')
ax2.ticklabel_format(style='plain', scilimits=(0,0), axis='y')        
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
ymin, ymax = ax2.get_xlim()
ax.set_xticks(np.round(np.linspace(ymin, ymax, 15), 2))
plt.show()

# save the plot as a file
#fig.savefig('referssdebt.jpg',
#            format='jpeg',
#            dpi=100,
#            bbox_inches='tight')


#%% ********   crcon1    lnre','crre  'drre'

topf['logwealthvariationapprox']=np.log(topf['wealthvariationapprox'])

topf['loglncon1']=np.log(topf['lncon1'])
topf['logdrcon1']=np.log(topf['drcon1'])
topf['logcrcon1']=np.log(topf['crcon1'])



topf['loglncon']=np.log(topf['lncon'])
topf['logdrcon']=np.log(topf['drcon'])
topf['logcrcon']=np.log(topf['crcon'])



topf['loglnre']=np.log(topf['lnre'])
topf['logcrre']=np.log(topf['crre'])
topf['logdrre']=np.log(topf['drre'])


#%% top per capita income abover the US national average


df1=topf.loc[topf['GeoName'].isin(['District of Columbia','Connecticut','New Jersey','Massachusetts','Maryland','New Hampshire','Virginia','New York','North Dakota','Alaska','Minnesota','Colorado','Washington','Rhode Island',	'Delaware',	'California','Illinois',	'Hawaii',	
'Wyoming','Pennsylvania','Vermont'])]

#%% bottom per capita income below the US national average
df2=topf.loc[topf['GeoName'].isin(['Iowa',	'Wisconsin','Maine','Kansas','Oregon','Nebraska','Texas','South Dakota',	'Ohio',	'Michigan','Florida',	
'Missouri',	'Montana','North Carolina','Nevada','Arizona','Georgia','Oklahoma',	
'Indiana','Tennessee','Utah','Louisiana','South Carolina','Idaho','Kentucky',	
'New Mexico','Alabama','Arkansas','West Virginia','Mississippi'])]	


#'Guam',	
#☻'Puerto Rico',	
#'Northern Mariana Islands',	
#'American Samoa'	
#'U.S. Virgin Islands',	


#%%



#%%

df1xample = df1
#df1xample = df1xample.rename(columns={"lncon": "Loans ($ Millions)","drcon":"Charge-off ($ Millions)"})
    
df2xample = df2
#df2xample = df2xample.rename(columns={"lncon": "Loans ($ Millions)","drcon":"Charge-off ($ Millions)"})
    


#%%



df1 = df1.set_index(['stalp2','years2'])



#%%
exog_vars = ['lncon1'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%   required by referee: errors clustered at state level cluster='time' 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1 = cons.fit()
print(two_res1)

#%% cov_type="clustered", clusters= df1['TimeState']   cov_type='clustered', cluster_entity=True
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1clust = cons.fit(cov_type="clustered", clusters= df1.Timestate)
print(two_res1clust)


#%%  cov_type="clustered", clusters=data.occupation


#%% wls

model_fitted_y = two_res1.fitted_values

df1["weight_1"] = model_fitted_y  
df1["weight_1"] = df1["weight_1"]**-1

model_wls_1 = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True, weights=df1.weight_1) 
two_res1wls = model_wls_1.fit() 
#print(mod_res_1.summary())
print(two_res1wls)




#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2 = cons.fit()
print(two_res2)

#%% cov_type="clustered", clusters= df1['Timestate']

print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2clust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])



print(two_res2clust)

#%% wls

model_fitted_y = two_res2.fitted_values

df1["weight_1"] = model_fitted_y  
df1["weight_1"] = df1["weight_1"]**-1

model_wls_1 = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True, weights=df1.weight_1) 
two_res2wls = model_wls_1.fit() 
#print(mod_res_1.summary())
print(two_res2wls)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3 = cons.fit()
print(two_res3)

#%%cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3clust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res3clust)
#%% wls

model_fitted_y = two_res3.fitted_values

df1["weight_1"] = model_fitted_y  
df1["weight_1"] = df1["weight_1"]**-1

model_wls_1 = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True, weights=df1.weight_1) 
two_res3wls = model_wls_1.fit() 
#print(mod_res_1.summary())
print(two_res3wls)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4 = cons.fit()
print(two_res4)

#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4clust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res4clust)

#%% THE SAME   but it s not significant
exog_vars = ['lncon1','crcon1'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1c = cons.fit()
print(two_res1c)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1cclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res1cclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2c = cons.fit()
print(two_res2c)


#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2cclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res2cclust)


#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3c = cons.fit()
print(two_res3c)

#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3cclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res3cclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4c = cons.fit()
print(two_res4c)

#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4cclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res4cclust)

#%%
exog_vars = ['lncon1','drcon1'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1cd = cons.fit()
print(two_res1cd)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1cdclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res1cdclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2cd = cons.fit()
print(two_res2cd)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2cdclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res2cdclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3cd = cons.fit()
print(two_res3cd)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3cdclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res3cdclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4cd = cons.fit()
print(two_res4cd)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4cdclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res4cdclust)
#%%

df2 = df2.set_index(['stalp2','years2'])
#%%

exog_vars = ['lncon1','crcon1']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11 = cons.fit()
print(two_res11)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11clust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res11clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22 = cons.fit()
print(two_res22)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22clust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res22clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33 = cons.fit()
print(two_res33)
#%%cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33clust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res33clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44 = cons.fit()
print(two_res44)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44clust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res44clust)
#%%

exog_vars = ['lncon1','drcon1']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11drc = cons.fit()
print(two_res11drc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11drcclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res11drcclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22drc = cons.fit()
print(two_res22drc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22drcclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res22drcclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33drc = cons.fit()
print(two_res33drc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33drcclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res33drcclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44drc = cons.fit()
print(two_res44drc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44drcclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res44drcclust)
#%% THE SAME 
exog_vars = ['lncon1']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11woc = cons.fit()
print(two_res11woc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11wocclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res11wocclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22woc = cons.fit()
print(two_res22woc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22wocclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res22wocclust)



#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33woc = cons.fit()
print(two_res33woc)

#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33wocclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res33wocclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44woc = cons.fit()
print(two_res44woc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44wocclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res44wocclust)

#%%
from linearmodels.iv import IV2SLS
df1ivreg = df1.copy()
df1ivreg['const'] = 1
df110 = IV2SLS(dependent=df1ivreg['wealthvariationapprox'],
            exog=df1ivreg['const'],
            endog=df1ivreg['lncon1'],
            instruments=df1ivreg['drcon1']).fit(cov_type='unadjusted')

print(df110.summary)
#%%
df2ivreg = df2.copy()
df2ivreg['const'] = 1
df220 = IV2SLS(dependent=df2ivreg['wealthvariationapprox'],
            exog=df2ivreg['const'],
            endog=df2ivreg['lncon1'],
            instruments=df2ivreg['drcon1']).fit(cov_type='unadjusted')

print(df220.summary)
	
#%% iv with fe
df1ivregfe = df1.copy()
df1ivregfe['const'] = 1
df1ivregfe_fsivfe = sm.OLS(df1ivregfe['lncon1'],
                    df1ivregfe[['const', 'drcon1']],
                    missing='drop').fit()
print(df1ivregfe_fsivfe.summary())

#%%
df1ivregfe['predicted_indivloan'] = df1ivregfe_fsivfe.predict()
#%%
exog_vars = ['predicted_indivloan','crcon1']
exog = sm.add_constant(df1ivregfe[exog_vars])
#%%
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
df1ivregfe_ss2ivfe1 = modiv.fit()
print(df1ivregfe_ss2ivfe1)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
df1ivregfe_ss2ivfe2 = modiv.fit()
print(df1ivregfe_ss2ivfe2)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
df1ivregfe_ss2ivfe3 = modiv.fit()
print(df1ivregfe_ss2ivfe3)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
df1ivregfe_ss2ivfe4 = modiv.fit()
print(df1ivregfe_ss2ivfe4)

#%%
exog_vars = ['predicted_indivloan']
exog = sm.add_constant(df1ivregfe[exog_vars])
#%%
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
df1ivregfe_ss2ivfe1woc = modiv.fit()
print(df1ivregfe_ss2ivfe1woc)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
df1ivregfe_ss2ivfe2woc = modiv.fit()
print(df1ivregfe_ss2ivfe2woc)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
df1ivregfe_ss2ivfe3woc = modiv.fit()
print(df1ivregfe_ss2ivfe3woc)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
df1ivregfe_ss2ivfe4woc = modiv.fit()
print(df1ivregfe_ss2ivfe4woc)





#%%
df2ivregfe = df2.copy()
df2ivregfe['const'] = 1
df2ivregfe_fsivfe = sm.OLS(df2ivregfe['lncon1'],
                    df2ivregfe[['const', 'drcon1']],
                    missing='drop').fit()
print(df2ivregfe_fsivfe.summary())

#%%
df2ivregfe['predicted_indivloan'] = df2ivregfe_fsivfe.predict()
#%%
exog_vars = ['predicted_indivloan','crcon1']
exog = sm.add_constant(df2ivregfe[exog_vars])
#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
df2ivregfe_ss2ivfe1 = modiv.fit()
print(df2ivregfe_ss2ivfe1)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
df2ivregfe_ss2ivfe2 = modiv.fit()
print(df2ivregfe_ss2ivfe2)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
df2ivregfe_ss2ivfe3 = modiv.fit()
print(df2ivregfe_ss2ivfe3)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
df2ivregfe_ss2ivfe4 = modiv.fit()
print(df2ivregfe_ss2ivfe4)

#%% the same 
exog_vars = ['predicted_indivloan']
exog = sm.add_constant(df2ivregfe[exog_vars])
#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
df2ivregfe_ss2ivfe1woc = modiv.fit()
print(df2ivregfe_ss2ivfe1woc)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
df2ivregfe_ss2ivfe2woc = modiv.fit()
print(df2ivregfe_ss2ivfe2woc)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
df2ivregfe_ss2ivfe3woc = modiv.fit()
print(df2ivregfe_ss2ivfe3woc)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
df2ivregfe_ss2ivfe4woc = modiv.fit()
print(df2ivregfe_ss2ivfe4woc)





#%%  loggggggggg

exog_vars = ['loglncon1'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1clustlog = cons.fit(cov_type="clustered", clusters= df1.Timestate)
print(two_res1clustlog)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2clustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res2clustlog)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3clustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res3clustlog)

#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4clustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res4clustlog)
#%%
exog_vars = ['loglncon1','logcrcon1'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1cclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res1cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2cclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res2cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3cclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res3cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4cclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'])
print(two_res4cclustlog)




#%%
exog_vars = ['loglncon1','logcrcon1']
exog = sm.add_constant(df2[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11clustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res11clustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22clustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res22clustlog)
#%%cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33clustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res33clustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44clustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res44clustlog)


#%%
exog_vars = ['loglncon1','logdrcon1']
exog = sm.add_constant(df2[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11drcclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res11drcclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22drcclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res22drcclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33drcclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res33drcclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44drcclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res44drcclustlog)
#%%
exog_vars = ['loglncon1']
exog = sm.add_constant(df2[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11wocclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res11wocclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22wocclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res22wocclustlog) 
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33wocclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res33wocclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44wocclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(two_res44wocclustlog)





#%% ols df2 :  two_res22   two_res33  two_res44    iv2sls: df2ivregfe_ss2ivfe4

# iv2sls df1 :  df1ivregfe_ss2ivfe4



import plotly
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
import numpy as np


#%% REPORT ONLY SIGNIFICANT RESULTS 
#%%

from collections import OrderedDict
from linearmodels.iv.results import compare

from linearmodels.iv import IV2SLS

import csv

from linearmodels.iv import IV2SLS

#%%
#from statsmodels.iolib.summary3 import summary_col


from linearmodels.iv.results import compare

#%%

from statsmodels.iolib.summary3 import summary_col

#from statsmodels.iolib.summary2 import summary_col


#%%
#from statsmodels.iolib.summary2 import summary_col


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

# bottom per capita income below the US national average


resultsrefree11 = summary_col([two_res22,two_res44,two_res22woc,two_res44woc,two_res22drc,two_res44drc],stars=True,show='se',float_format='%0.5f') 



dfoutput = open("resultsrefree11ap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

# bottom per capita income below the US national average


resultsrefree11clust = summary_col([two_res22clust,two_res44clust,two_res22wocclust,two_res44wocclust,two_res22drcclust,two_res44drcclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsrefree11ap08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11clust.as_latex())
dfoutput.write(endtex)
dfoutput.close()




#%%
#%% THis one final 12/19/2022

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

# bottom per capita income below the US national average


resultsrefree11clustlog = summary_col([two_res22clustlog,two_res44clustlog,two_res22wocclustlog,two_res44wocclustlog,two_res22drcclustlog,two_res44drcclustlog],stars=True,float_format='%0.5f',show='se') 



dfoutput = open("resultsrefree11ap08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11clustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%  two_res1   over the us average (top)



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


# TOP per capita income ABOVE the US national average


resultsrefree1 = summary_col([two_res1,two_res2,two_res3,two_res4],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsrefree1ap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree1.as_latex())
dfoutput.write(endtex)
dfoutput.close()









#%% clusterre erros


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


# TOP per capita income ABOVE the US national average


resultsrefree1clust = summary_col([two_res1clust,two_res2clust,two_res3clust,two_res4clust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsrefree1ap08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree1clust.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%




beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


# TOP per capita income ABOVE the US national average


resultsrefree1clustlog = summary_col([two_res1clustlog,two_res2clustlog,two_res3clustlog,two_res4clustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsrefree1ap08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree1clustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%



#%% I couldn't report df110



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

# TOP per capita income ABOVE the US national average



resultsdf1ivregfe_ss2ivfe = summary_col([df1ivregfe_ss2ivfe3,df1ivregfe_ss2ivfe4,df1ivregfe_ss2ivfe1woc,df1ivregfe_ss2ivfe2woc,df1ivregfe_ss2ivfe3woc,df1ivregfe_ss2ivfe4woc],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsdf1ivregfe_ss2ivfeap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsdf1ivregfe_ss2ivfe.as_latex())
dfoutput.write(endtex)
dfoutput.close()





#%%




#%%


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

# BOTTOM per capita income BELOW the US national average



resultsdf2ivregfe_ss2ivfe = summary_col([df2ivregfe_ss2ivfe4],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsdf2ivregfe_ss2ivfe.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsdf2ivregfe_ss2ivfe.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%% topf topf['lncon'].quantile(0.5)
#Out[53]: 3397469.5
# topf['lncon1'].quantile(0.5)
#Out[333]: 37611457.0
# : topf['lncon1'].quantile(0.5)
#Out[483]: 37611.456999999995


#%%

overtopfl50 = topf.drop(topf[(topf.lncon1 > 37611.456999999995)].index)
#%%

belowtopfl50 = topf.drop(topf[(topf.lncon1 < 37611.456999999995)].index)

#%%

overtopfl50 = overtopfl50.set_index(['stalp2','years2'])
#%%
exog_vars = ['lncon1'] # ,'crcon'
exog = sm.add_constant(overtopfl50[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
overtwo_res1 = cons.fit()
print(overtwo_res1)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
overtwo_res1clust = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res1clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
overtwo_res2 = cons.fit()
print(overtwo_res2)
#%% cov_type="clustered", clusters= df1['Timestate']  
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
overtwo_res2clust = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'])
print(overtwo_res2clust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
overtwo_res3 = cons.fit()
print(overtwo_res3)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
overtwo_res3clust = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'])
print(overtwo_res3clust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
overtwo_res4 = cons.fit()
print(overtwo_res4)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
overtwo_res4clust = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'])
print(overtwo_res4clust)
#%% the same with c
exog_vars = ['lncon1','crcon1'] # 
exog = sm.add_constant(overtopfl50[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
overtwo_res1c = cons.fit()
print(overtwo_res1c)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
overtwo_res1cclust = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res1cclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
overtwo_res2c = cons.fit()
print(overtwo_res2c)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
overtwo_res2cclust = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res2cclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
overtwo_res3c = cons.fit()
print(overtwo_res3c)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
overtwo_res3cclust = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res3cclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
overtwo_res4c = cons.fit()
print(overtwo_res4c)
#%%  cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
overtwo_res4cclust = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res4cclust)

#%%

belowtopfl50 = belowtopfl50.set_index(['stalp2','years2'])
#%%
exog_vars = ['lncon1']# ,'crcon'
exog = sm.add_constant(belowtopfl50[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
belowtwo_res1 = cons.fit()
print(belowtwo_res1)
#%%  cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
belowtwo_res1clust = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res1clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
belowtwo_res2 = cons.fit()
print(belowtwo_res2)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
belowtwo_res2clust = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res2clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
belowtwo_res3 = cons.fit()
print(belowtwo_res3)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
belowtwo_res3clust = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res3clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
belowtwo_res4 = cons.fit()
print(belowtwo_res4)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
belowtwo_res4clust = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res4clust)

#%%
exog_vars = ['lncon1','crcon1']# 
exog = sm.add_constant(belowtopfl50[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
belowtwo_res1c = cons.fit()
print(belowtwo_res1c)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
belowtwo_res1cclust = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res1cclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
belowtwo_res2c = cons.fit()
print(belowtwo_res2c)

#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
belowtwo_res2cclust = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res2cclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
belowtwo_res3c = cons.fit()
print(belowtwo_res3c)

#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
belowtwo_res3cclust = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res3cclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
belowtwo_res4c = cons.fit()
print(belowtwo_res4c)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
belowtwo_res4cclust = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res4cclust)

#%% IV 
from linearmodels.iv import IV2SLS
overtopfl50ivreg = overtopfl50.copy()
overtopfl50ivreg['const'] = 1
overtopfl50110 = IV2SLS(dependent=overtopfl50ivreg['wealthvariationapprox'],
            exog=overtopfl50ivreg['const'],
            endog=overtopfl50ivreg['lncon1'],
            instruments=overtopfl50ivreg['drcon1']).fit(cov_type='unadjusted')

print(overtopfl50110.summary)
#%%
belowtopfl50ivreg = belowtopfl50.copy()
belowtopfl50ivreg['const'] = 1
belowtopfl50220 = IV2SLS(dependent=belowtopfl50ivreg['wealthvariationapprox'],
            exog=belowtopfl50ivreg['const'],
            endog=belowtopfl50ivreg['lncon1'],
            instruments=belowtopfl50ivreg['drcon1']).fit(cov_type='unadjusted')

print(belowtopfl50220.summary)

#%% IV WITH FE 
#%% iv with fe
overtopfl50ivregfe = overtopfl50.copy()
overtopfl50ivregfe['const'] = 1
overtopfl50ivregfe_fsivfe = sm.OLS(overtopfl50ivregfe['lncon1'],
                    overtopfl50ivregfe[['const', 'drcon1']],
                    missing='drop').fit()
print(overtopfl50ivregfe_fsivfe.summary())

#%%
overtopfl50ivregfe['predicted_indivloan'] = overtopfl50ivregfe_fsivfe.predict()
#%%
exog_vars = ['predicted_indivloan']# ,'crcon'
exog = sm.add_constant(overtopfl50ivregfe[exog_vars])
#%%
modiv =  PanelOLS(overtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
overtopfl50ivregfe_ss2ivfe1 = modiv.fit()
print(overtopfl50ivregfe_ss2ivfe1)

#%%  
modiv =  PanelOLS(overtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
overtopfl50ivregfe_ss2ivfe2 = modiv.fit()
print(overtopfl50ivregfe_ss2ivfe2)

#%%  
modiv =  PanelOLS(overtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
overtopfl50ivregfe_ss2ivfe3 = modiv.fit()
print(overtopfl50ivregfe_ss2ivfe3)

#%%  
modiv =  PanelOLS(overtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
overtopfl50ivregfe_ss2ivfe4 = modiv.fit()
print(overtopfl50ivregfe_ss2ivfe4)

#%% THE SAME WITH C
exog_vars = ['predicted_indivloan','crcon1']# 
exog = sm.add_constant(overtopfl50ivregfe[exog_vars])
#%%
modiv =  PanelOLS(overtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
overtopfl50ivregfe_ss2ivfe1c = modiv.fit()
print(overtopfl50ivregfe_ss2ivfe1c)

#%%  
modiv =  PanelOLS(overtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
overtopfl50ivregfe_ss2ivfe2c = modiv.fit()
print(overtopfl50ivregfe_ss2ivfe2c)

#%%  
modiv =  PanelOLS(overtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
overtopfl50ivregfe_ss2ivfe3c = modiv.fit()
print(overtopfl50ivregfe_ss2ivfe3c)

#%%  
modiv =  PanelOLS(overtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
overtopfl50ivregfe_ss2ivfe4c = modiv.fit()
print(overtopfl50ivregfe_ss2ivfe4c)




#%%
belowtopfl50ivregfe = belowtopfl50.copy()
belowtopfl50ivregfe['const'] = 1
belowtopfl50ivregfe_fsivfe = sm.OLS(belowtopfl50ivregfe['lncon1'],
                    belowtopfl50ivregfe[['const', 'drcon1']],
                    missing='drop').fit()
print(belowtopfl50ivregfe_fsivfe.summary())

#%%
belowtopfl50ivregfe['predicted_indivloan'] = belowtopfl50ivregfe_fsivfe.predict()
#%%
exog_vars = ['predicted_indivloan'] # ,'crcon'
exog = sm.add_constant(belowtopfl50ivregfe[exog_vars])
#%%  
modiv =  PanelOLS(belowtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
belowtopfl50ivregfe_ss2ivfe1 = modiv.fit()
print(belowtopfl50ivregfe_ss2ivfe1)

#%%  
modiv =  PanelOLS(belowtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
belowtopfl50ivregfe_ss2ivfe2 = modiv.fit()
print(belowtopfl50ivregfe_ss2ivfe2)

#%%  
modiv =  PanelOLS(belowtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
belowtopfl50ivregfe_ss2ivfe3 = modiv.fit()
print(belowtopfl50ivregfe_ss2ivfe3)

#%%  
modiv =  PanelOLS(belowtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
belowtopfl50ivregfe_ss2ivfe4 = modiv.fit()
print(belowtopfl50ivregfe_ss2ivfe4)


#%% THE SAME WITH C
exog_vars = ['predicted_indivloan','crcon1'] # 
exog = sm.add_constant(belowtopfl50ivregfe[exog_vars])
#%%  
modiv =  PanelOLS(belowtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
belowtopfl50ivregfe_ss2ivfe1c = modiv.fit()
print(belowtopfl50ivregfe_ss2ivfe1c)

#%%  
modiv =  PanelOLS(belowtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
belowtopfl50ivregfe_ss2ivfe2c = modiv.fit()
print(belowtopfl50ivregfe_ss2ivfe2c)

#%%  
modiv =  PanelOLS(belowtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
belowtopfl50ivregfe_ss2ivfe3c = modiv.fit()
print(belowtopfl50ivregfe_ss2ivfe3c)

#%%  
modiv =  PanelOLS(belowtopfl50ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
belowtopfl50ivregfe_ss2ivfe4c = modiv.fit()
print(belowtopfl50ivregfe_ss2ivfe4c)



#%%






#%% loggggggggg
exog_vars = ['loglncon1'] # ,'crcon'
exog = sm.add_constant(overtopfl50[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
overtwo_res1clustlog = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res1clustlog)
#%% cov_type="clustered", clusters= df1['Timestate']  
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
overtwo_res2clustlog = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'])
print(overtwo_res2clustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
overtwo_res3clustlog = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'])
print(overtwo_res3clustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
overtwo_res4clustlog = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'])
print(overtwo_res4clustlog)
#%% the same with c
exog_vars = ['loglncon1','logcrcon1'] # 
exog = sm.add_constant(overtopfl50[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
overtwo_res1cclustlog = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res1cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
overtwo_res2cclustlog = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res2cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
overtwo_res3cclustlog = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res3cclustlog)
#%%  cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(overtopfl50.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
overtwo_res4cclustlog = cons.fit(cov_type="clustered", clusters= overtopfl50['Timestate'] )
print(overtwo_res4cclustlog)
#%%
#belowtopfl50 = belowtopfl50.set_index(['stalp2','years2'])
#%%
exog_vars = ['loglncon1']# ,'crcon'
exog = sm.add_constant(belowtopfl50[exog_vars])
#%%  cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
belowtwo_res1clustlog = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res1clustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
belowtwo_res2clustlog = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res2clustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
belowtwo_res3clustlog = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res3clustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
belowtwo_res4clustlog = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res4clustlog)

#%%
exog_vars = ['loglncon1','logcrcon1']# 
exog = sm.add_constant(belowtopfl50[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
belowtwo_res1cclustlog = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res1cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
belowtwo_res2cclustlog = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res2cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
belowtwo_res3cclustlog = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res3cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(belowtopfl50.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
belowtwo_res4cclustlog = cons.fit(cov_type="clustered", clusters= belowtopfl50['Timestate'])
print(belowtwo_res4cclustlog)






#%%


#overtwo_res1
#overtwo_res2
#overtwo_res3
#overtwo_res4



#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsovertwo_res = summary_col([overtwo_res2,overtwo_res3,overtwo_res4,overtwo_res2c,overtwo_res3c,overtwo_res4c],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsovertwo_res.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsovertwo_res.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%% 
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsovertwo_resclust = summary_col([overtwo_res2clust,overtwo_res3clust,overtwo_res4clust,overtwo_res2cclust,overtwo_res3cclust,overtwo_res4cclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsovertwo_resclust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsovertwo_resclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%
#%% 
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsovertwo_resclustlog = summary_col([overtwo_res2clustlog,overtwo_res3clustlog,overtwo_res4clustlog,overtwo_res2cclustlog,overtwo_res3cclustlog,overtwo_res4cclustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsovertwo_resclustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsovertwo_resclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%


#belowtwo_res1
#belowtwo_res2
#belowtwo_res3
#belowtwo_res4



#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsbelowtwo_res = summary_col([belowtwo_res1,belowtwo_res2,belowtwo_res3,belowtwo_res1c,belowtwo_res3c],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsbelowtwo_res.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsbelowtwo_res.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsbelowtwo_resclust = summary_col([belowtwo_res1clust,belowtwo_res2clust,belowtwo_res3clust,belowtwo_res1cclust,belowtwo_res3cclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsbelowtwo_resclust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsbelowtwo_resclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsbelowtwo_resclustlog = summary_col([belowtwo_res1clustlog,belowtwo_res2clustlog,belowtwo_res3clustlog,belowtwo_res1cclustlog,belowtwo_res3cclustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsbelowtwo_resclustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsbelowtwo_resclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()
#%%



#overtopfl50110



#%%
#belowtopfl50220


#%%
#overtopfl50ivregfe_ss2ivfe1
#overtopfl50ivregfe_ss2ivfe2
#overtopfl50ivregfe_ss2ivfe3
#overtopfl50ivregfe_ss2ivfe4

#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

#overtopfl50ivregfe_ss2ivfe4

resultsovertopfl50ivregfe_ss2ivfe = summary_col([overtopfl50ivregfe_ss2ivfe1,overtopfl50ivregfe_ss2ivfe2,overtopfl50ivregfe_ss2ivfe3,overtopfl50ivregfe_ss2ivfe1c,overtopfl50ivregfe_ss2ivfe2c,overtopfl50ivregfe_ss2ivfe3c],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsovertopfl50ivregfe_ss2ivfe.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsovertopfl50ivregfe_ss2ivfe.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%
#belowtopfl50ivregfe_ss2ivfe1
#belowtopfl50ivregfe_ss2ivfe2
#belowtopfl50ivregfe_ss2ivfe3
#belowtopfl50ivregfe_ss2ivfe4




#%%


#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


#belowtopfl50ivregfe_ss2ivfe2

resultsbelowtopfl50ivregfe_ss2ivfe = summary_col([belowtopfl50ivregfe_ss2ivfe1,belowtopfl50ivregfe_ss2ivfe3,belowtopfl50ivregfe_ss2ivfe4,belowtopfl50ivregfe_ss2ivfe4c],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsbelowtopfl50ivregfe_ss2ivfe.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsbelowtopfl50ivregfe_ss2ivfe.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%




import numpy 
import numpy as np
import pandas

import pandas as pd
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

import plotly
import plotly.graph_objs as go

import addfips


from linearmodels import PanelOLS
from linearmodels import RandomEffects

from linearmodels.panel import PooledOLS
import statsmodels
import statsmodels.api as sm

from linearmodels.panel import compare




import seaborn 
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api

from pandas import DataFrame
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

import regex as re
import spacy
import csv


#import datetime as dt

#from numpy import diag, sqrt
#from pandas import Series, concat
#from scipy import stats
#from statsmodels.iolib.summary import SimpleTable, fmt_2cols, fmt_params
#from linearmodels.compat.statsmodels import Summary
#from linearmodels.iv.results import default_txt_fmt, stub_concat, table_concat
#from linearmodels.utility import (_ModelComparison, _SummaryStr, _str,
#                                  pval_format)
                                  
                                  
import seaborn 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pandas import DataFrame
import statsmodels.formula.api as smf
#from statsmodels.iolib.summary2 import summary_col

import regex as re
import spacy
import csv

#%%


import seaborn as sns


#%% CORRELATION BETWEEN LNCON AND DRCON   *************************************************
#***************************
import matplotlib.style as style
style.use('seaborn-white')        
        


#['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight',
# 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind',
# 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid',
# 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook',
# 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster',
# 'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
# 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2',
# 'tableau-colorblind10', '_classic_test']


df1456example = topf
descriptx_datadf1456 = df1456example.rename(columns={"lncon1": "Loans ($ Millions)","drcon1":"Charge-off ($ Millions)"})
    
#descriptx_datadf1456 = df1456example    
#%%       this one -***********************************************************************
import matplotlib.ticker as ticker


import matplotlib.style as style
style.use('seaborn-white')   

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.jointplot(data=descriptx_datadf1456, x='Loans ($ Millions)', y='Charge-off ($ Millions)', kind='reg', color='#0F95D7')



ax = plt.gca()


ax.get_xaxis().get_major_formatter().set_scientific(False)
ax.get_yaxis().get_major_formatter().set_scientific(False)

#ax.xaxis.set_major_formatter(ticker.EngFormatter())

#ax.yaxis.set_major_formatter(ticker.EngFormatter())

#ax.set(xlim = (-400,200000000))  # 200 Million $


xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
#ax.set(ylim = (-400,8000000))   # 8 Million $
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/1000] #
ax.set_yticklabels(ylabels)





#plt.grid(True)
plt.axis('tight')


plt.xlabel(r'Loans (\$ Millions)',fontsize = 20)
plt.ylabel(r'Charge-off (\$ Millions)',fontsize = 20)


plt.savefig('correlationlncondrconnewrefree.png', bbox_inches='tight',transparent=True)






#%% THIS one ***********************






import matplotlib.style as style
style.use('seaborn-white')  



fig = plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


plt.plot(descriptx_datadf1456['perincome'],descriptx_datadf1456['Loans ($ Millions)'],'r.')
ax = plt.axis()
plt.ticklabel_format(style='plain', axis='y')



ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.set(xlim = (0,2000000))  # 1400 Billion $
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
#ax.set(ylim = (0,18000000))   # 8 Million $
#ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/1000]
#ax.set_yticklabels(ylabels)

ax.set(ylim = (-400,18000000))  # 200 Million $
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/100000]
ax.set_yticklabels(ylabels)




#x =np.linspace(ax[0],ax[1]+0.01)
#plt.plot(x, model.params[0] + model.params[1] * x,'b',lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Personal Income (\$ Billions)',fontsize = 16)
plt.ylabel(r'Household Loans (\$ Millions)',fontsize = 16)
plt.show()
fig.savefig('PERLOANnewreferee.png', bbox_inches='tight')





#%%

descriptx_datadf1456['perincomelog'] = np.log(descriptx_datadf1456['perincome'])


#%%
descriptx_datadf1456['Loanslog'] = np.log(descriptx_datadf1456['Loans ($ Millions)'])


#%%

descriptx_datadf1456['Charge-offlog'] = np.log(descriptx_datadf1456['Charge-off ($ Millions)'])

#%%

import matplotlib.style as style
style.use('seaborn-white')  
#fig = 

plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
sns_fi = sns.jointplot(data=descriptx_datadf1456, x='perincomelog', y='Loanslog', kind='reg', color='#0F95D7')

ax = plt.gca()

plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Log Personal Income',fontsize = 16)
plt.ylabel(r'Log Household Loans',fontsize = 16)
plt.ticklabel_format(style='plain', axis='y')


plt.savefig('PERLOANnewrefereelog.png', bbox_inches='tight')
plt.show()















#%%     I want to keep  2015 - 2018 obsevations
test1 = descriptx_datadf1456.copy()
#%%
test1['years2'] = test1.index
#%%
    
test1 = test1.drop(test1[(test1['years2'] == 1997 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 1998 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 1999 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2000 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2001 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2002 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2003 ) ].index)

test1 = test1.drop(test1[(test1['years2'] == 2004 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2005 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2006 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2007 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2008 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2009 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2010 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2011 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2012 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2013 ) ].index)
test1 = test1.drop(test1[(test1['years2'] == 2014 ) ].index)


#%%
#test1['years222'] = test1['years2']
#test1['years2'] = test1.index
#%%
#test1.reset_index()

#%%
#test1.set_index("years", inplace=True)



#%% 
forplot1518  = test1.copy()

#%%


#test1.loc[pd.Index(["2015","2016","2017","2018"], name="years")]


#%% previous is better


import matplotlib.style as style
style.use('seaborn-white')  



fig = plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(forplot1518['perincome'],forplot1518['Loans ($ Millions)'],'r.')
ax = plt.axis()
plt.ticklabel_format(style='plain', axis='y')


ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.set(xlim = (0,2000000))  # 1400 Billion $
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
#ax.set(ylim = (0,18000000))   # 8 Million $
#ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/1000]
#ax.set_yticklabels(ylabels)

ax.set(ylim = (-400,18000000))  # 200 Million $
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/100000]
ax.set_yticklabels(ylabels)




#x =np.linspace(ax[0],ax[1]+0.01)
#plt.plot(x, model.params[0] + model.params[1] * x,'b',lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Personal Income (\$ Billions)',fontsize = 16)
plt.ylabel(r'Household Loans (\$ Millions)',fontsize = 16)
plt.show()
fig.savefig('PERLOANnewrefereeBETTER.png', bbox_inches='tight')


#%%



import matplotlib.style as style
style.use('seaborn-white')  



plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns_fi = sns.jointplot(data=forplot1518, x='perincomelog',y='Loanslog',kind='reg', color='#0F95D7')
ax = plt.gca()

plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Log Personal Income',fontsize = 16)
plt.ylabel(r'Log Household Loans ',fontsize = 16)
plt.ticklabel_format(style='plain', axis='y')

plt.savefig('PERLOANnewrefereeBETTERlog.png', bbox_inches='tight')
plt.show()














#%%


import matplotlib.style as style
style.use('seaborn-white')  



fig = plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(forplot1518['perincome'],forplot1518['lncon'],'r.')
ax = plt.axis()
plt.ticklabel_format(style='plain', axis='y')



ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.set(xlim = (0,2000000))  # 1400 Billion $
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
#ax.set(ylim = (0,18000000))   # 8 Million $
#ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/1000]
#ax.set_yticklabels(ylabels)

ax.set(ylim = (-400,18000000))  # 200 Million $
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/100000]
ax.set_yticklabels(ylabels)




#x =np.linspace(ax[0],ax[1]+0.01)
#plt.plot(x, model.params[0] + model.params[1] * x,'b',lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Personal Income (\$ Billions)',fontsize = 16)
plt.ylabel(r'Household Loans (\$ Millions)',fontsize = 16)
plt.show()
fig.savefig('PERLOANnewrefereeBETTERLNCONCONLY.png', bbox_inches='tight')



#%%


fig = plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(descriptx_datadf1456['perincome'],descriptx_datadf1456['Charge-off ($ Millions)'],'r.')
ax = plt.axis()
plt.ticklabel_format(style='plain', axis='y')


ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.set(xlim = (0,2000000))  # 1400 Billion $
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)


ax.set(ylim = (-400,430000))   # 4 Million $ after removing outliers
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/100000]
ax.set_yticklabels(ylabels)

#ax = plt.gca()
#ax.tick_params(axis = 'both', which = 'major', labelsize = 24)
#ax.tick_params(axis = 'both', which = 'minor', )



#x =np.linspace(ax[0],ax[1]+0.01)
#plt.plot(x, model.params[0] + model.params[1] * x,'b',lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Personal Income (\$ Billions)',fontsize = 16)
plt.ylabel(r'Loan Charge-off (\$ Millions)',fontsize = 16)
plt.show()
fig.savefig('PERCHAnewreferee.png', bbox_inches='tight')
#%%

plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


sns_fi=sns.jointplot(data= descriptx_datadf1456, x= 'perincomelog',y='Charge-offlog',kind='reg', color='#0F95D7')



ax = plt.gca()
plt.ticklabel_format(style='plain', axis='y')



plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Log Personal Income',fontsize = 16)
plt.ylabel(r'Log Loan Charge-off ',fontsize = 16)
plt.savefig('PERCHAnewrefereeLOG.png', bbox_inches='tight')
plt.show()

#%%


fig = plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(forplot1518['perincome'],forplot1518['Charge-off ($ Millions)'],'r.')
ax = plt.axis()
plt.ticklabel_format(style='plain', axis='y')


ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.set(xlim = (0,2000000))  # 1400 Billion $
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)


ax.set(ylim = (-400,430000))   # 4 Million $ after removing outliers
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/100000]
ax.set_yticklabels(ylabels)

#ax = plt.gca()
#ax.tick_params(axis = 'both', which = 'major', labelsize = 24)
#ax.tick_params(axis = 'both', which = 'minor', )



#x =np.linspace(ax[0],ax[1]+0.01)
#plt.plot(x, model.params[0] + model.params[1] * x,'b',lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Personal Income (\$ Billions)',fontsize = 16)
plt.ylabel(r'Loan Charge-off (\$ Millions)',fontsize = 16)
plt.show()
fig.savefig('PERCHAnewrefereeBETTER.png', bbox_inches='tight')


#%%



fig = plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.jointplot(data=forplot1518, x ='perincomelog',y='Charge-offlog',kind='reg', color='#0F95D7')
ax = plt.axis()


ax = plt.gca()


plt.ticklabel_format(style='plain', axis='y')


#x =np.linspace(ax[0],ax[1]+0.01)
#plt.plot(x, model.params[0] + model.params[1] * x,'b',lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Log Personal Income',fontsize = 16)
plt.ylabel(r'Log Loan Charge-off',fontsize = 16)
plt.savefig('PERCHAnewrefereeBETTERLOG.png', bbox_inches='tight')
plt.show()


#%%

fig = plt.figure() 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot(forplot1518['perincome'],forplot1518['drcon'],'r.')
ax = plt.axis()
plt.ticklabel_format(style='plain', axis='y')


ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

ax.set(xlim = (0,2000000))  # 1400 Billion $
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)


ax.set(ylim = (-400,430000))   # 4 Million $ after removing outliers
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/100000]
ax.set_yticklabels(ylabels)

#ax = plt.gca()
#ax.tick_params(axis = 'both', which = 'major', labelsize = 24)
#ax.tick_params(axis = 'both', which = 'minor', )



#x =np.linspace(ax[0],ax[1]+0.01)
#plt.plot(x, model.params[0] + model.params[1] * x,'b',lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Personal Income (\$ Billions)',fontsize = 16)
plt.ylabel(r'Loan Charge-off (\$ Millions)',fontsize = 16)
plt.show()
fig.savefig('PERCHAnewrefereeBETTERLNCON.png', bbox_inches='tight')

#%%

sns.pairplot(descriptx_datadf1456,  vars=["Loans ($ Millions)", "Charge-off ($ Millions)"], hue="years", height=3, palette="Blues");
plt.ticklabel_format(style='plain', axis='y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


plt.savefig('paiyearloancharreferee.png', bbox_inches='tight')

#%%


import matplotlib.style as style
style.use('seaborn-white')         
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.lmplot(x='Loans ($ Millions)', y='Charge-off ($ Millions)', hue='years', data=descriptx_datadf1456, fit_reg=False, palette="Blues")

ax = plt.gca()

ax.get_xaxis().get_major_formatter().set_scientific(False)

xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/10000]
ax.set_xticklabels(xlabels)

#ax.set(ylim = (-400,430000))  # million $ because we remove outlier

ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/10000]
ax.set_yticklabels(ylabels)

#ax.set(xlim = (-400,430000))  #  Million $


ax.set_xlabel(r"Loans (\$ Millions)",fontsize=16)
ax.set_ylabel(r"Charge-off (\$ Millions)",fontsize=16)

plt.savefig('paiyearloancharnewreferee.png', bbox_inches='tight')

#%%

sns.pairplot(descriptx_datadf1456,  vars=["Loans ($ Millions)", "perincome"], hue="years", height=3, palette="Blues");
plt.ticklabel_format(style='plain', axis='y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.savefig('paiyearloanperincreferee.png', bbox_inches='tight')


#%%

import matplotlib.style as style
style.use('seaborn-white')  
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.lmplot(x='Loans ($ Millions)', y='perincome', hue='years', data=descriptx_datadf1456, fit_reg=False, palette="Blues")
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/100000]
ax.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/1000]
ax.set_yticklabels(ylabels)


ax.set_xlabel(r"Loans (\$ Millions)",fontsize=16)
ax.set_ylabel(r"Personal Income (\$ Billions)",fontsize=16)

plt.savefig('paiyearloanperincnewreferee.png', bbox_inches='tight')


#%%



descriptx_datadf1456 = df1456example.rename(columns={"perincome": "Personal Income ($ Billions)","drcon1":"Charge-off ($ Millions)"})
#%%

import matplotlib.style as style
#fig.set_size_inches(12, 8)
sns.set(font_scale=1.7)
style.use('seaborn-white')  
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.pairplot(descriptx_datadf1456, hue ="years", vars =['Personal Income ($ Billions)'], height=10, palette="Blues")


ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
#plt.xlim([50000, 5000000]) 

#max_diam =max('Personal Income ($ Billions)')
#plt.set(xlim=(0, max_diam))

plt.savefig('newpaiyearperincnewreferee.png', bbox_inches='tight')

#%% 


import matplotlib.style as style
style.use('seaborn-white') 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
 
sns.lmplot(x='Charge-off ($ Millions)', y='Personal Income ($ Billions)', hue='years', data=descriptx_datadf1456, fit_reg=False, palette="Blues")
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)
xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/10000]
ax.set_xticklabels(xlabels)
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/1000]
ax.set_yticklabels(ylabels)


ax.set_xlabel(r"Charge-off (\$ Millions)",fontsize=16)
ax.set_ylabel(r"Personal Income (\$ Billions)",fontsize=16)


plt.savefig('paiyearcharperinconewreferee.png', bbox_inches='tight')
#%%



#%%


import matplotlib.style as style
style.use('seaborn-white')        
        

 

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.jointplot(data=descriptx_datadf1456, x='Personal Income ($ Billions)', y='Charge-off ($ Millions)', kind='reg', color='#0F95D7')


ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

#ax.set(xlim = (-400,200000000))  # 1400 Billion $

xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
#ax.set(ylim = (-400,8000000))   # 8 Million $
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/10000]
ax.set_yticklabels(ylabels)



#plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Personal Income (\$ Billions)',fontsize = 20)
plt.ylabel(r'Charge-off (\$ Millions)',fontsize = 20)


plt.savefig('correlationperincomdrconnewreferee.png', bbox_inches='tight')




     
        

#%% ------------------similar to this 




import matplotlib.style as style
style.use('seaborn-white')        
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
        

 


sns.jointplot(data=descriptx_datadf1456, x='wealthvariationapprox', y='Charge-off ($ Millions)', kind='reg', color='#0F95D7')


ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

#ax.set(xlim = (-400,200000000))  # 1400 Billion $

xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
#ax.set(ylim = (-400,8000000))   # 8 Million $
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/10000]
ax.set_yticklabels(ylabels)



#plt.grid(True)
plt.axis('tight')
plt.xlabel(r'Wealth Variation (\$ Billions)',fontsize = 20)
plt.ylabel(r'Charge-off (\$ Millions)',fontsize = 20)


plt.savefig('correlationwealthdrconnewreferee.png', bbox_inches='tight')




#%%
#df1xample = df1xample.groupby(['A2'])[["wealthvariationapprox", "perincom", "totalpersconsumexpenditurmillionaofd"]].mean()



forplottingdf1xample = df1xample.groupby(['years'])[["consump","wealthvariationapprox","perincome"]].mean()
#%%
forplottingdf1xample["lncon1"] = df1xample.groupby(['years'])[["lncon1"]].sum()


forplottingdf1xample["drcon1"] = df1xample.groupby(['years'])[["drcon1"]].sum()

forplottingdf1xample["Time"] = forplottingdf1xample.index

#%%

forplottingdf2xample = df2xample.groupby(['years'])[["consump","wealthvariationapprox","perincome"]].mean()
#%%

forplottingdf2xample["lncon1"] = df2xample.groupby(['years'])[["lncon1"]].sum()


forplottingdf2xample["drcon1"] = df2xample.groupby(['years'])[["drcon1"]].sum()


forplottingdf2xample["Time"] = forplottingdf2xample.index




#%%

#df1xample["Time"] = df1xample.index
#

#%%

style.use('seaborn-white')         


#%% wealth vs debt 

style.use('seaborn-white')

import numpy as np
import matplotlib.pyplot as plt

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(forplottingdf1xample.Time, forplottingdf1xample.lncon1/1000, color="#88070D", marker="o")
# set x-axis label
ax.set_xlabel("Time",fontsize=18)
# set y-axis label
ax.set_ylabel("Individual Loan ($ Millions)",color="#88070D",fontsize=18)
              
              
#df1xample_graph = df1xample.plot(x = 'years', y = ["wealthvariationapprox", "lncon"], figsize = (12,8),color = colors, 
#                                 label=["Wealth Variation","Individual debt"])

              


              
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(forplottingdf1xample.Time, forplottingdf1xample.wealthvariationapprox/1000,color="#081A40", marker="o")
ax2.set_ylabel("Wealth Variation ($ Billions)",color="#081A40",fontsize=18)

ax.ticklabel_format(style='plain', scilimits=(0,0), axis='y')
ax2.ticklabel_format(style='plain', scilimits=(0,0), axis='y')        
               
#ax2.ticklabel_format(style='plain', axis='x') # this should work but doesn't

#ax2.xaxis.set_minor_locator()
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')


#ymin, ymax = ax2.get_xlim()
#ax.set_xticks(np.round(np.linspace(ymin, ymax, 15), 2))

#ax2.get_xaxis().get_major_formatter().set_scientific(False)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

# save the plot as a file
fig.savefig('twinwealthtop10parcapitaincomereferee.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')

#%%

style.use('seaborn-white')

import numpy as np
import matplotlib.pyplot as plt

# create figure and axis objects with subplots()
fig,ax = plt.subplots()

# make a plot
ax.plot(forplottingdf2xample.Time, forplottingdf2xample.lncon1/1000, color="#88070D", marker="o")
# set x-axis label
ax.set_xlabel("Time",fontsize=18)
# set y-axis label
ax.set_ylabel("Individual Loan (\$ Millions)",color="#88070D",fontsize=18)
              
              
#df1xample_graph = df1xample.plot(x = 'years', y = ["wealthvariationapprox", "lncon"], figsize = (12,8),color = colors, 
#                                 label=["Wealth Variation","Individual debt"])

              


              
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(forplottingdf2xample.Time, forplottingdf2xample.wealthvariationapprox/1000,color="#081A40", marker="o")
ax2.set_ylabel("Wealth Variation (\$ Billions)",color="#081A40",fontsize=18)

ax.ticklabel_format(style='plain', scilimits=(0,0), axis='y')
ax2.ticklabel_format(style='plain', scilimits=(0,0), axis='y')        
               
#ax2.ticklabel_format(style='plain', axis='x') # this should work but doesn't

#ax2.xaxis.set_minor_locator()
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')




#ymin, ymax = ax2.get_xlim()
#ax.set_xticks(np.round(np.linspace(ymin, ymax, 15), 2))

#ax2.get_xaxis().get_major_formatter().set_scientific(False)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

# save the plot as a file
fig.savefig('twinwealthbottom10parcapitaincomereferee.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')
#%%

#topf.perincome.quantile(0.25)
#Out[238]: 63730.174999999996

#topf.perincome.quantile(0.75)
#Out[239]: 367012.025

#topf.perincome.quantile(0.05)
#Out[255]: 27590.124999999996

#topf.perincome.quantile(0.95)
#Out[256]: 1734278.725


#topf.perincome.quantile(0.9)
#Out[257]: 853647.0

#topf.perincome.quantile(0.1)
#Out[258]: 36733.8

#topf.perincome.quantile(0.8)
#Out[259]: 451950.4

#topf.perincome.quantile(0.2)
#Out[260]: 51961.9

#top10perincomstate = topf.drop(topf[(topf.perincome < 162414.25)].index)

top10perincomstate =topf.loc[topf['GeoName'].isin(['District of Columbia','Connecticut','New Jersey','Massachusetts','Maryland','New Hampshire','Virginia','New York','North Dakota','Alaska','Minnesota','Colorado','Washington','Rhode Island',	'Delaware',	'California','Illinois',	'Hawaii',	
'Wyoming','Pennsylvania','Vermont'])]

#%% bottom per capita income below the US national average
bot10perincomstate = topf.loc[topf['GeoName'].isin(['Iowa',	'Wisconsin','Maine','Kansas','Oregon','Nebraska','Texas','South Dakota',	'Ohio',	'Michigan','Florida',	
'Missouri',	'Montana','North Carolina','Nevada','Arizona','Georgia','Oklahoma',	
'Indiana','Tennessee','Utah','Louisiana','South Carolina','Idaho','Kentucky',	
'New Mexico','Alabama','Arkansas','West Virginia','Mississippi'])]	



#bot10perincomstate = topf.drop(topf[(topf.perincome > 162414.25)].index)

#%%

top10perincomstatexample = top10perincomstate.groupby(['years'])[["consump","wealthvariationapprox","perincome"]].mean()
#%%

top10perincomstatexample["lncon1"] = top10perincomstate.groupby(['years'])[["lncon1"]].sum()

top10perincomstatexample["drcon1"] = top10perincomstate.groupby(['years'])[["drcon1"]].sum()


top10perincomstatexample["Time"] = top10perincomstatexample.index

#%%


bot10perincomstatexample = bot10perincomstate.groupby(['years'])[["consump","wealthvariationapprox","perincome"]].mean()
#%%

bot10perincomstatexample["lncon1"] = bot10perincomstate.groupby(['years'])[["lncon1"]].sum()

bot10perincomstatexample["drcon1"] = bot10perincomstate.groupby(['years'])[["drcon1"]].sum()


bot10perincomstatexample["Time"] = bot10perincomstatexample.index

#%%


style.use('seaborn-white')

import numpy as np
import matplotlib.pyplot as plt

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(top10perincomstatexample.Time, top10perincomstatexample.lncon1/1000, color="#88070D", marker="o")
# set x-axis label
ax.set_xlabel("Time",fontsize=18)
# set y-axis label
ax.set_ylabel("Individual Loan ($ Millions)",color="#88070D",fontsize=18)
              
              
#df1xample_graph = df1xample.plot(x = 'years', y = ["wealthvariationapprox", "lncon"], figsize = (12,8),color = colors, 
#                                 label=["Wealth Variation","Individual debt"])

              


              
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(top10perincomstatexample.Time, top10perincomstatexample.wealthvariationapprox/1000,color="#081A40", marker="o")
ax2.set_ylabel("Wealth Variation ($ Billions)",color="#081A40",fontsize=18)

ax.ticklabel_format(style='plain', scilimits=(0,0), axis='y')
ax2.ticklabel_format(style='plain', scilimits=(0,0), axis='y')        
               
#ax2.ticklabel_format(style='plain', axis='x') # this should work but doesn't

#ax2.xaxis.set_minor_locator()
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')


#ymin, ymax = ax2.get_xlim()
#ax.set_xticks(np.round(np.linspace(ymin, ymax, 15), 2))

#ax2.get_xaxis().get_major_formatter().set_scientific(False)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

# save the plot as a file
fig.savefig('twinwealthdebttop10referee.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')



#%%


style.use('seaborn-white')

import numpy as np
import matplotlib.pyplot as plt

# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(bot10perincomstatexample.Time, bot10perincomstatexample.lncon1/1000, color="#88070D", marker="o")
# set x-axis label
ax.set_xlabel("Time",fontsize=18)
# set y-axis label
ax.set_ylabel("Individual Loan ($ Millions)",color="#88070D",fontsize=18)
              
              
#df1xample_graph = df1xample.plot(x = 'years', y = ["wealthvariationapprox", "lncon"], figsize = (12,8),color = colors, 
#                                 label=["Wealth Variation","Individual debt"])

              


              
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(bot10perincomstatexample.Time, bot10perincomstatexample.wealthvariationapprox/1000,color="#081A40", marker="o")
ax2.set_ylabel("Wealth Variation ($ Billions)",color="#081A40",fontsize=18)

ax.ticklabel_format(style='plain', scilimits=(0,0), axis='y')
ax2.ticklabel_format(style='plain', scilimits=(0,0), axis='y')        
               
#ax2.ticklabel_format(style='plain', axis='x') # this should work but doesn't

#ax2.xaxis.set_minor_locator()
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')


#ymin, ymax = ax2.get_xlim()
#ax.set_xticks(np.round(np.linspace(ymin, ymax, 15), 2))

#ax2.get_xaxis().get_major_formatter().set_scientific(False)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.show()

# save the plot as a file
fig.savefig('twinwealthdebtbot10referee.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')




#%%

style.use('seaborn-white')

import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
ax.plot(top10perincomstatexample.Time, top10perincomstatexample.consump/1000, color="#88070D", marker="o")
ax.set_xlabel("Time",fontsize=18)
ax.set_ylabel("Personal Consumption ($ Billions)",color="#88070D",fontsize=18)
 


              
ax2=ax.twinx()
ax2.plot(top10perincomstatexample.Time, top10perincomstatexample.perincome/1000,color="#081A40", marker="o")
ax2.set_ylabel("Personal Income ($ Billions)",color="#081A40",fontsize=18)

ax.ticklabel_format(style='plain', scilimits=(0,0), axis='y')
ax2.ticklabel_format(style='plain', scilimits=(0,0), axis='y')        
               
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')



plt.show()

fig.savefig('twinconsumpincometop10referee.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')
#%%




style.use('seaborn-white')

import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
ax.plot(bot10perincomstatexample.Time, bot10perincomstatexample.consump/1000, color="#88070D", marker="o")
ax.set_xlabel("Time",fontsize=18)
ax.set_ylabel("Personal Consumption ($ Billions)",color="#88070D",fontsize=18)
 


              
ax2=ax.twinx()
ax2.plot(bot10perincomstatexample.Time, bot10perincomstatexample.perincome/1000,color="#081A40", marker="o")
ax2.set_ylabel("Personal Income ($ Billions)",color="#081A40",fontsize=18)

ax.ticklabel_format(style='plain', scilimits=(0,0), axis='y')
ax2.ticklabel_format(style='plain', scilimits=(0,0), axis='y')        
               
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')



plt.show()

fig.savefig('twinconsumpincomebot10referee.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')



#%% HOUSING DEBT REGRESSION ONLY --------------------------------------------------






#%% top per capita income abover the US national average
df1=topf.loc[topf['GeoName'].isin(['District of Columbia','Connecticut','New Jersey','Massachusetts','Maryland','New Hampshire','Virginia','New York','North Dakota','Alaska','Minnesota','Colorado','Washington','Rhode Island',	'Delaware',	'California','Illinois',	'Hawaii',	
'Wyoming','Pennsylvania','Vermont'])]

#%% bottom per capita income below the US national average
df2=topf.loc[topf['GeoName'].isin(['Iowa',	'Wisconsin','Maine','Kansas','Oregon','Nebraska','Texas','South Dakota',	'Ohio',	'Michigan','Florida',	
'Missouri',	'Montana','North Carolina','Nevada','Arizona','Georgia','Oklahoma',	
'Indiana','Tennessee','Utah','Louisiana','South Carolina','Idaho','Kentucky',	
'New Mexico','Alabama','Arkansas','West Virginia','Mississippi'])]	


#'Guam',	
#☻'Puerto Rico',	
#'Northern Mariana Islands',	
#'American Samoa'	
#'U.S. Virgin Islands',	

#%%

df1xample = df1
#df1xample = df1xample.rename(columns={"lncon": "Loans ($ Millions)","drcon":"Charge-off ($ Millions)"})
    
df2xample = df2
#df2xample = df2xample.rename(columns={"lncon": "Loans ($ Millions)","drcon":"Charge-off ($ Millions)"})
    


#%%



df1 = df1.set_index(['stalp2','years2'])



#%%
exog_vars = ['lnre'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1re = cons.fit()
print(two_res1re)
#%%  cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1reclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res1reclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2re = cons.fit()
print(two_res2re)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2reclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res2reclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3re = cons.fit()
print(two_res3re)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3reclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res3reclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4re = cons.fit()
print(two_res4re)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4reclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res4reclust)



#%% THE SAME   but it s not significant
exog_vars = ['lnre','crre'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1cre = cons.fit()
print(two_res1cre)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1creclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res1creclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2cre = cons.fit()
print(two_res2cre)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2creclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res2creclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3cre = cons.fit()
print(two_res3cre)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3creclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res3creclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4cre = cons.fit()
print(two_res4cre)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4creclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res4creclust)
#%%  
exog_vars = ['lnre','drre'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1cdre = cons.fit()
print(two_res1cdre)
#%% #%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1cdreclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res1cdreclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2cdre = cons.fit()
print(two_res2cdre)
#%% #%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2cdreclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res2cdreclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3cdre = cons.fit()
print(two_res3cdre)

#%% #%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3cdreclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res3cdreclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4cdre = cons.fit()
print(two_res4cdre)
#%% #%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4cdreclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res4cdreclust)

#%%

df2 = df2.set_index(['stalp2','years2'])
#%%

exog_vars = ['lnre','crre']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11re = cons.fit()
print(two_res11re)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11reclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res11reclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22re = cons.fit()
print(two_res22re)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22reclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res22reclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33re = cons.fit()
print(two_res33re)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33reclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res33reclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44re = cons.fit()
print(two_res44re)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44reclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res44reclust)
#%% THE SAME 
exog_vars = ['lnre']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11wocre = cons.fit()
print(two_res11wocre)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11wocreclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res11wocreclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22wocre = cons.fit()
print(two_res22wocre)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22wocreclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res22wocreclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33wocre = cons.fit()
print(two_res33wocre)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33wocreclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res33wocreclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44wocre = cons.fit()
print(two_res44wocre)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44wocreclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res44wocreclust)
#%%
from linearmodels.iv import IV2SLS
df1ivreg = df1.copy()
df1ivreg['const'] = 1
df110re = IV2SLS(dependent=df1ivreg['wealthvariationapprox'],
            exog=df1ivreg['const'],
            endog=df1ivreg['lnre'],
            instruments=df1ivreg['drre']).fit(cov_type='unadjusted')

print(df110re.summary)
#%%
df2ivreg = df2.copy()
df2ivreg['const'] = 1
df220re = IV2SLS(dependent=df2ivreg['wealthvariationapprox'],
            exog=df2ivreg['const'],
            endog=df2ivreg['lnre'],
            instruments=df2ivreg['drre']).fit(cov_type='unadjusted')

print(df220re.summary)
	
#%% iv with fe
df1ivregfe = df1.copy()
df1ivregfe['const'] = 1
df1ivregfe_fsivfere = sm.OLS(df1ivregfe['lnre'],
                    df1ivregfe[['const', 'drre']],
                    missing='drop').fit()
print(df1ivregfe_fsivfere.summary())

#%%
df1ivregfe['predicted_indivloan'] = df1ivregfe_fsivfere.predict()
#%%
exog_vars = ['predicted_indivloan','crre']
exog = sm.add_constant(df1ivregfe[exog_vars])
#%%
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
df1ivregfe_ss2ivfe1re = modiv.fit()
print(df1ivregfe_ss2ivfe1re)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
df1ivregfe_ss2ivfe2re = modiv.fit()
print(df1ivregfe_ss2ivfe2re)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
df1ivregfe_ss2ivfe3re = modiv.fit()
print(df1ivregfe_ss2ivfe3re)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
df1ivregfe_ss2ivfe4re = modiv.fit()
print(df1ivregfe_ss2ivfe4re)

#%%
exog_vars = ['predicted_indivloan']
exog = sm.add_constant(df1ivregfe[exog_vars])
#%%
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
df1ivregfe_ss2ivfe1wocre = modiv.fit()
print(df1ivregfe_ss2ivfe1wocre)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
df1ivregfe_ss2ivfe2wocre = modiv.fit()
print(df1ivregfe_ss2ivfe2wocre)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
df1ivregfe_ss2ivfe3wocre = modiv.fit()
print(df1ivregfe_ss2ivfe3wocre)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
df1ivregfe_ss2ivfe4wocre = modiv.fit()
print(df1ivregfe_ss2ivfe4wocre)





#%%
df2ivregfe = df2.copy()
df2ivregfe['const'] = 1
df2ivregfe_fsivfere = sm.OLS(df2ivregfe['lnre'],
                    df2ivregfe[['const', 'drre']],
                    missing='drop').fit()
print(df2ivregfe_fsivfere.summary())

#%%
df2ivregfe['predicted_indivloan'] = df2ivregfe_fsivfere.predict()
#%%
exog_vars = ['predicted_indivloan','crre']
exog = sm.add_constant(df2ivregfe[exog_vars])
#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
df2ivregfe_ss2ivfe1re = modiv.fit()
print(df2ivregfe_ss2ivfe1re)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
df2ivregfe_ss2ivfe2re = modiv.fit()
print(df2ivregfe_ss2ivfe2re)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
df2ivregfe_ss2ivfe3re = modiv.fit()
print(df2ivregfe_ss2ivfe3re)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
df2ivregfe_ss2ivfe4re = modiv.fit()
print(df2ivregfe_ss2ivfe4re)

#%% the same 
exog_vars = ['predicted_indivloan']
exog = sm.add_constant(df2ivregfe[exog_vars])
#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
df2ivregfe_ss2ivfe1wocre = modiv.fit()
print(df2ivregfe_ss2ivfe1wocre)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
df2ivregfe_ss2ivfe2wocre = modiv.fit()
print(df2ivregfe_ss2ivfe2wocre)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
df2ivregfe_ss2ivfe3wocre = modiv.fit()
print(df2ivregfe_ss2ivfe3wocre)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
df2ivregfe_ss2ivfe4wocre = modiv.fit()
print(df2ivregfe_ss2ivfe4wocre)






#%% log

#%%
exog_vars = ['loglnre'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%  cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1reclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res1reclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2reclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res2reclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3reclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res3reclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4reclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res4reclustlog)



#%% THE SAME   but it s not significant
exog_vars = ['loglnre','logcrre'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1creclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res1creclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2creclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res2creclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3creclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res3creclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4creclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res4creclustlog)
#%%  
exog_vars = ['loglnre','logdrre'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%% #%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res1cdreclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res1cdreclustlog)
#%% #%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res2cdreclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res2cdreclustlog)
#%% #%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res3cdreclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res3cdreclustlog)
#%% #%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res4cdreclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(two_res4cdreclustlog)

#%%

exog_vars = ['loglnre','logcrre']
exog = sm.add_constant(df2[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11reclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res11reclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22reclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res22reclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33reclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res33reclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44reclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res44reclustlog)
#%% THE SAME 
exog_vars = ['loglnre']
exog = sm.add_constant(df2[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11wocreclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res11wocreclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
two_res22wocreclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res22wocreclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
two_res33wocreclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res33wocreclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
two_res44wocreclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(two_res44wocreclustlog)


#%% ols df2 :  two_res22   two_res33  two_res44    iv2sls: df2ivregfe_ss2ivfe4

# iv2sls df1 :  df1ivregfe_ss2ivfe4



import plotly
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
import numpy as np

#%% REPORT ONLY SIGNIFICANT RESULTS
#%%

from collections import OrderedDict
from linearmodels.iv.results import compare

from linearmodels.iv import IV2SLS

import csv

from linearmodels.iv import IV2SLS


from statsmodels.iolib.summary3 import summary_col







from statsmodels.iolib.summary3 import summary_col






beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsrefree11re = summary_col([two_res22re,two_res44re,two_res22wocre,two_res44wocre],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("BOTTOMresultsrefree11rezAP08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11re.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsrefree11reclust = summary_col([two_res22reclust,two_res44reclust,two_res22wocreclust,two_res44wocreclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("BOTTOMresultsrefree11rezAP08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11reclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()




#%%
#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsrefree11reclustlog = summary_col([two_res22reclustlog,two_res44reclustlog,two_res22wocreclustlog,two_res44wocreclustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("BOTTOMresultsrefree11rezAP08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11reclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%  two_res1   over the us average (top)



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"




resultsrefree1re = summary_col([two_res2re,two_res4re,two_res2cdre,two_res4cdre],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("TOPresultsrefree1rezAP08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree1re.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"




resultsrefree1reclust = summary_col([two_res2reclust,two_res4reclust,two_res2cdreclust,two_res4cdreclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("TOPresultsrefree1rezAP08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree1reclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()






#%%



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"




resultsrefree1reclustlog = summary_col([two_res2reclustlog,two_res4reclustlog,two_res2cdreclustlog,two_res4cdreclustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("TOPresultsrefree1rezAP08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree1reclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()
#%%



#%%

# iv 

#resultsdf110 = df110.summary.as_text()
#dfoutput = open("resultsrefreedf110.csv",'w')
#dfoutput.write(resultsdf110)
#dfoutput.close()
#%%
#resultsdf220 = df220.summary.as_text()
#dfoutput = open("resultsrefreedf220.csv",'w')
#dfoutput.write(resultsdf220)
#dfoutput.close()






#%%



#%%



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"




resultsdf1ivregfe_ss2ivfere = summary_col([df1ivregfe_ss2ivfe3re,df1ivregfe_ss2ivfe4re,df1ivregfe_ss2ivfe1wocre,df1ivregfe_ss2ivfe2wocre,df1ivregfe_ss2ivfe3wocre,df1ivregfe_ss2ivfe4wocre],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("TOPresultsdf1ivregfe_ss2ivferez.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsdf1ivregfe_ss2ivfere.as_latex())
dfoutput.write(endtex)
dfoutput.close()





#%%




#%%


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"




resultsdf2ivregfe_ss2ivfere = summary_col([df2ivregfe_ss2ivfe4re],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("BOTTOMresultsdf2ivregfe_ss2ivferez.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsdf2ivregfe_ss2ivfere.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%% topf topf['lncon'].quantile(0.5)
#Out[53]: 3397469.5
# topf['lncon1'].quantile(0.5)
#Out[333]: 37611457.0






#%%

df1456example = topf


df1456example  = df1456example .set_index(['stalp2','years2'])

exog_vars = ['lncon1']
exog = sm.add_constant(df1456example [exog_vars])
#%% False True
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1456example.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
two_res11 = cons.fit()
print(two_res11)

d1datahighdfe = df1456example.copy()


#%%
import numpy as np

d1datahighdfe["udf1"] = two_res11.resids






#%%

np.corrcoef(d1datahighdfe.drcon1,d1datahighdfe.udf1)
#%%



#array([[1.        , 0.17896026],
#       [0.17896026, 1.        ]])

#array([[1.        , 0.06986361],
#       [0.06986361, 1.        ]])

#%%
import matplotlib.style as style
style.use('seaborn-white')        
        


sns.jointplot(data=d1datahighdfe, x='udf1', y='drcon1', kind='reg', color='#0F95D7')


ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_scientific(False)

#ax.set(xlim = (-400,200000000))  # 1400 Billion $

xlabels = ['{:,.0f}'.format(x)  for x in ax.get_xticks()/1000]
ax.set_xticklabels(xlabels)
#ax.set(ylim = (-400,8000000))   # 8 Million $
ylabels = ['{:,.0f}'.format(y)  for y in ax.get_yticks()/1000]
ax.set_yticklabels(ylabels)



#plt.grid(True)
plt.axis('tight')
plt.xlabel('Residuals',fontsize = 20)
plt.ylabel('Charge-off ($ Millions)',fontsize = 20)


plt.savefig('residualchargeoff.png', bbox_inches='tight')


#%% INDIVIDUAL LOAN ALONE 08 APRIL 














#%%
exog_vars = ['lncon'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1 = cons.fit()
print(indivtwo_res1)
#%%  cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1clust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res1clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2 = cons.fit()
print(indivtwo_res2)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2clust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res2clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3 = cons.fit()
print(indivtwo_res3)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3clust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res3clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4 = cons.fit()
print(indivtwo_res4)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4clust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res4clust)

#%% THE SAME   
exog_vars = ['lncon','crcon'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1c = cons.fit()
print(indivtwo_res1c)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1cclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res1cclust)

#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2c = cons.fit()
print(indivtwo_res2c)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2cclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res2cclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3c = cons.fit()
print(indivtwo_res3c)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3cclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res3cclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4c = cons.fit()
print(indivtwo_res4c)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4cclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res4cclust)
#%%
exog_vars = ['lncon','drcon'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1cd = cons.fit()
print(indivtwo_res1cd)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1cdclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res1cdclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2cd = cons.fit()
print(indivtwo_res2cd)
#%% cov_type="clustered", clusters= df1['Timestate'] 
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2cdclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res2cdclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3cd = cons.fit()
print(indivtwo_res3cd)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3cdclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res3cdclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4cd = cons.fit()
print(indivtwo_res4cd)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4cdclust = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res4cdclust)
#%%

#%%

exog_vars = ['lncon','crcon']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11 = cons.fit()
print(indivtwo_res11)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11clust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(indivtwo_res11clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22 = cons.fit()
print(indivtwo_res22)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22clust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(indivtwo_res22clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33 = cons.fit()
print(indivtwo_res33)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33clust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(indivtwo_res33clust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44 = cons.fit()
print(indivtwo_res44)
#%% cov_type="clustered", clusters= df1['Timestate']  
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44clust = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(indivtwo_res44clust)
#%%

exog_vars = ['lncon','drcon']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11drc = cons.fit()
print(indivtwo_res11drc)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11drcclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res11drcclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22drc = cons.fit()
print(indivtwo_res22drc)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22drcclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res22drcclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33drc = cons.fit()
print(indivtwo_res33drc)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33drcclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res33drcclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44drc = cons.fit()
print(indivtwo_res44drc)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44drcclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res44drcclust)


#%% THE SAME 
exog_vars = ['lncon']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11woc = cons.fit()
print(indivtwo_res11woc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11wocclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res11wocclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22woc = cons.fit()
print(indivtwo_res22woc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22wocclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res22wocclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33woc = cons.fit()
print(indivtwo_res33woc)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33wocclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res33wocclust)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44woc = cons.fit()
print(indivtwo_res44woc)

#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44wocclust = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res44wocclust)




#%%




#%% iv with fe
df1ivregfe = df1.copy()
df1ivregfe['const'] = 1
df1ivregfe_fsivfe = sm.OLS(df1ivregfe['lncon'],
                    df1ivregfe[['const', 'drcon']],
                    missing='drop').fit()
print(df1ivregfe_fsivfe.summary())

#%%
df1ivregfe['predicted_indivloan'] = df1ivregfe_fsivfe.predict()
#%%
exog_vars = ['predicted_indivloan','crcon']
exog = sm.add_constant(df1ivregfe[exog_vars])
#%%
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
indivdf1ivregfe_ss2ivfe1 = modiv.fit()
print(indivdf1ivregfe_ss2ivfe1)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
indivdf1ivregfe_ss2ivfe2 = modiv.fit()
print(indivdf1ivregfe_ss2ivfe2)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
indivdf1ivregfe_ss2ivfe3 = modiv.fit()
print(indivdf1ivregfe_ss2ivfe3)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
indivdf1ivregfe_ss2ivfe4 = modiv.fit()
print(indivdf1ivregfe_ss2ivfe4)

#%%
exog_vars = ['predicted_indivloan']
exog = sm.add_constant(df1ivregfe[exog_vars])
#%%
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
indivdf1ivregfe_ss2ivfe1woc = modiv.fit()
print(indivdf1ivregfe_ss2ivfe1woc)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
indivdf1ivregfe_ss2ivfe2woc = modiv.fit()
print(indivdf1ivregfe_ss2ivfe2woc)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
indivdf1ivregfe_ss2ivfe3woc = modiv.fit()
print(indivdf1ivregfe_ss2ivfe3woc)

#%%  
modiv =  PanelOLS(df1ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
indivdf1ivregfe_ss2ivfe4woc = modiv.fit()
print(indivdf1ivregfe_ss2ivfe4woc)





#%%
df2ivregfe = df2.copy()
df2ivregfe['const'] = 1
df2ivregfe_fsivfe = sm.OLS(df2ivregfe['lncon'],
                    df2ivregfe[['const', 'drcon']],
                    missing='drop').fit()
print(df2ivregfe_fsivfe.summary())

#%%
df2ivregfe['predicted_indivloan'] = df2ivregfe_fsivfe.predict()
#%%
exog_vars = ['predicted_indivloan','crcon']
exog = sm.add_constant(df2ivregfe[exog_vars])
#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
indivdf2ivregfe_ss2ivfe1 = modiv.fit()
print(indivdf2ivregfe_ss2ivfe1)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
indivdf2ivregfe_ss2ivfe2 = modiv.fit()
print(indivdf2ivregfe_ss2ivfe2)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
indivdf2ivregfe_ss2ivfe3 = modiv.fit()
print(indivdf2ivregfe_ss2ivfe3)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
indivdf2ivregfe_ss2ivfe4 = modiv.fit()
print(indivdf2ivregfe_ss2ivfe4)

#%% the same 
exog_vars = ['predicted_indivloan']
exog = sm.add_constant(df2ivregfe[exog_vars])
#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=True)
indivdf2ivregfe_ss2ivfe1woc = modiv.fit()
print(indivdf2ivregfe_ss2ivfe1woc)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=True,time_effects=False)
indivdf2ivregfe_ss2ivfe2woc = modiv.fit()
print(indivdf2ivregfe_ss2ivfe2woc)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=True)
indivdf2ivregfe_ss2ivfe3woc = modiv.fit()
print(indivdf2ivregfe_ss2ivfe3woc)

#%%  
modiv =  PanelOLS(df2ivregfe.wealthvariationapprox,
                    exog,entity_effects=False,time_effects=False)
indivdf2ivregfe_ss2ivfe4woc = modiv.fit()
print(indivdf2ivregfe_ss2ivfe4woc)



#%%

#%%
from linearmodels.iv import IV2SLS
df1ivreg = df1.copy()
df1ivreg['const'] = 1
INDdf110 = IV2SLS(dependent=df1ivreg['wealthvariationapprox'],
            exog=df1ivreg['const'],
            endog=df1ivreg['lncon'],
            instruments=df1ivreg['drcon']).fit(cov_type='unadjusted')

print(INDdf110.summary)
#%%
df2ivreg = df2.copy()
df2ivreg['const'] = 1
INDdf220 = IV2SLS(dependent=df2ivreg['wealthvariationapprox'],
            exog=df2ivreg['const'],
            endog=df2ivreg['lncon'],
            instruments=df2ivreg['drcon']).fit(cov_type='unadjusted')

print(INDdf220.summary)





#%% logggggg


#%%
exog_vars = ['loglncon'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%%  cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1clustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res1clustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2clustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res2clustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3clustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res3clustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4clustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res4clustlog)

#%% THE SAME   
exog_vars = ['loglncon','logcrcon'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1cclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res1cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2cclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res2cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3cclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res3cclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4cclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res4cclustlog)
#%%
exog_vars = ['loglncon','logdrcon'] # crcon
exog = sm.add_constant(df1[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res1cdclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res1cdclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res2cdclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res2cdclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res3cdclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res3cdclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res4cdclustlog = cons.fit(cov_type="clustered", clusters= df1['Timestate'] )
print(indivtwo_res4cdclustlog)
#%%

#%%

exog_vars = ['loglncon','logcrcon']
exog = sm.add_constant(df2[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11clustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(indivtwo_res11clustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22clustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(indivtwo_res22clustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33clustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(indivtwo_res33clustlog)
#%% cov_type="clustered", clusters= df1['Timestate']  
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44clustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'] )
print(indivtwo_res44clustlog)
#%%

exog_vars = ['loglncon','logdrcon']
exog = sm.add_constant(df2[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11drcclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res11drcclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22drcclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res22drcclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33drcclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res33drcclustlog)
#%% cov_type="clustered", clusters= df1['Timestate'] 
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44drcclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res44drcclustlog)


#%% THE SAME 
exog_vars = ['loglncon']
exog = sm.add_constant(df2[exog_vars])
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=True)
indivtwo_res11wocclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res11wocclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=True)
indivtwo_res22wocclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res22wocclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=True,time_effects=False)
indivtwo_res33wocclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res33wocclustlog)
#%% cov_type="clustered", clusters= df1['Timestate']
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.logwealthvariationapprox,exog,entity_effects=False,time_effects=False)
indivtwo_res44wocclustlog = cons.fit(cov_type="clustered", clusters= df2['Timestate'])
print(indivtwo_res44wocclustlog)



#%%



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIV = summary_col([indivtwo_res1,indivtwo_res2,indivtwo_res3,indivtwo_res4],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIV.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclust = summary_col([indivtwo_res1clust,indivtwo_res2clust,indivtwo_res3clust,indivtwo_res4clust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopap08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclustlog = summary_col([indivtwo_res1clustlog,indivtwo_res2clustlog,indivtwo_res3clustlog,indivtwo_res4clustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopap08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()
#%%


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIV = summary_col([indivtwo_res11woc,indivtwo_res22woc,indivtwo_res33woc,indivtwo_res44woc],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIV.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclust = summary_col([indivtwo_res11wocclust,indivtwo_res22wocclust,indivtwo_res33wocclust,indivtwo_res44wocclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotap08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()




#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclustlog = summary_col([indivtwo_res11wocclustlog,indivtwo_res22wocclustlog,indivtwo_res33wocclustlog,indivtwo_res44wocclustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotap08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%

#%%





beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIV = summary_col([indivtwo_res1cd,indivtwo_res2cd,indivtwo_res3cd,indivtwo_res4cd],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopDRCONap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIV.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%




beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclust = summary_col([indivtwo_res1cdclust,indivtwo_res2cdclust,indivtwo_res3cdclust,indivtwo_res4cdclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopDRCONap08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%

beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclustlog = summary_col([indivtwo_res1cdclustlog,indivtwo_res2cdclustlog,indivtwo_res3cdclustlog,indivtwo_res4cdclustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopDRCONap08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIV = summary_col([indivtwo_res22drc,indivtwo_res44drc],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotDRCONap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIV.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclust = summary_col([indivtwo_res22drcclust,indivtwo_res44drcclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotDRCONap08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclustlog = summary_col([indivtwo_res22drcclustlog,indivtwo_res44drcclustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotDRCONap08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%





beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIV = summary_col([indivtwo_res1c,indivtwo_res2c,indivtwo_res3c,indivtwo_res4c],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopCRCONap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIV.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%




beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclust = summary_col([indivtwo_res1cclust,indivtwo_res2cclust,indivtwo_res3cclust,indivtwo_res4cclust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopCRCONap08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%



beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclustlog = summary_col([indivtwo_res1cclustlog,indivtwo_res2cclustlog,indivtwo_res3cclustlog,indivtwo_res4cclustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVtopCRCONap08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIV = summary_col([indivtwo_res11,indivtwo_res22,indivtwo_res44],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotCRCONap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIV.as_latex())
dfoutput.write(endtex)
dfoutput.close()

#%%


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclust = summary_col([indivtwo_res11clust,indivtwo_res22clust,indivtwo_res44clust],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotCRCONap08clust.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclust.as_latex())
dfoutput.write(endtex)
dfoutput.close()





#%%


beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIVclustlog = summary_col([indivtwo_res11clustlog,indivtwo_res22clustlog,indivtwo_res44clustlog],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("resultsINDIVbotCRCONap08clustlog.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIVclustlog.as_latex())
dfoutput.write(endtex)
dfoutput.close()




#%% top iv
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIV = summary_col([indivdf1ivregfe_ss2ivfe1,indivdf1ivregfe_ss2ivfe3,indivdf1ivregfe_ss2ivfe4],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("iv2slsresultsINDIVtopCRCONap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIV.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%                        

                            
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"


resultsINDIV = summary_col([indivdf1ivregfe_ss2ivfe1woc,indivdf1ivregfe_ss2ivfe2woc,indivdf1ivregfe_ss2ivfe3woc,indivdf1ivregfe_ss2ivfe4woc],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("iv2slsresultsINDIVtopap08.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsINDIV.as_latex())
dfoutput.write(endtex)
dfoutput.close()


#%%
df1['logwealthvariationapprox']=np.log(df1['wealthvariationapprox'])
#%%
df2['logwealthvariationapprox']=np.log(df2['wealthvariationapprox'])
#%%
df1['loglncon1']=np.log(df1['lncon1'])
#%%
df2['loglncon1']=np.log(df2['lncon1'])
#%%

import matplotlib.style as style
style.use('seaborn-white')        
        
sns.jointplot(data=df1, x='logwealthvariationapprox', y='loglncon1', kind='reg', color='#0F95D7')

ax = plt.gca()

plt.axis('tight')
plt.xlabel('Debt',fontsize = 20)
plt.ylabel('Wealth ',fontsize = 20)

plt.savefig('motivajlogfdic.png', bbox_inches='tight')
#%%



import matplotlib.style as style
style.use('seaborn-white')        
        
sns.jointplot(data=df2, x='logwealthvariationapprox', y='loglncon1', kind='reg', color='#0F95D7')

ax = plt.gca()

plt.axis('tight')
plt.xlabel('Debt',fontsize = 20)
plt.ylabel('Wealth ',fontsize = 20)

plt.savefig('motivajlogfdic2.png', bbox_inches='tight')
                                               

    
#%%