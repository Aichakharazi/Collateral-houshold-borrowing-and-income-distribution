# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:06:50 2020

@author: Aicha-PC
"""







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



#%%


downincome97 = pd.read_csv('download1997.csv', sep=';',error_bad_lines=False)

#%%
downbank97 = pd.read_csv('SDI_Download_Data1997.csv', sep=',',error_bad_lines=False)



#%%


downincome97["GeoFips"] = pandas.to_numeric(downincome97["GeoFips"], errors="coerce")
downincome97["1997"] = pandas.to_numeric(downincome97["1997"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank97["cert"] = pandas.to_numeric(downbank97["cert"], errors="coerce")
downbank97["docket"] = pandas.to_numeric(downbank97["docket"], errors="coerce")
downbank97["rssdhcr"] = pandas.to_numeric(downbank97["rssdhcr"], errors="coerce")
downbank97["fed_rssd"] = pandas.to_numeric(downbank97["fed_rssd"], errors="coerce")

downbank97["hctmult"] = pandas.to_numeric(downbank97["hctmult"], errors="coerce")
downbank97["zip"] = pandas.to_numeric(downbank97["zip"], errors="coerce")


downbank97["lncon"] = pandas.to_numeric(downbank97["lncon"], errors="coerce")
downbank97["drcon"] = pandas.to_numeric(downbank97["drcon"], errors="coerce")
downbank97["crcon"] = pandas.to_numeric(downbank97["crcon"], errors="coerce")
downbank97["ntcon"] = pandas.to_numeric(downbank97["ntcon"], errors="coerce")


downbank97["lnre"] = pandas.to_numeric(downbank97["lnre"], errors="coerce")
downbank97["drre"] = pandas.to_numeric(downbank97["drre"], errors="coerce")
downbank97["crre"] = pandas.to_numeric(downbank97["crre"], errors="coerce")
downbank97["ntre"] = pandas.to_numeric(downbank97["ntre"], errors="coerce")




#%%

downbank97 =  downbank97.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()

#%%

pritthree97= pd.merge(downincome97, downbank97, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree97 = pritthree97.sort_values('county', ascending=False)
pritthree97 = pritthree97.drop_duplicates(subset='county', keep='first')



#%%


downincome98 = pd.read_csv('download1998.csv', sep=';',error_bad_lines=False)

#%%
downbank98 = pd.read_csv('SDI_Download_Data1998.csv', sep=',',error_bad_lines=False)



#%%


downincome98["GeoFips"] = pandas.to_numeric(downincome98["GeoFips"], errors="coerce")
downincome98["1998"] = pandas.to_numeric(downincome98["1998"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank98["cert"] = pandas.to_numeric(downbank98["cert"], errors="coerce")
downbank98["docket"] = pandas.to_numeric(downbank98["docket"], errors="coerce")
downbank98["rssdhcr"] = pandas.to_numeric(downbank98["rssdhcr"], errors="coerce")
downbank98["fed_rssd"] = pandas.to_numeric(downbank98["fed_rssd"], errors="coerce")

downbank98["hctmult"] = pandas.to_numeric(downbank98["hctmult"], errors="coerce")
downbank98["zip"] = pandas.to_numeric(downbank98["zip"], errors="coerce")


downbank98["lncon"] = pandas.to_numeric(downbank98["lncon"], errors="coerce")
downbank98["drcon"] = pandas.to_numeric(downbank98["drcon"], errors="coerce")
downbank98["crcon"] = pandas.to_numeric(downbank98["crcon"], errors="coerce")
downbank98["ntcon"] = pandas.to_numeric(downbank98["ntcon"], errors="coerce")

#%%

downbank98["lnre"] = pandas.to_numeric(downbank98["lnre"], errors="coerce")
downbank98["drre"] = pandas.to_numeric(downbank98["drre"], errors="coerce")
downbank98["crre"] = pandas.to_numeric(downbank98["crre"], errors="coerce")
downbank98["ntre"] = pandas.to_numeric(downbank98["ntre"], errors="coerce")




#%%

downbank98 =  downbank98.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()



#%%

pritthree98= pd.merge(downincome98, downbank98, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree98 = pritthree98.sort_values('county', ascending=False)
pritthree98 = pritthree98.drop_duplicates(subset='county', keep='first')



#%%


downincome99 = pd.read_csv('download1999.csv', sep=';',error_bad_lines=False)

#%%
downbank99 = pd.read_csv('SDI_Download_Data1999.csv', sep=',',error_bad_lines=False)



#%%


downincome99["GeoFips"] = pandas.to_numeric(downincome99["GeoFips"], errors="coerce")
downincome99["1999"] = pandas.to_numeric(downincome99["1999"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank99["cert"] = pandas.to_numeric(downbank99["cert"], errors="coerce")
downbank99["docket"] = pandas.to_numeric(downbank99["docket"], errors="coerce")
downbank99["rssdhcr"] = pandas.to_numeric(downbank99["rssdhcr"], errors="coerce")
downbank99["fed_rssd"] = pandas.to_numeric(downbank99["fed_rssd"], errors="coerce")

downbank99["hctmult"] = pandas.to_numeric(downbank99["hctmult"], errors="coerce")
downbank99["zip"] = pandas.to_numeric(downbank99["zip"], errors="coerce")


downbank99["lncon"] = pandas.to_numeric(downbank99["lncon"], errors="coerce")
downbank99["drcon"] = pandas.to_numeric(downbank99["drcon"], errors="coerce")
downbank99["crcon"] = pandas.to_numeric(downbank99["crcon"], errors="coerce")
downbank99["ntcon"] = pandas.to_numeric(downbank99["ntcon"], errors="coerce")



downbank99["lnre"] = pandas.to_numeric(downbank99["lnre"], errors="coerce")
downbank99["drre"] = pandas.to_numeric(downbank99["drre"], errors="coerce")
downbank99["crre"] = pandas.to_numeric(downbank99["crre"], errors="coerce")
downbank99["ntre"] = pandas.to_numeric(downbank99["ntre"], errors="coerce")




#%%

downbank99 =  downbank99.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()

#%%


#%%

pritthree99= pd.merge(downincome99, downbank99, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree99 = pritthree99.sort_values('county', ascending=False)
pritthree99 = pritthree99.drop_duplicates(subset='county', keep='first')



#%%


downincome00 = pd.read_csv('download2000.csv', sep=';',error_bad_lines=False)

#%%
downbank00 = pd.read_csv('SDI_Download_Data2000.csv', sep=',',error_bad_lines=False)



#%%


downincome00["GeoFips"] = pandas.to_numeric(downincome00["GeoFips"], errors="coerce")
downincome00["2000"] = pandas.to_numeric(downincome00["2000"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank00["cert"] = pandas.to_numeric(downbank00["cert"], errors="coerce")
downbank00["docket"] = pandas.to_numeric(downbank00["docket"], errors="coerce")
downbank00["rssdhcr"] = pandas.to_numeric(downbank00["rssdhcr"], errors="coerce")
downbank00["fed_rssd"] = pandas.to_numeric(downbank00["fed_rssd"], errors="coerce")

downbank00["hctmult"] = pandas.to_numeric(downbank00["hctmult"], errors="coerce")
downbank00["zip"] = pandas.to_numeric(downbank00["zip"], errors="coerce")


downbank00["lncon"] = pandas.to_numeric(downbank00["lncon"], errors="coerce")
downbank00["drcon"] = pandas.to_numeric(downbank00["drcon"], errors="coerce")
downbank00["crcon"] = pandas.to_numeric(downbank00["crcon"], errors="coerce")
downbank00["ntcon"] = pandas.to_numeric(downbank00["ntcon"], errors="coerce")


downbank00["lnre"] = pandas.to_numeric(downbank00["lnre"], errors="coerce")
downbank00["drre"] = pandas.to_numeric(downbank00["drre"], errors="coerce")
downbank00["crre"] = pandas.to_numeric(downbank00["crre"], errors="coerce")
downbank00["ntre"] = pandas.to_numeric(downbank00["ntre"], errors="coerce")




#%%

downbank00 =  downbank00.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree00= pd.merge(downincome00, downbank00, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree00 = pritthree00.sort_values('county', ascending=False)
pritthree00 = pritthree00.drop_duplicates(subset='county', keep='first')



#%%


downincome01 = pd.read_csv('download2001.csv', sep=';',error_bad_lines=False)

#%%
downbank01 = pd.read_csv('SDI_Download_Data2001.csv', sep=',',error_bad_lines=False)



#%%


downincome01["GeoFips"] = pandas.to_numeric(downincome01["GeoFips"], errors="coerce")
downincome01["2001"] = pandas.to_numeric(downincome01["2001"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank01["cert"] = pandas.to_numeric(downbank01["cert"], errors="coerce")
downbank01["docket"] = pandas.to_numeric(downbank01["docket"], errors="coerce")
downbank01["rssdhcr"] = pandas.to_numeric(downbank01["rssdhcr"], errors="coerce")
downbank01["fed_rssd"] = pandas.to_numeric(downbank01["fed_rssd"], errors="coerce")

downbank01["hctmult"] = pandas.to_numeric(downbank01["hctmult"], errors="coerce")
downbank01["zip"] = pandas.to_numeric(downbank01["zip"], errors="coerce")


downbank01["lncon"] = pandas.to_numeric(downbank01["lncon"], errors="coerce")
downbank01["drcon"] = pandas.to_numeric(downbank01["drcon"], errors="coerce")
downbank01["crcon"] = pandas.to_numeric(downbank01["crcon"], errors="coerce")
downbank01["ntcon"] = pandas.to_numeric(downbank01["ntcon"], errors="coerce")


downbank01["lnre"] = pandas.to_numeric(downbank01["lnre"], errors="coerce")
downbank01["drre"] = pandas.to_numeric(downbank01["drre"], errors="coerce")
downbank01["crre"] = pandas.to_numeric(downbank01["crre"], errors="coerce")
downbank01["ntre"] = pandas.to_numeric(downbank01["ntre"], errors="coerce")




#%%

downbank01 =  downbank01.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()




#%%

pritthree01= pd.merge(downincome01, downbank01, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree01 = pritthree01.sort_values('county', ascending=False)
pritthree01 = pritthree01.drop_duplicates(subset='county', keep='first')


#%%


downincome02 = pd.read_csv('download2002.csv', sep=';',error_bad_lines=False)

#%%
downbank02 = pd.read_csv('SDI_Download_Data2002.csv', sep=',',error_bad_lines=False)



#%%


downincome02["GeoFips"] = pandas.to_numeric(downincome02["GeoFips"], errors="coerce")
downincome02["2002"] = pandas.to_numeric(downincome02["2002"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank02["cert"] = pandas.to_numeric(downbank02["cert"], errors="coerce")
downbank02["docket"] = pandas.to_numeric(downbank02["docket"], errors="coerce")
downbank02["rssdhcr"] = pandas.to_numeric(downbank02["rssdhcr"], errors="coerce")
downbank02["fed_rssd"] = pandas.to_numeric(downbank02["fed_rssd"], errors="coerce")

downbank02["hctmult"] = pandas.to_numeric(downbank02["hctmult"], errors="coerce")
downbank02["zip"] = pandas.to_numeric(downbank02["zip"], errors="coerce")


downbank02["lncon"] = pandas.to_numeric(downbank02["lncon"], errors="coerce")
downbank02["drcon"] = pandas.to_numeric(downbank02["drcon"], errors="coerce")
downbank02["crcon"] = pandas.to_numeric(downbank02["crcon"], errors="coerce")
downbank02["ntcon"] = pandas.to_numeric(downbank02["ntcon"], errors="coerce")



downbank02["lnre"] = pandas.to_numeric(downbank02["lnre"], errors="coerce")
downbank02["drre"] = pandas.to_numeric(downbank02["drre"], errors="coerce")
downbank02["crre"] = pandas.to_numeric(downbank02["crre"], errors="coerce")
downbank02["ntre"] = pandas.to_numeric(downbank02["ntre"], errors="coerce")




#%%

downbank02 =  downbank02.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()




#%%

pritthree02= pd.merge(downincome02, downbank02, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree02 = pritthree02.sort_values('county', ascending=False)
pritthree02 = pritthree02.drop_duplicates(subset='county', keep='first')


#%%


downincome03 = pd.read_csv('download2003.csv', sep=';',error_bad_lines=False)

#%%
downbank03 = pd.read_csv('SDI_Download_Data2003.csv', sep=',',error_bad_lines=False)



#%%


downincome03["GeoFips"] = pandas.to_numeric(downincome03["GeoFips"], errors="coerce")
downincome03["2003"] = pandas.to_numeric(downincome03["2003"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank03["cert"] = pandas.to_numeric(downbank03["cert"], errors="coerce")
downbank03["docket"] = pandas.to_numeric(downbank03["docket"], errors="coerce")
downbank03["rssdhcr"] = pandas.to_numeric(downbank03["rssdhcr"], errors="coerce")
downbank03["fed_rssd"] = pandas.to_numeric(downbank03["fed_rssd"], errors="coerce")

downbank03["hctmult"] = pandas.to_numeric(downbank03["hctmult"], errors="coerce")
downbank03["zip"] = pandas.to_numeric(downbank03["zip"], errors="coerce")


downbank03["lncon"] = pandas.to_numeric(downbank03["lncon"], errors="coerce")
downbank03["drcon"] = pandas.to_numeric(downbank03["drcon"], errors="coerce")
downbank03["crcon"] = pandas.to_numeric(downbank03["crcon"], errors="coerce")
downbank03["ntcon"] = pandas.to_numeric(downbank03["ntcon"], errors="coerce")



downbank03["lnre"] = pandas.to_numeric(downbank03["lnre"], errors="coerce")
downbank03["drre"] = pandas.to_numeric(downbank03["drre"], errors="coerce")
downbank03["crre"] = pandas.to_numeric(downbank03["crre"], errors="coerce")
downbank03["ntre"] = pandas.to_numeric(downbank03["ntre"], errors="coerce")




#%%

downbank03 =  downbank03.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree03= pd.merge(downincome03, downbank03, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree03 = pritthree03.sort_values('county', ascending=False)
pritthree03 = pritthree03.drop_duplicates(subset='county', keep='first')


#%%


downincome04 = pd.read_csv('download2004.csv', sep=';',error_bad_lines=False)

#%%
downbank04 = pd.read_csv('SDI_Download_Data2004.csv', sep=',',error_bad_lines=False)



#%%


downincome04["GeoFips"] = pandas.to_numeric(downincome04["GeoFips"], errors="coerce")
downincome04["2004"] = pandas.to_numeric(downincome04["2004"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank04["cert"] = pandas.to_numeric(downbank04["cert"], errors="coerce")
downbank04["docket"] = pandas.to_numeric(downbank04["docket"], errors="coerce")
downbank04["rssdhcr"] = pandas.to_numeric(downbank04["rssdhcr"], errors="coerce")
downbank04["fed_rssd"] = pandas.to_numeric(downbank04["fed_rssd"], errors="coerce")

downbank04["hctmult"] = pandas.to_numeric(downbank04["hctmult"], errors="coerce")
downbank04["zip"] = pandas.to_numeric(downbank04["zip"], errors="coerce")


downbank04["lncon"] = pandas.to_numeric(downbank04["lncon"], errors="coerce")
downbank04["drcon"] = pandas.to_numeric(downbank04["drcon"], errors="coerce")
downbank04["crcon"] = pandas.to_numeric(downbank04["crcon"], errors="coerce")
downbank04["ntcon"] = pandas.to_numeric(downbank04["ntcon"], errors="coerce")



downbank04["lnre"] = pandas.to_numeric(downbank04["lnre"], errors="coerce")
downbank04["drre"] = pandas.to_numeric(downbank04["drre"], errors="coerce")
downbank04["crre"] = pandas.to_numeric(downbank04["crre"], errors="coerce")
downbank04["ntre"] = pandas.to_numeric(downbank04["ntre"], errors="coerce")




#%%

downbank04 =  downbank04.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree04= pd.merge(downincome04, downbank04, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree04 = pritthree04.sort_values('county', ascending=False)
pritthree04 = pritthree04.drop_duplicates(subset='county', keep='first')


#%%


downincome05 = pd.read_csv('download2005.csv', sep=';',error_bad_lines=False)

#%%
downbank05 = pd.read_csv('SDI_Download_Data2005.csv', sep=',',error_bad_lines=False)



#%%


downincome05["GeoFips"] = pandas.to_numeric(downincome05["GeoFips"], errors="coerce")
downincome05["2005"] = pandas.to_numeric(downincome05["2005"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank05["cert"] = pandas.to_numeric(downbank05["cert"], errors="coerce")
downbank05["docket"] = pandas.to_numeric(downbank05["docket"], errors="coerce")
downbank05["rssdhcr"] = pandas.to_numeric(downbank05["rssdhcr"], errors="coerce")
downbank05["fed_rssd"] = pandas.to_numeric(downbank05["fed_rssd"], errors="coerce")

downbank05["hctmult"] = pandas.to_numeric(downbank05["hctmult"], errors="coerce")
downbank05["zip"] = pandas.to_numeric(downbank05["zip"], errors="coerce")


downbank05["lncon"] = pandas.to_numeric(downbank05["lncon"], errors="coerce")
downbank05["drcon"] = pandas.to_numeric(downbank05["drcon"], errors="coerce")
downbank05["crcon"] = pandas.to_numeric(downbank05["crcon"], errors="coerce")
downbank05["ntcon"] = pandas.to_numeric(downbank05["ntcon"], errors="coerce")



downbank05["lnre"] = pandas.to_numeric(downbank05["lnre"], errors="coerce")
downbank05["drre"] = pandas.to_numeric(downbank05["drre"], errors="coerce")
downbank05["crre"] = pandas.to_numeric(downbank05["crre"], errors="coerce")
downbank05["ntre"] = pandas.to_numeric(downbank05["ntre"], errors="coerce")




#%%

downbank05 =  downbank05.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()



#%%

pritthree05= pd.merge(downincome05, downbank05, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree05 = pritthree05.sort_values('county', ascending=False)
pritthree05 = pritthree05.drop_duplicates(subset='county', keep='first')

#%%

downincome06 = pd.read_csv('download2006.csv', sep=';',error_bad_lines=False)

#%%
downbank06 = pd.read_csv('SDI_Download_Data2006.csv', sep=',',error_bad_lines=False)



#%%


downincome06["GeoFips"] = pandas.to_numeric(downincome06["GeoFips"], errors="coerce")
downincome06["2006"] = pandas.to_numeric(downincome06["2006"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank06["cert"] = pandas.to_numeric(downbank06["cert"], errors="coerce")
downbank06["docket"] = pandas.to_numeric(downbank06["docket"], errors="coerce")
downbank06["rssdhcr"] = pandas.to_numeric(downbank06["rssdhcr"], errors="coerce")
downbank06["fed_rssd"] = pandas.to_numeric(downbank06["fed_rssd"], errors="coerce")

downbank06["hctmult"] = pandas.to_numeric(downbank06["hctmult"], errors="coerce")
downbank06["zip"] = pandas.to_numeric(downbank06["zip"], errors="coerce")


downbank06["lncon"] = pandas.to_numeric(downbank06["lncon"], errors="coerce")
downbank06["drcon"] = pandas.to_numeric(downbank06["drcon"], errors="coerce")
downbank06["crcon"] = pandas.to_numeric(downbank06["crcon"], errors="coerce")
downbank06["ntcon"] = pandas.to_numeric(downbank06["ntcon"], errors="coerce")


downbank06["lnre"] = pandas.to_numeric(downbank06["lnre"], errors="coerce")
downbank06["drre"] = pandas.to_numeric(downbank06["drre"], errors="coerce")
downbank06["crre"] = pandas.to_numeric(downbank06["crre"], errors="coerce")
downbank06["ntre"] = pandas.to_numeric(downbank06["ntre"], errors="coerce")




#%%

downbank06 =  downbank06.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()

#%%

pritthree06= pd.merge(downincome06, downbank06, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree06 = pritthree06.sort_values('county', ascending=False)
pritthree06 = pritthree06.drop_duplicates(subset='county', keep='first')


#%%
downincome07 = pd.read_csv('download2007.csv', sep=';',error_bad_lines=False)

#%%
downbank07 = pd.read_csv('SDI_Download_Data2007.csv', sep=',',error_bad_lines=False)



#%%


downincome07["GeoFips"] = pandas.to_numeric(downincome07["GeoFips"], errors="coerce")
downincome07["2007"] = pandas.to_numeric(downincome07["2007"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank07["cert"] = pandas.to_numeric(downbank07["cert"], errors="coerce")
downbank07["docket"] = pandas.to_numeric(downbank07["docket"], errors="coerce")
downbank07["rssdhcr"] = pandas.to_numeric(downbank07["rssdhcr"], errors="coerce")
downbank07["fed_rssd"] = pandas.to_numeric(downbank07["fed_rssd"], errors="coerce")

downbank07["hctmult"] = pandas.to_numeric(downbank07["hctmult"], errors="coerce")
downbank07["zip"] = pandas.to_numeric(downbank07["zip"], errors="coerce")


downbank07["lncon"] = pandas.to_numeric(downbank07["lncon"], errors="coerce")
downbank07["drcon"] = pandas.to_numeric(downbank07["drcon"], errors="coerce")
downbank07["crcon"] = pandas.to_numeric(downbank07["crcon"], errors="coerce")
downbank07["ntcon"] = pandas.to_numeric(downbank07["ntcon"], errors="coerce")


downbank07["lnre"] = pandas.to_numeric(downbank07["lnre"], errors="coerce")
downbank07["drre"] = pandas.to_numeric(downbank07["drre"], errors="coerce")
downbank07["crre"] = pandas.to_numeric(downbank07["crre"], errors="coerce")
downbank07["ntre"] = pandas.to_numeric(downbank07["ntre"], errors="coerce")




#%%

downbank07 =  downbank07.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()



#%%

pritthree07= pd.merge(downincome07, downbank07, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree07 = pritthree07.sort_values('county', ascending=False)
pritthree07 = pritthree07.drop_duplicates(subset='county', keep='first')



#%%

downincome08 = pd.read_csv('download2008.csv', sep=';',error_bad_lines=False)

#%%
downbank08 = pd.read_csv('SDI_Download_Data2008.csv', sep=',',error_bad_lines=False)



#%%


downincome08["GeoFips"] = pandas.to_numeric(downincome08["GeoFips"], errors="coerce")
downincome08["2008"] = pandas.to_numeric(downincome08["2008"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank08["cert"] = pandas.to_numeric(downbank08["cert"], errors="coerce")
downbank08["docket"] = pandas.to_numeric(downbank08["docket"], errors="coerce")
downbank08["rssdhcr"] = pandas.to_numeric(downbank08["rssdhcr"], errors="coerce")
downbank08["fed_rssd"] = pandas.to_numeric(downbank08["fed_rssd"], errors="coerce")

downbank08["hctmult"] = pandas.to_numeric(downbank08["hctmult"], errors="coerce")
downbank08["zip"] = pandas.to_numeric(downbank08["zip"], errors="coerce")


downbank08["lncon"] = pandas.to_numeric(downbank08["lncon"], errors="coerce")
downbank08["drcon"] = pandas.to_numeric(downbank08["drcon"], errors="coerce")
downbank08["crcon"] = pandas.to_numeric(downbank08["crcon"], errors="coerce")
downbank08["ntcon"] = pandas.to_numeric(downbank08["ntcon"], errors="coerce")


downbank08["lnre"] = pandas.to_numeric(downbank08["lnre"], errors="coerce")
downbank08["drre"] = pandas.to_numeric(downbank08["drre"], errors="coerce")
downbank08["crre"] = pandas.to_numeric(downbank08["crre"], errors="coerce")
downbank08["ntre"] = pandas.to_numeric(downbank08["ntre"], errors="coerce")




#%%

downbank08 =  downbank08.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree08= pd.merge(downincome08, downbank08, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree08 = pritthree08.sort_values('county', ascending=False)
pritthree08 = pritthree08.drop_duplicates(subset='county', keep='first')



#%%

downincome09 = pd.read_csv('download2009.csv', sep=';',error_bad_lines=False)

#%%
downbank09 = pd.read_csv('SDI_Download_Data2009.csv', sep=',',error_bad_lines=False)



#%%


downincome09["GeoFips"] = pandas.to_numeric(downincome09["GeoFips"], errors="coerce")
downincome09["2009"] = pandas.to_numeric(downincome09["2009"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank09["cert"] = pandas.to_numeric(downbank09["cert"], errors="coerce")
downbank09["docket"] = pandas.to_numeric(downbank09["docket"], errors="coerce")
downbank09["rssdhcr"] = pandas.to_numeric(downbank09["rssdhcr"], errors="coerce")
downbank09["fed_rssd"] = pandas.to_numeric(downbank09["fed_rssd"], errors="coerce")

downbank09["hctmult"] = pandas.to_numeric(downbank09["hctmult"], errors="coerce")
downbank09["zip"] = pandas.to_numeric(downbank09["zip"], errors="coerce")


downbank09["lncon"] = pandas.to_numeric(downbank09["lncon"], errors="coerce")
downbank09["drcon"] = pandas.to_numeric(downbank09["drcon"], errors="coerce")
downbank09["crcon"] = pandas.to_numeric(downbank09["crcon"], errors="coerce")
downbank09["ntcon"] = pandas.to_numeric(downbank09["ntcon"], errors="coerce")


downbank09["lnre"] = pandas.to_numeric(downbank09["lnre"], errors="coerce")
downbank09["drre"] = pandas.to_numeric(downbank09["drre"], errors="coerce")
downbank09["crre"] = pandas.to_numeric(downbank09["crre"], errors="coerce")
downbank09["ntre"] = pandas.to_numeric(downbank09["ntre"], errors="coerce")




#%%

downbank09 =  downbank09.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree09= pd.merge(downincome09, downbank09, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree09 = pritthree09.sort_values('county', ascending=False)
pritthree09 = pritthree09.drop_duplicates(subset='county', keep='first')


#%%
downincome10 = pd.read_csv('download2010.csv', sep=';',error_bad_lines=False)

#%%
downbank10 = pd.read_csv('SDI_Download_Data2010.csv', sep=',',error_bad_lines=False)



#%%


downincome10["GeoFips"] = pandas.to_numeric(downincome10["GeoFips"], errors="coerce")
downincome10["2010"] = pandas.to_numeric(downincome10["2010"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank10["cert"] = pandas.to_numeric(downbank10["cert"], errors="coerce")
downbank10["docket"] = pandas.to_numeric(downbank10["docket"], errors="coerce")
downbank10["rssdhcr"] = pandas.to_numeric(downbank10["rssdhcr"], errors="coerce")
downbank10["fed_rssd"] = pandas.to_numeric(downbank10["fed_rssd"], errors="coerce")

downbank10["hctmult"] = pandas.to_numeric(downbank10["hctmult"], errors="coerce")
downbank10["zip"] = pandas.to_numeric(downbank10["zip"], errors="coerce")


downbank10["lncon"] = pandas.to_numeric(downbank10["lncon"], errors="coerce")
downbank10["drcon"] = pandas.to_numeric(downbank10["drcon"], errors="coerce")
downbank10["crcon"] = pandas.to_numeric(downbank10["crcon"], errors="coerce")
downbank10["ntcon"] = pandas.to_numeric(downbank10["ntcon"], errors="coerce")


downbank10["lnre"] = pandas.to_numeric(downbank10["lnre"], errors="coerce")
downbank10["drre"] = pandas.to_numeric(downbank10["drre"], errors="coerce")
downbank10["crre"] = pandas.to_numeric(downbank10["crre"], errors="coerce")
downbank10["ntre"] = pandas.to_numeric(downbank10["ntre"], errors="coerce")




#%%

downbank10 =  downbank10.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()

#%%

pritthree10= pd.merge(downincome10, downbank10, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree10 = pritthree10.sort_values('county', ascending=False)
pritthree10 = pritthree10.drop_duplicates(subset='county', keep='first')




#%%


downincome11 = pd.read_csv('download2011.csv', sep=';',error_bad_lines=False)

#%%
downbank11 = pd.read_csv('SDI_Download_Data2011.csv', sep=',',error_bad_lines=False)



#%%


downincome11["GeoFips"] = pandas.to_numeric(downincome11["GeoFips"], errors="coerce")
downincome11["2011"] = pandas.to_numeric(downincome11["2011"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank11["cert"] = pandas.to_numeric(downbank11["cert"], errors="coerce")
downbank11["docket"] = pandas.to_numeric(downbank11["docket"], errors="coerce")
downbank11["rssdhcr"] = pandas.to_numeric(downbank11["rssdhcr"], errors="coerce")
downbank11["fed_rssd"] = pandas.to_numeric(downbank11["fed_rssd"], errors="coerce")

downbank11["hctmult"] = pandas.to_numeric(downbank11["hctmult"], errors="coerce")
downbank11["zip"] = pandas.to_numeric(downbank11["zip"], errors="coerce")


downbank11["lncon"] = pandas.to_numeric(downbank11["lncon"], errors="coerce")
downbank11["drcon"] = pandas.to_numeric(downbank11["drcon"], errors="coerce")
downbank11["crcon"] = pandas.to_numeric(downbank11["crcon"], errors="coerce")
downbank11["ntcon"] = pandas.to_numeric(downbank11["ntcon"], errors="coerce")



downbank11["lnre"] = pandas.to_numeric(downbank11["lnre"], errors="coerce")
downbank11["drre"] = pandas.to_numeric(downbank11["drre"], errors="coerce")
downbank11["crre"] = pandas.to_numeric(downbank11["crre"], errors="coerce")
downbank11["ntre"] = pandas.to_numeric(downbank11["ntre"], errors="coerce")




#%%

downbank11 =  downbank11.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree11= pd.merge(downincome11, downbank11, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree11 = pritthree11.sort_values('county', ascending=False)
pritthree11 = pritthree11.drop_duplicates(subset='county', keep='first')



#%%


downincome12 = pd.read_csv('download2012.csv', sep=';',error_bad_lines=False)

#%%
downbank12 = pd.read_csv('SDI_Download_Data2012.csv', sep=',',error_bad_lines=False)



#%%


downincome12["GeoFips"] = pandas.to_numeric(downincome12["GeoFips"], errors="coerce")
downincome12["2012"] = pandas.to_numeric(downincome12["2012"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank12["cert"] = pandas.to_numeric(downbank12["cert"], errors="coerce")
downbank12["docket"] = pandas.to_numeric(downbank12["docket"], errors="coerce")
downbank12["rssdhcr"] = pandas.to_numeric(downbank12["rssdhcr"], errors="coerce")
downbank12["fed_rssd"] = pandas.to_numeric(downbank12["fed_rssd"], errors="coerce")

downbank12["hctmult"] = pandas.to_numeric(downbank12["hctmult"], errors="coerce")
downbank12["zip"] = pandas.to_numeric(downbank12["zip"], errors="coerce")


downbank12["lncon"] = pandas.to_numeric(downbank12["lncon"], errors="coerce")
downbank12["drcon"] = pandas.to_numeric(downbank12["drcon"], errors="coerce")
downbank12["crcon"] = pandas.to_numeric(downbank12["crcon"], errors="coerce")
downbank12["ntcon"] = pandas.to_numeric(downbank12["ntcon"], errors="coerce")



downbank12["lnre"] = pandas.to_numeric(downbank12["lnre"], errors="coerce")
downbank12["drre"] = pandas.to_numeric(downbank12["drre"], errors="coerce")
downbank12["crre"] = pandas.to_numeric(downbank12["crre"], errors="coerce")
downbank12["ntre"] = pandas.to_numeric(downbank12["ntre"], errors="coerce")




#%%

downbank12 =  downbank12.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree12= pd.merge(downincome12, downbank12, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree12 = pritthree12.sort_values('county', ascending=False)
pritthree12 = pritthree12.drop_duplicates(subset='county', keep='first')




#%%


downincome13 = pd.read_csv('download2013.csv', sep=';',error_bad_lines=False)

#%%
downbank13 = pd.read_csv('SDI_Download_Data2013.csv', sep=',',error_bad_lines=False)



#%%


downincome13["GeoFips"] = pandas.to_numeric(downincome13["GeoFips"], errors="coerce")
downincome13["2013"] = pandas.to_numeric(downincome13["2013"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank13["cert"] = pandas.to_numeric(downbank13["cert"], errors="coerce")
downbank13["docket"] = pandas.to_numeric(downbank13["docket"], errors="coerce")
downbank13["rssdhcr"] = pandas.to_numeric(downbank13["rssdhcr"], errors="coerce")
downbank13["fed_rssd"] = pandas.to_numeric(downbank13["fed_rssd"], errors="coerce")

downbank13["hctmult"] = pandas.to_numeric(downbank13["hctmult"], errors="coerce")
downbank13["zip"] = pandas.to_numeric(downbank13["zip"], errors="coerce")


downbank13["lncon"] = pandas.to_numeric(downbank13["lncon"], errors="coerce")
downbank13["drcon"] = pandas.to_numeric(downbank13["drcon"], errors="coerce")
downbank13["crcon"] = pandas.to_numeric(downbank13["crcon"], errors="coerce")
downbank13["ntcon"] = pandas.to_numeric(downbank13["ntcon"], errors="coerce")


downbank13["lnre"] = pandas.to_numeric(downbank13["lnre"], errors="coerce")
downbank13["drre"] = pandas.to_numeric(downbank13["drre"], errors="coerce")
downbank13["crre"] = pandas.to_numeric(downbank13["crre"], errors="coerce")
downbank13["ntre"] = pandas.to_numeric(downbank13["ntre"], errors="coerce")




#%%

downbank13 =  downbank13.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()

#%%

pritthree13= pd.merge(downincome13, downbank13, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree13 = pritthree13.sort_values('county', ascending=False)
pritthree13 = pritthree13.drop_duplicates(subset='county', keep='first')



#%%


downincome14 = pd.read_csv('download2014.csv', sep=';',error_bad_lines=False)

#%%
downbank14 = pd.read_csv('SDI_Download_Data2014.csv', sep=',',error_bad_lines=False)



#%%


downincome14["GeoFips"] = pandas.to_numeric(downincome14["GeoFips"], errors="coerce")
downincome14["2014"] = pandas.to_numeric(downincome14["2014"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank14["cert"] = pandas.to_numeric(downbank14["cert"], errors="coerce")
downbank14["docket"] = pandas.to_numeric(downbank14["docket"], errors="coerce")
downbank14["rssdhcr"] = pandas.to_numeric(downbank14["rssdhcr"], errors="coerce")
downbank14["fed_rssd"] = pandas.to_numeric(downbank14["fed_rssd"], errors="coerce")

downbank14["hctmult"] = pandas.to_numeric(downbank14["hctmult"], errors="coerce")
downbank14["zip"] = pandas.to_numeric(downbank14["zip"], errors="coerce")


downbank14["lncon"] = pandas.to_numeric(downbank14["lncon"], errors="coerce")
downbank14["drcon"] = pandas.to_numeric(downbank14["drcon"], errors="coerce")
downbank14["crcon"] = pandas.to_numeric(downbank14["crcon"], errors="coerce")
downbank14["ntcon"] = pandas.to_numeric(downbank14["ntcon"], errors="coerce")


downbank14["lnre"] = pandas.to_numeric(downbank14["lnre"], errors="coerce")
downbank14["drre"] = pandas.to_numeric(downbank14["drre"], errors="coerce")
downbank14["crre"] = pandas.to_numeric(downbank14["crre"], errors="coerce")
downbank14["ntre"] = pandas.to_numeric(downbank14["ntre"], errors="coerce")




#%%

downbank14 =  downbank14.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree14= pd.merge(downincome14, downbank14, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree14 = pritthree14.sort_values('county', ascending=False)
pritthree14 = pritthree14.drop_duplicates(subset='county', keep='first')



#%%


downincome15 = pd.read_csv('download2015.csv', sep=';',error_bad_lines=False)

#%%
downbank15 = pd.read_csv('SDI_Download_Data2015.csv', sep=',',error_bad_lines=False)



#%%


downincome15["GeoFips"] = pandas.to_numeric(downincome15["GeoFips"], errors="coerce")
downincome15["2015"] = pandas.to_numeric(downincome15["2015"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank15["cert"] = pandas.to_numeric(downbank15["cert"], errors="coerce")
downbank15["docket"] = pandas.to_numeric(downbank15["docket"], errors="coerce")
downbank15["rssdhcr"] = pandas.to_numeric(downbank15["rssdhcr"], errors="coerce")
downbank15["fed_rssd"] = pandas.to_numeric(downbank15["fed_rssd"], errors="coerce")

downbank15["hctmult"] = pandas.to_numeric(downbank15["hctmult"], errors="coerce")
downbank15["zip"] = pandas.to_numeric(downbank15["zip"], errors="coerce")


downbank15["lncon"] = pandas.to_numeric(downbank15["lncon"], errors="coerce")
downbank15["drcon"] = pandas.to_numeric(downbank15["drcon"], errors="coerce")
downbank15["crcon"] = pandas.to_numeric(downbank15["crcon"], errors="coerce")
downbank15["ntcon"] = pandas.to_numeric(downbank15["ntcon"], errors="coerce")


downbank15["lnre"] = pandas.to_numeric(downbank15["lnre"], errors="coerce")
downbank15["drre"] = pandas.to_numeric(downbank15["drre"], errors="coerce")
downbank15["crre"] = pandas.to_numeric(downbank15["crre"], errors="coerce")
downbank15["ntre"] = pandas.to_numeric(downbank15["ntre"], errors="coerce")




#%%

downbank15 =  downbank15.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()

#%%

pritthree15= pd.merge(downincome15, downbank15, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree15 = pritthree15.sort_values('county', ascending=False)
pritthree15 = pritthree15.drop_duplicates(subset='county', keep='first')



#%%


downincome16 = pd.read_csv('download2016.csv', sep=';',error_bad_lines=False)

#%%
downbank16 = pd.read_csv('SDI_Download_Data2016.csv', sep=',',error_bad_lines=False)



#%%


downincome16["GeoFips"] = pandas.to_numeric(downincome16["GeoFips"], errors="coerce")
downincome16["2016"] = pandas.to_numeric(downincome16["2016"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank16["cert"] = pandas.to_numeric(downbank16["cert"], errors="coerce")
downbank16["docket"] = pandas.to_numeric(downbank16["docket"], errors="coerce")
downbank16["rssdhcr"] = pandas.to_numeric(downbank16["rssdhcr"], errors="coerce")
downbank16["fed_rssd"] = pandas.to_numeric(downbank16["fed_rssd"], errors="coerce")

downbank16["hctmult"] = pandas.to_numeric(downbank16["hctmult"], errors="coerce")
downbank16["zip"] = pandas.to_numeric(downbank16["zip"], errors="coerce")


downbank16["lncon"] = pandas.to_numeric(downbank16["lncon"], errors="coerce")
downbank16["drcon"] = pandas.to_numeric(downbank16["drcon"], errors="coerce")
downbank16["crcon"] = pandas.to_numeric(downbank16["crcon"], errors="coerce")
downbank16["ntcon"] = pandas.to_numeric(downbank16["ntcon"], errors="coerce")


downbank16["lnre"] = pandas.to_numeric(downbank16["lnre"], errors="coerce")
downbank16["drre"] = pandas.to_numeric(downbank16["drre"], errors="coerce")
downbank16["crre"] = pandas.to_numeric(downbank16["crre"], errors="coerce")
downbank16["ntre"] = pandas.to_numeric(downbank16["ntre"], errors="coerce")




#%%

downbank16 =  downbank16.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree16= pd.merge(downincome16, downbank16, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree16 = pritthree16.sort_values('county', ascending=False)
pritthree16 = pritthree16.drop_duplicates(subset='county', keep='first')



#%%


downincome17 = pd.read_csv('download2017.csv', sep=';',error_bad_lines=False)

#%%
downbank17 = pd.read_csv('SDI_Download_Data2017.csv', sep=',',error_bad_lines=False)



#%%


downincome17["GeoFips"] = pandas.to_numeric(downincome17["GeoFips"], errors="coerce")
downincome17["2017"] = pandas.to_numeric(downincome17["2017"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank17["cert"] = pandas.to_numeric(downbank17["cert"], errors="coerce")
downbank17["docket"] = pandas.to_numeric(downbank17["docket"], errors="coerce")
downbank17["rssdhcr"] = pandas.to_numeric(downbank17["rssdhcr"], errors="coerce")
downbank17["fed_rssd"] = pandas.to_numeric(downbank17["fed_rssd"], errors="coerce")

downbank17["hctmult"] = pandas.to_numeric(downbank17["hctmult"], errors="coerce")
downbank17["zip"] = pandas.to_numeric(downbank17["zip"], errors="coerce")


downbank17["lncon"] = pandas.to_numeric(downbank17["lncon"], errors="coerce")
downbank17["drcon"] = pandas.to_numeric(downbank17["drcon"], errors="coerce")
downbank17["crcon"] = pandas.to_numeric(downbank17["crcon"], errors="coerce")
downbank17["ntcon"] = pandas.to_numeric(downbank17["ntcon"], errors="coerce")


downbank17["lnre"] = pandas.to_numeric(downbank17["lnre"], errors="coerce")
downbank17["drre"] = pandas.to_numeric(downbank17["drre"], errors="coerce")
downbank17["crre"] = pandas.to_numeric(downbank17["crre"], errors="coerce")
downbank17["ntre"] = pandas.to_numeric(downbank17["ntre"], errors="coerce")




#%%

downbank17 =  downbank17.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%

pritthree17= pd.merge(downincome17, downbank17, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree17 = pritthree17.sort_values('county', ascending=False)
pritthree17 = pritthree17.drop_duplicates(subset='county', keep='first')



#%%


downincome18 = pd.read_csv('download2018.csv', sep=';',error_bad_lines=False)

#%%
downbank18 = pd.read_csv('SDI_Download_Data2018.csv', sep=',',error_bad_lines=False)



#%%

downincome18["GeoFips"] = pandas.to_numeric(downincome18["GeoFips"], errors="coerce")
downincome18["2018"] = pandas.to_numeric(downincome18["2018"], errors="coerce")


#downincome.set_index('county','stalp')
#%%
downbank18["cert"] = pandas.to_numeric(downbank18["cert"], errors="coerce")
downbank18["docket"] = pandas.to_numeric(downbank18["docket"], errors="coerce")
downbank18["rssdhcr"] = pandas.to_numeric(downbank18["rssdhcr"], errors="coerce")
downbank18["fed_rssd"] = pandas.to_numeric(downbank18["fed_rssd"], errors="coerce")

downbank18["hctmult"] = pandas.to_numeric(downbank18["hctmult"], errors="coerce")
downbank18["zip"] = pandas.to_numeric(downbank18["zip"], errors="coerce")


downbank18["lncon"] = pandas.to_numeric(downbank18["lncon"], errors="coerce")
downbank18["drcon"] = pandas.to_numeric(downbank18["drcon"], errors="coerce")
downbank18["crcon"] = pandas.to_numeric(downbank18["crcon"], errors="coerce")
downbank18["ntcon"] = pandas.to_numeric(downbank18["ntcon"], errors="coerce")


downbank18["lnre"] = pandas.to_numeric(downbank18["lnre"], errors="coerce")
downbank18["drre"] = pandas.to_numeric(downbank18["drre"], errors="coerce")
downbank18["crre"] = pandas.to_numeric(downbank18["crre"], errors="coerce")
downbank18["ntre"] = pandas.to_numeric(downbank18["ntre"], errors="coerce")




#%%

downbank18 =  downbank18.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()

#%%



pritthree18= pd.merge(downincome18, downbank18, on=['county'], how='right')


#%% I sort an remove the  duplicates

pritthree18 = pritthree18.sort_values('county', ascending=False)
pritthree18 = pritthree18.drop_duplicates(subset='county', keep='first')


#%%



downincome19 = pd.read_csv('download2019.csv', sep=',',error_bad_lines=False)

#%%
downbank19 = pd.read_csv('SDI_Download_Data2019.csv', sep=',',error_bad_lines=False)


#%%

downincome19["GeoFips"] = pandas.to_numeric(downincome19["GeoFips"], errors="coerce")
downincome19["2019"] = pandas.to_numeric(downincome19["2019"], errors="coerce")



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


downincome19['stalp'] = downincome19['GeoName'].map(us_state_abbrev)



#downincome.set_index('county','stalp')
#%%
downbank19["cert"] = pandas.to_numeric(downbank19["cert"], errors="coerce")
downbank19["docket"] = pandas.to_numeric(downbank19["docket"], errors="coerce")
downbank19["rssdhcr"] = pandas.to_numeric(downbank19["rssdhcr"], errors="coerce")
downbank19["fed_rssd"] = pandas.to_numeric(downbank19["fed_rssd"], errors="coerce")

downbank19["hctmult"] = pandas.to_numeric(downbank19["hctmult"], errors="coerce")
downbank19["zip"] = pandas.to_numeric(downbank19["zip"], errors="coerce")


downbank19["lncon"] = pandas.to_numeric(downbank19["lncon"], errors="coerce")
downbank19["drcon"] = pandas.to_numeric(downbank19["drcon"], errors="coerce")
downbank19["crcon"] = pandas.to_numeric(downbank19["crcon"], errors="coerce")
downbank19["ntcon"] = pandas.to_numeric(downbank19["ntcon"], errors="coerce")



downbank19["lnre"] = pandas.to_numeric(downbank19["lnre"], errors="coerce")
downbank19["drre"] = pandas.to_numeric(downbank19["drre"], errors="coerce")
downbank19["crre"] = pandas.to_numeric(downbank19["crre"], errors="coerce")
downbank19["ntre"] = pandas.to_numeric(downbank19["ntre"], errors="coerce")




#%%

downbank19 =  downbank19.groupby("county", sort=False)["drcon","crcon","ntcon","lncon","lnre","drre","crre","ntre"].sum()


#%%


#%%




#%%

downbank19["stalp"] = downbank19.index

#%%

downbank19.reset_index(drop=True, inplace=True)


#del downbank19['Index']
#downbank19.reset_index()
#index.downbank19 = None
#%%

pritthree19= pd.merge(downincome19, downbank19, on=['stalp'], how='right')


#%%
pritthree19 = pritthree19.dropna()


#%% CREATE A COLUMN WITH THE YEAR  FOR EACH DATA SET
#pritthree92['A'] = '1992'
#pritthree93['A'] = '1993'
#pritthree94['A'] = '1994'
#pritthree95['A'] = '1995'
#pritthree96['A'] = '1996'
pritthree97['A'] = '1997'
pritthree98['A'] = '1998'
pritthree99['A'] = '1999'
pritthree00['A'] = '2000'
pritthree01['A'] = '2001'
pritthree02['A'] = '2002'
pritthree03['A'] = '2003'
pritthree04['A'] = '2004'
pritthree05['A'] = '2005'
pritthree06['A'] = '2006'
pritthree07['A'] = '2007'
pritthree08['A'] = '2008'
pritthree09['A'] = '2009'
pritthree10['A'] = '2010'
pritthree11['A'] = '2011'
pritthree12['A'] = '2012'
pritthree13['A'] = '2013'
pritthree14['A'] = '2014'
pritthree15['A'] = '2015'
pritthree16['A'] = '2016'
pritthree17['A'] = '2017'
pritthree18['A'] = '2018'
pritthree19['A'] = '2019'


#%% RENAME THE COLUMN OF PERSNAL INCOME SO WE HAVE A UNIFIED 
#pritthree92 = pritthree92.rename(columns={"1992": "perincom"})
#pritthree93 = pritthree93.rename(columns={"1993": "perincom"})
#pritthree94 = pritthree94.rename(columns={"1994": "perincom"})
#pritthree95 = pritthree95.rename(columns={"1995": "perincom"})
#pritthree96 = pritthree96.rename(columns={"1996": "perincom"})
pritthree97 = pritthree97.rename(columns={"1997": "perincom"})
pritthree98 = pritthree98.rename(columns={"1998": "perincom"})
pritthree99 = pritthree99.rename(columns={"1999": "perincom"})
pritthree00 = pritthree00.rename(columns={"2000": "perincom"})
pritthree01 = pritthree01.rename(columns={"2001": "perincom"})
pritthree02 = pritthree02.rename(columns={"2002": "perincom"})
pritthree03 = pritthree03.rename(columns={"2003": "perincom"})
pritthree04 = pritthree04.rename(columns={"2004": "perincom"})
pritthree05 = pritthree05.rename(columns={"2005": "perincom"})
pritthree06 = pritthree06.rename(columns={"2006": "perincom"})
pritthree07 = pritthree07.rename(columns={"2007": "perincom"})
pritthree08 = pritthree08.rename(columns={"2008": "perincom"})
pritthree09 = pritthree09.rename(columns={"2009": "perincom"})
pritthree10 = pritthree10.rename(columns={"2010": "perincom"})
pritthree11 = pritthree11.rename(columns={"2011": "perincom"})
pritthree12 = pritthree12.rename(columns={"2012": "perincom"})
pritthree13 = pritthree13.rename(columns={"2013": "perincom"})
pritthree14 = pritthree14.rename(columns={"2014": "perincom"})
pritthree15 = pritthree15.rename(columns={"2015": "perincom"})
pritthree16 = pritthree16.rename(columns={"2016": "perincom"})
pritthree17 = pritthree17.rename(columns={"2017": "perincom"})
pritthree18 = pritthree18.rename(columns={"2018": "perincom"})
pritthree19 = pritthree19.rename(columns={"2019": "perincom"})


#%%

#examplepd =pritthree92.append(pritthree93).reset_index(drop=True)
#examplepd =examplepd.append(pritthree94).reset_index(drop=True)
#examplepd =examplepd.append(pritthree95).reset_index(drop=True)
#examplepd =examplepd.append(pritthree96).reset_index(drop=True)
#examplepd =examplepd.append(pritthree97).reset_index(drop=True)
examplepd =pritthree97.append(pritthree98).reset_index(drop=True)
examplepd =examplepd.append(pritthree99).reset_index(drop=True)
examplepd =examplepd.append(pritthree00).reset_index(drop=True)
examplepd =examplepd.append(pritthree01).reset_index(drop=True)
examplepd =examplepd.append(pritthree02).reset_index(drop=True)
examplepd =examplepd.append(pritthree03).reset_index(drop=True)
examplepd =examplepd.append(pritthree04).reset_index(drop=True)
examplepd =examplepd.append(pritthree05).reset_index(drop=True)
examplepd =examplepd.append(pritthree06).reset_index(drop=True)
examplepd =examplepd.append(pritthree07).reset_index(drop=True)
examplepd =examplepd.append(pritthree08).reset_index(drop=True)
examplepd =examplepd.append(pritthree09).reset_index(drop=True)
examplepd =examplepd.append(pritthree10).reset_index(drop=True)
examplepd =examplepd.append(pritthree11).reset_index(drop=True)
examplepd =examplepd.append(pritthree12).reset_index(drop=True)
examplepd =examplepd.append(pritthree13).reset_index(drop=True)
examplepd =examplepd.append(pritthree14).reset_index(drop=True)
examplepd =examplepd.append(pritthree15).reset_index(drop=True)
examplepd =examplepd.append(pritthree16).reset_index(drop=True)
examplepd =examplepd.append(pritthree17).reset_index(drop=True)
examplepd =examplepd.append(pritthree18).reset_index(drop=True)
examplepd =examplepd.append(pritthree19).reset_index(drop=True)



#%% I drop the unnamed  columns 
examplepd = examplepd.drop(examplepd.columns[[4]], axis=1)  # df.columns is zero-based pd.Index 
#%%
examplepd = examplepd.drop(['GeoName', 'county'], axis=1)


#%% I drop observations with nan in personal income
#examplepd.dropna(subset = ["perincom"], inplace=True) #231379

examplepd = examplepd.dropna() #230941
#%%

examplepd["lncon"] = pandas.to_numeric(examplepd["lncon"], errors="coerce")
examplepd["drcon"] = pandas.to_numeric(examplepd["drcon"], errors="coerce")
#examplepd["cert"] = pandas.to_numeric(examplepd["cert"], errors="coerce")
#examplepd["cert2"] = pandas.to_numeric(examplepd["cert"], errors="coerce")
examplepd["A"] = pandas.to_numeric(examplepd["A"], errors="coerce")
examplepd["A2"] = pandas.to_numeric(examplepd["A"], errors="coerce")


examplepd["lnre"] = pandas.to_numeric(examplepd["lnre"], errors="coerce")
examplepd["drre"] = pandas.to_numeric(examplepd["drre"], errors="coerce")
examplepd["crre"] = pandas.to_numeric(examplepd["crre"], errors="coerce")
examplepd["ntre"] = pandas.to_numeric(examplepd["ntre"], errors="coerce")



#%% export ro excel

examplepd.to_excel('databycountyus9218re.xlsx')
