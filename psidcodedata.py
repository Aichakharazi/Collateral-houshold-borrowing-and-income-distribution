# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:14:19 2021

@author: Aicha-PC
"""


#%% run this code in old computer -  better 


import csv
import pandas
#%%
from matplotlib import pyplot as plt
#%%
#from chart-studio import pyplot as ply
from chart_studio import plotly
#%%

import numpy as np
import pandas as pd
#%%
import plotly
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.graph_objs as go
#%%
import addfips
import numpy 
import numpy as np
import pandas
import pandas as pd
#%%
import plotly.figure_factory as ff
import plotly
import plotly.graph_objs as go
#%%
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels.panel import PooledOLS
import statsmodels
import statsmodels.api as sm
from linearmodels.panel import compare
import seaborn 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas import DataFrame
import statsmodels.formula.api as smf
import regex as re
import spacy
import csv
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.iolib.summary3 import summary_col
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style
style.available

#%% 

import pandas as pd

import pandas as pd 
import pandas as pd 

#%%
psiddebt1 = pd.read_csv("psid_debt.csv", delimiter=';', encoding='cp1252') 

psiddebt1.head()

#%%


import pandas as pd 

psidwealth1 = pd.read_csv("psid_wealthstateJ288850.csv", delimiter=';', encoding='cp1252') 

psidwealth1.head()

#%%
new99psid =psidwealth1[['statecode', '1999IDNUMBER', 'IMPWEALTHWEQUITYWEALTH299']]
#%%
new2001psid =psidwealth1[['statecode', '2001IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22001']]
#%%
new2003psid =psidwealth1[['statecode', '2003IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22003']]
#%%
new2005psid =psidwealth1[['statecode', '2005IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22005']]
#%%
new2007psid =psidwealth1[['statecode', '2007IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22007']]
#%%
new2009psid =psidwealth1[['statecode', '2009IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22009']]
#%%
new2011psid =psidwealth1[['statecode', '2011IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22011']]
#%%
new2013psid =psidwealth1[['statecode', '2013IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22013']]
#%%
new2015psid =psidwealth1[['statecode', '2015IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22015']]
#%%
new2017psid =psidwealth1[['statecode', '2017IDNUMBER', 'IMPWEALTHWEQUITYWEALTH22017']]
#%%
new99psid['Year'] = '1999'
new2001psid['Year'] = '2001'
new2003psid['Year'] = '2003'
new2005psid['Year'] = '2005'
new2007psid['Year'] = '2007'
new2009psid['Year'] = '2009'
new2011psid['Year'] = '2011'
new2013psid['Year'] = '2013'
new2015psid['Year'] = '2015'
new2017psid['Year'] = '2017'
#%%
new99psid = new99psid.rename(columns={"1999IDNUMBER": "Idnumber"})
new99psid = new99psid.rename(columns={"IMPWEALTHWEQUITYWEALTH299": "wealth"})

new2001psid = new2001psid.rename(columns={"2001IDNUMBER": "Idnumber"})
new2001psid = new2001psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22001": "wealth"})

new2003psid = new2003psid.rename(columns={"2003IDNUMBER": "Idnumber"})
new2003psid = new2003psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22003": "wealth"})

new2005psid = new2005psid.rename(columns={"2005IDNUMBER": "Idnumber"})
new2005psid = new2005psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22005": "wealth"})

new2007psid = new2007psid.rename(columns={"2007IDNUMBER": "Idnumber"})
new2007psid = new2007psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22007": "wealth"})

new2009psid = new2009psid.rename(columns={"2009IDNUMBER": "Idnumber"})
new2009psid = new2009psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22009": "wealth"})

new2011psid = new2011psid.rename(columns={"2011IDNUMBER": "Idnumber"})
new2011psid = new2011psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22011": "wealth"})

new2013psid = new2013psid.rename(columns={"2013IDNUMBER": "Idnumber"})
new2013psid = new2013psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22013": "wealth"})

new2015psid = new2015psid.rename(columns={"2015IDNUMBER": "Idnumber"})
new2015psid = new2015psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22015": "wealth"})

new2017psid = new2017psid.rename(columns={"2017IDNUMBER": "Idnumber"})
new2017psid = new2017psid.rename(columns={"IMPWEALTHWEQUITYWEALTH22017": "wealth"})
#%%
examplepd =new99psid.append(new2001psid).reset_index(drop=True)
examplepd =examplepd.append(new2003psid).reset_index(drop=True)
examplepd =examplepd.append(new2005psid).reset_index(drop=True)
examplepd =examplepd.append(new2007psid).reset_index(drop=True)
examplepd =examplepd.append(new2009psid).reset_index(drop=True)
examplepd =examplepd.append(new2011psid).reset_index(drop=True)
examplepd =examplepd.append(new2013psid).reset_index(drop=True)
examplepd =examplepd.append(new2015psid).reset_index(drop=True)
examplepd =examplepd.append(new2017psid).reset_index(drop=True)
#%%



debt99psid =psiddebt1[['1999FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW3999']]

debt01psid =psiddebt1[['2001FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW3901']]
 
debt03psid =psiddebt1[['2003FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW3903']]

debt05psid =psiddebt1[['2005FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW3905']]

debt07psid =psiddebt1[['2007FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW3907']]

debt09psid =psiddebt1[['2009FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW3909']]

debt11psid =psiddebt1[['2011FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW392011']]

debt13psid =psiddebt1[['2013FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW392013']]

debt15psid =psiddebt1[['2015FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW32015']]

debt17psid =psiddebt1[['2017FAMILYINTERVIEWIDNUMBER', 'IMPVALUEOTHDEBTW392017']]


#%%


debt99psid['Year'] = '1999'
debt01psid['Year'] = '2001'
debt03psid['Year'] = '2003'
debt05psid['Year'] = '2005'
debt07psid['Year'] = '2007'
debt09psid['Year'] = '2009'
debt11psid['Year'] = '2011'
debt13psid['Year'] = '2013'
debt15psid['Year'] = '2015'
debt17psid['Year'] = '2017'


#%%


debt99psid = debt99psid.rename(columns={"1999FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt99psid = debt99psid.rename(columns={"IMPVALUEOTHDEBTW3999": "debt"})


debt01psid =debt01psid.rename(columns={"2001FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt01psid =debt01psid.rename(columns={"IMPVALUEOTHDEBTW3901": "debt"})
 
debt03psid =debt03psid.rename(columns={"2003FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt03psid =debt03psid.rename(columns={"IMPVALUEOTHDEBTW3903": "debt"})

debt05psid =debt05psid.rename(columns={"2005FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt05psid =debt05psid.rename(columns={"IMPVALUEOTHDEBTW3905": "debt"})

debt07psid =debt07psid.rename(columns={"2007FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt07psid =debt07psid.rename(columns={"IMPVALUEOTHDEBTW3907": "debt"})

debt09psid =debt09psid.rename(columns={"2009FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt09psid =debt09psid.rename(columns={"IMPVALUEOTHDEBTW3909": "debt"})

debt11psid =debt11psid.rename(columns={"2011FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt11psid =debt11psid.rename(columns={"IMPVALUEOTHDEBTW392011": "debt"})

debt13psid =debt13psid.rename(columns={"2013FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt13psid =debt13psid.rename(columns={"IMPVALUEOTHDEBTW392013": "debt"})

debt15psid =debt15psid.rename(columns={"2015FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt15psid =debt15psid.rename(columns={"IMPVALUEOTHDEBTW32015": "debt"})

debt17psid =debt17psid.rename(columns={"2017FAMILYINTERVIEWIDNUMBER": "Idnumber"})
debt17psid =debt17psid.rename(columns={"IMPVALUEOTHDEBTW392017": "debt"})

#%%
examplepd2 =debt99psid.append(debt01psid).reset_index(drop=True)
examplepd2 =examplepd2.append(debt03psid).reset_index(drop=True)
examplepd2 =examplepd2.append(debt05psid).reset_index(drop=True)
examplepd2 =examplepd2.append(debt07psid).reset_index(drop=True)
examplepd2 =examplepd2.append(debt09psid).reset_index(drop=True)
examplepd2 =examplepd2.append(debt11psid).reset_index(drop=True)
examplepd2 =examplepd2.append(debt13psid).reset_index(drop=True)
examplepd2 =examplepd2.append(debt15psid).reset_index(drop=True)
examplepd2 =examplepd2.append(debt17psid).reset_index(drop=True)




#%%
examplepd21 = examplepd2[examplepd2['Idnumber'].notna()]

#%%


examplepd31 = examplepd[examplepd['Idnumber'].notna()]

#%%
mergeddebtwealth1 = examplepd21.merge(examplepd31, left_on=['Year','Idnumber']
                                , right_on=['Year','Idnumber'],how='left')

mergeddebtwealth1.dtypes 
#%%
mergeddebtwealth1["Year"] = mergeddebtwealth1.Year.astype(float)


#%%
import numpy as np

#%% top states
filter_list1 = ['09','34','25','24','33','51','36','38','02','27','08','53','44','10','06','17','56','42','50']


#mergeddebtwealth1.loc[mergeddebtwealth1['statecode'] == '09']

#%% bottom states
filter_list2 = ['19','55','23','20','41','31','48','46','39','26','12','29','30','37','32','04','13','40','18','47','49','22','45','16','21','35','01','05','54','28']
#%%  for top sample 
df1 = mergeddebtwealth1.loc[mergeddebtwealth1['statecode'].isin(filter_list1)]

#%%
df1 = mergeddebtwealth1.loc[mergeddebtwealth1['statecode'].isin(filter_list1)]


#%% for the sample bottom per capita income below the US national average  
df2 = mergeddebtwealth1.loc[mergeddebtwealth1['statecode'].isin(filter_list2)]	


#%%
df1 = df1.assign(change=df1.wealth - df1.groupby(['Idnumber']).wealth.shift(1))






#%%

df2 = df2.assign(change= df2.wealth - df2.groupby(['Idnumber']).wealth.shift(1))




#%%


#%%
df1x =df1.copy()
#%%
df2x =df2.copy()

#%%

df2x['logwealth']=np.log(df2x['wealth'])
#%%

df2x['logdebt']=np.log(df2x['debt'])

#%%
df2x.replace([np.inf,-np.inf],np.nan,inplace=True)

#%%
df2x.dropna(subset=["logdebt"],inplace=True)

df2x.dropna(subset=["logwealth"],inplace=True)
#%%


#%% bottom sample

import matplotlib.style as style
style.use('seaborn-white')        
        


sns.jointplot(data=df2x, x='logdebt', y='logwealth', kind='reg', color='#0F95D7')


ax = plt.gca()





#plt.grid(True)
plt.axis('tight')
plt.xlabel('Debt',fontsize = 20)
plt.ylabel('Wealth ',fontsize = 20)


plt.savefig('motivaj.png', bbox_inches='tight')






#%%


df1x['logwealth']=np.log(df1x['wealth'])

#%%
df1x['logdebt']=np.log(df1x['debt'])

#%%
df1x.replace([np.inf,-np.inf],np.nan,inplace=True)

#%%
df1x.dropna(subset=["logdebt"],inplace=True)

df1x.dropna(subset=["logwealth"],inplace=True)
#%%


#%%

import matplotlib.style as style
style.use('seaborn-white')        
        
sns.jointplot(data=df1x, x='logdebt', y='logwealth', kind='reg', color='#0F95D7')

ax = plt.gca()

plt.axis('tight')
plt.xlabel('Debt',fontsize = 20)
plt.ylabel('Wealth ',fontsize = 20)

plt.savefig('motivaj2.png', bbox_inches='tight')




#%%

import matplotlib.style as style
style.use('seaborn-white')        
        
sns.jointplot(data=df2x, x='logdebt', y='logwealth', kind='reg', color='#0F95D7')

ax = plt.gca()

plt.axis('tight')
plt.xlabel('Debt',fontsize = 20)
plt.ylabel('Wealth ',fontsize = 20)

plt.savefig('motivaj3.png', bbox_inches='tight')


#%%



#%%




import matplotlib.style as style
#fig.set_size_inches(12, 8)
sns.set(font_scale=1.7)
style.use('seaborn-white')  

sns.pairplot(mergeddebtwealth1, hue ="Year", vars =['wealth'], height=10, palette="Blues")

plt.savefig('plotfordensitywealthA', bbox_inches='tight')



#%%


df1  = df1.set_index(['Idnumber','Year'])
#%%

exog_vars = ['debt']
exog = sm.add_constant(df1[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealth,exog,entity_effects=True,time_effects=True)
two_res11re = cons.fit()
print(two_res11re)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealth,exog,entity_effects=False,time_effects=True)
two_res22re = cons.fit()
print(two_res22re)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealth,exog,entity_effects=True,time_effects=False)
two_res33re = cons.fit()
print(two_res33re)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.wealth,exog,entity_effects=False,time_effects=False)
two_res44re = cons.fit()
print(two_res44re)


#%%

print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.change,exog,entity_effects=True,time_effects=True)
change_res11re = cons.fit()
print(change_res11re)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.change,exog,entity_effects=False,time_effects=True)
change_res22re = cons.fit()
print(change_res22re)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.change,exog,entity_effects=True,time_effects=False)
change_res33re = cons.fit()
print(change_res33re)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df1.change,exog,entity_effects=False,time_effects=False)
change_res44re = cons.fit()
print(change_res44re)



#%%

df2  = df2.set_index(['Idnumber','Year'])
#%%

exog_vars = ['debt']
exog = sm.add_constant(df2[exog_vars])
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealth,exog,entity_effects=True,time_effects=True)
two_res11re2 = cons.fit()
print(two_res11re2)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealth,exog,entity_effects=False,time_effects=True)
two_res22re2 = cons.fit()
print(two_res22re2)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealth,exog,entity_effects=True,time_effects=False)
two_res33re2 = cons.fit()
print(two_res33re2)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.wealth,exog,entity_effects=False,time_effects=False)
two_res44re2 = cons.fit()
print(two_res44re2)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.change,exog,entity_effects=True,time_effects=True )
change_res11re2 = cons.fit()
print(change_res11re2)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.change,exog,entity_effects=False,time_effects=True)
change_res22re2 = cons.fit()
print(change_res22re2)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.change,exog,entity_effects=True,time_effects=False)
change_res33re2 = cons.fit()
print(change_res33re2)
#%%
print ("OLS regression model for the association between  ")
cons = PanelOLS(df2.change,exog,entity_effects=False,time_effects=False)
change_res44re2 = cons.fit()
print(change_res44re2)


#%%


#%%




from statsmodels.iolib.summary3 import summary_col






beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsrefree11 = summary_col([two_res11re2,two_res22re2,two_res33re2,two_res44re2],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("PSIDrefree22.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%




from statsmodels.iolib.summary3 import summary_col






beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsrefree11 = summary_col([two_res11re,two_res22re,two_res33re,two_res44re],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("PSIDrefree11.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%










beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsrefree11 = summary_col([change_res11re2,change_res22re2,change_res33re2,change_res44re2],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("PSIDrefree22change.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%




from statsmodels.iolib.summary3 import summary_col






beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"



resultsrefree11 = summary_col([change_res11re,change_res22re,change_res33re,change_res44re],stars=True,float_format='%0.5f',show='se') 


dfoutput = open("PSIDrefree11change.tex",'w')
dfoutput.write(beginningtex)
dfoutput.write(resultsrefree11.as_latex())
dfoutput.write(endtex)
dfoutput.close()



#%%






import numpy as np
import matplotlib.pyplot as plt



#%% ensure wealth is sorted from lowest to highest values first!
mergeddebtwealth1['wealth2'] = mergeddebtwealth1['wealth']
mergeddebtwealth1.index = mergeddebtwealth1.wealth.astype(int)
mergeddebtwealth1 = mergeddebtwealth1.sort_index()


#%%  divides the prefix sum by the total sum
scaled_prefix_sum = mergeddebtwealth1['wealth2'].cumsum() / mergeddebtwealth1['wealth2'].sum()
#%%
import numpy as np
import pandas as pd

scaled_prefix_sum = scaled_prefix_sum.to_frame()
#%% add this value to the df
mergeddebtwealth1['retulorrenz'] = scaled_prefix_sum
#%%
lorenz_curve = mergeddebtwealth1['retulorrenz']
#%%

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
#from statsmodels.iolib.summary3 import summary_col
#from summary3 import summary_col 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style
style.available



style.use('seaborn-white')
ax = plt.gca()
x1 = np.linspace(0.0, 1.0, lorenz_curve.size)
# we need the X values to be between 0.0 to 1.0
plt.plot(x1, mergeddebtwealth1.retulorrenz,color='#0F95D7', linewidth=3,  linestyle='--')
# plot the straight line perfect equality curve
plt.plot([0,1], [0,1])



ax.set_ylabel("Cumulative Wealth",fontsize=16)
ax.set_xlabel("Cumulative Population",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
#ax.legend()
ax.autoscale(enable=True,axis='x',tight=True)
ax.autoscale(enable=True,axis='y',tight=True)




# 
plt.savefig('lorenzcurvewealthfin1.png', bbox_inches='tight')

plt.show()
#%%



#%%


#%%


#%% ensure debt is sorted from lowest to highest values first!
mergeddebtwealth1['debt2'] = mergeddebtwealth1['debt']
mergeddebtwealth1.index = mergeddebtwealth1.wealth.astype(int)
mergeddebtwealth1 = mergeddebtwealth1.sort_index()


#%%  divides the prefix sum by the total sum
scaled_prefix_sum = mergeddebtwealth1['debt2'].cumsum() / mergeddebtwealth1['debt2'].sum()
#%%
import numpy as np
import pandas as pd

scaled_prefix_sum = scaled_prefix_sum.to_frame()
#%% add this value to the df
mergeddebtwealth1['retulorrenzdebt'] = scaled_prefix_sum
#%%
lorenz_curvedebt = mergeddebtwealth1['retulorrenzdebt']
#%%

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
#from statsmodels.iolib.summary3 import summary_col
#from summary3 import summary_col 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style
style.available



style.use('seaborn-white')
ax = plt.gca()
x1 = np.linspace(0.0, 1.0, lorenz_curvedebt.size)
# we need the X values to be between 0.0 to 1.0
plt.plot(x1, mergeddebtwealth1.retulorrenzdebt)
# plot the straight line perfect equality curve
plt.plot([0,1], [0,1])



ax.set_ylabel("Cumulative Debt",fontsize=16)
ax.set_xlabel("Cumulative Population",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
ax.legend()
ax.autoscale(enable=True,axis='x',tight=True)
ax.autoscale(enable=True,axis='y',tight=True)


plt.savefig('lorenzcurvedebtfin1.png', bbox_inches='tight')
plt.show()

#%%






import numpy as np
import matplotlib.pyplot as plt



#%% ensure wealth is sorted from lowest to highest values first!
df1['wealth2'] = df1['wealth']
df1.index = df1.wealth.astype(int)
df1 = df1.sort_index()


#%%  divides the prefix sum by the total sum
scaled_prefix_sum = df1['wealth2'].cumsum() / df1['wealth2'].sum()
#%%
import numpy as np
import pandas as pd

scaled_prefix_sum = scaled_prefix_sum.to_frame()
#%% add this value to the df
df1['retulorrenz'] = scaled_prefix_sum
#%%
lorenz_curve = df1['retulorrenz']
#%%

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
#from statsmodels.iolib.summary3 import summary_col
#from summary3 import summary_col 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style
style.available



style.use('seaborn-white')
ax = plt.gca()
x1 = np.linspace(0.0, 1.0, lorenz_curve.size)
# we need the X values to be between 0.0 to 1.0
plt.plot(x1, df1.retulorrenz)
# plot the straight line perfect equality curve
plt.plot([0,1], [0,1])



ax.set_ylabel("Cumulative Wealth",fontsize=16)
ax.set_xlabel("Cumulative Population",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
ax.legend()
ax.autoscale(enable=True,axis='x',tight=True)
ax.autoscale(enable=True,axis='y',tight=True)

plt.show()




#%%

df2['wealth2'] = df2['wealth']
df2.index = df2.wealth.astype(int)
df2 = df2.sort_index()


#%%  divides the prefix sum by the total sum
scaled_prefix_sum = df2['wealth2'].cumsum() / df2['wealth2'].sum()
#%%
import numpy as np
import pandas as pd

scaled_prefix_sum = scaled_prefix_sum.to_frame()
#%% add this value to the df
df2['retulorrenz'] = scaled_prefix_sum
#%%
lorenz_curve = df2['retulorrenz']
#%%

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
#from statsmodels.iolib.summary3 import summary_col
#from summary3 import summary_col 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.style as style
style.available



style.use('seaborn-white')
ax = plt.gca()
x1 = np.linspace(0.0, 1.0, lorenz_curve.size)
# we need the X values to be between 0.0 to 1.0
plt.plot(x1, df2.retulorrenz)

# plot the straight line perfect equality curve
plt.plot([0,1], [0,1])



ax.set_ylabel("Cumulative Wealth",fontsize=16)
ax.set_xlabel("Cumulative Population",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
ax.legend()
ax.autoscale(enable=True,axis='x',tight=True)
ax.autoscale(enable=True,axis='y',tight=True)

plt.show()




#%%


df1["debttowealth"] = df1["debt"]/df1["wealth"] 
#%%

#%%
df2["debttowealth"] = df2["debt"]/df2["wealth"] 
#%%
#%%
mergeddebtwealth1xample= mergeddebtwealth1.groupby(['Year'])[["wealth"]].sum()

#%%
mergeddebtwealth1xample["debt"] = mergeddebtwealth1.groupby(['Year'])[["debt"]].sum()
#%%
mergeddebtwealth1xample["Year1"] = mergeddebtwealth1xample.index


#%%

mergeddebtwealth1xample['debtlog']=np.log(mergeddebtwealth1xample['debt'])
#%%
mergeddebtwealth1xample['wealthlog']=np.log(mergeddebtwealth1xample['wealth'])

#%% , marker="o"
from datetime import datetime
from matplotlib.dates import date2num
fig,ax = plt.subplots()
ax.plot(mergeddebtwealth1xample.Year1, mergeddebtwealth1xample.debtlog, color="#0F95D7")
ax.set_xlabel("Time",fontsize=16)
ax.set_ylabel("Log Debt ",fontsize=16)

ax.ticklabel_format(style='plain', scilimits=(0,0), axis='y')
ax.autoscale(enable=True,axis='x',tight=True)#
#ax.autoscale(enable=True,axis='y',tight=True)

plt.grid(axis = 'y', linestyle = '--', linewidth = 0.9, which='major')


plt.savefig('debtpsidovertimefin1.png', bbox_inches='tight')
plt.show()



#%%

df1['debtlog']=np.log(df1['debt'])
df1['wealthlog']=np.log(df1['wealth'])

df1['debttowealthlog']=np.log(df1['debttowealth'])

#%%
df2['wealthlog']=np.log(df2['wealth'])
df2['debtlog']=np.log(df2['debt'])
df2['debttowealthlog']=np.log(df2['debttowealth'])

#%%
import matplotlib.pyplot as plt
import seaborn as sns

fig
sns.regplot(x='debttowealthlog', y='wealthlog', data=df1)
ax = plt.gca()
ax.set_ylabel(" Log Debt to Wealth",fontsize=16)
ax.set_xlabel(" Log Debt",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
ax.legend()
plt.legend(fontsize=26)         

#%%

import matplotlib.pyplot as plt
import seaborn as sns

fig 
sns.regplot(x='debttowealthlog', y='wealthlog', data=df2)
ax = plt.gca()
ax.set_ylabel(" Log Debt to Wealth",fontsize=16)
ax.set_xlabel(" Log Debt",fontsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.yaxis.grid()
plt.legend(fontsize=26)    



#%% , dropna=True, inplace=True


# 
df1.replace([np.inf,-np.inf], np.nan,inplace=True)

import matplotlib.style as style
style.use('seaborn-white')        
        
sns.jointplot(data=df1, x='wealthlog', y='debtlog', kind='reg', color='#0F95D7')

ax = plt.gca()
#ax.get_xaxis().get_major_formatter().set_scientific(False)

plt.axis('tight')
plt.xlabel('Log Wealth',fontsize = 20)
plt.ylabel('Log Debt',fontsize = 20)

plt.savefig('logdebtlogwealthtopfin1.png', bbox_inches='tight')


#%%


# 
df2.replace([np.inf,-np.inf], np.nan,inplace=True)

import matplotlib.style as style
style.use('seaborn-white')        
        
sns.jointplot(data=df2, x='wealthlog', y='debtlog', kind='reg', color='#0F95D7')

ax = plt.gca()
#ax.get_xaxis().get_major_formatter().set_scientific(False)

plt.axis('tight')
plt.xlabel('Log Wealth',fontsize = 20)
plt.ylabel('Log Debt',fontsize = 20)

plt.savefig('logdebtlogwealthbottomfin1.png', bbox_inches='tight')
