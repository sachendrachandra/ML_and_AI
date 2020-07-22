import sys
import os
import pandas as pd
import numpy as np
import pickle
#import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
import itertools
from datetime import date,datetime,timedelta


start_date=sys.argv[1]
start_date=datetime.strptime(start_date,'%Y-%m-%d')
end_date=sys.argv[2]
end_date=datetime.strptime(end_date,'%Y-%m-%d')
output_folder=sys.argv[3]
os.makedirs(output_folder)
output_India=open(output_folder + '/India.pred','w')
output_Italy=open(output_folder + '/Italy.pred','w')
output_Belgium=open(output_folder + '/Belgium.pred','w')
output_US=open(output_folder + '/US.pred','w')

# output_India=open('output_India','w')

# start_date=date(2020,5,1)
# end_date=date(2020,5,7)
start=start_date
ending=end_date
model_end_date=datetime(2020,4,30)
delta=start_date-model_end_date
step=end_date-model_end_date
step=step.days
t=delta.days
m=t
delta = timedelta(days=1)

#INDIA-------------------------------
#covid_india_confirmed=pd.read_csv('/home2/e0268-26/a4/data/India.csv',index_col=0,sep='\t',usecols=[0,1])
#covid_india_recovered=pd.read_csv('/home2/e0268-26/a4/data/India.csv',index_col=0,sep='\t',usecols=[0,2])
#covid_india_death=pd.read_csv('/home2/e0268-26/a4/data/India.csv',index_col=0,sep='\t',usecols=[0,3])

# print(covid_india_recovered)
#X=covid_india_confirmed.values
#Y=covid_india_recovered.values
#Z=covid_india_death.values

#model_arima_confirmed=ARIMA(X,order=(5,2,4))
#model_arima_confirmed_fit=model_arima_confirmed.fit()
#with open('/home2/e0268-26/a4/model_confirmed.pkl','wb') as f:
#  pickle.dump(model_arima_confirmed_fit,f)
#print(model_arima_confirmed_fit.aic)
model_con_india=pickle.load(open('/home2/e0268-26/a4/model_confirmed.pkl','rb'))
predictions1=model_con_india.forecast(steps=step+1)[0]

#model_arima_recovered=ARIMA(Y,order=(3,2,5))
#model_arima_recovered_fit=model_arima_recovered.fit()
#with open('/home2/e0268-26/a4/model_recovered.pkl','wb') as f:
#  pickle.dump(model_arima_recovered_fit,f)
#print(model_arima_recovered_fit.aic)
model_rec_india=pickle.load(open('/home2/e0268-26/a4/model_recovered.pkl','rb'))
predictions2=model_rec_india.forecast(steps=step+1)[0]

#model_arima_death=ARIMA(Z,order=(7,2,2))
#model_arima_death_fit=model_arima_death.fit()
#with open('/home2/e0268-26/a4/model_death.pkl','wb') as f:
#  pickle.dump(model_arima_death_fit,f)
#print(model_arima_death_fit.aic)
model_death_india=pickle.load(open('/home2/e0268-26/a4/model_death.pkl','rb'))
predictions3=model_death_india.forecast(steps=step+1)[0]

# while start_date <= end_date:
#     print (start_date.strftime("%Y-%m-%d\t"),end="")
#     print("\t",end="")
#     print(predictions1[t],end="")
#     print("\t",end="")
#     print(predictions2[t],end="")
#     print("\t",end="")
#     print(predictions3[t],end="")
#     print('\n')
#     t=t+1
#     start_date += delta

start_date=start
end_date=ending
#t=delta.days
t=m
while start_date <= end_date:
  output_India.write(str(start_date.strftime("%Y-%m-%d"))+'\t')
  output_India.write(str(predictions1[t])+'\t')
  # output_India.write(str("\t"))
  output_India.write(str(predictions2[t])+'\t')
  # output_India.write(str("\t"))
  output_India.write(str(predictions3[t])+'\n')
  # output_India.write(str("\n"))
  t=t+1
  start_date += delta



# # Italy ---------------------------------

#covid_italy_confirmed=pd.read_csv('/home2/e0268-26/a4/data/Italy.csv',index_col=0,sep='\t',usecols=[0,1])
#covid_italy_recovered=pd.read_csv('/home2/e0268-26/a4/data/Italy.csv',index_col=0,sep='\t',usecols=[0,2])
#covid_italy_death=pd.read_csv('/home2/e0268-26/a4/data/Italy.csv',index_col=0,sep='\t',usecols=[0,3])

#X2=covid_italy_confirmed.values
#Y2=covid_italy_recovered.values
#Z2=covid_italy_death.values

#model_arima_confirmed2=ARIMA(X2,order=(6,2,2))
#model_arima_confirmed_fit2=model_arima_confirmed2.fit()
#with open('/home2/e0268-26/a4/model_confirmed2.pkl','wb') as f:
#  pickle.dump(model_arima_confirmed_fit2,f)
#print(model_arima_confirmed_fit2.aic)
model_con_italy=pickle.load(open('/home2/e0268-26/a4/model_confirmed2.pkl','rb'))
predictions12=model_con_italy.forecast(steps=step+1)[0]

#model_arima_recovered2=ARIMA(Y2,order=(4,2,2))
#model_arima_recovered_fit2=model_arima_recovered2.fit()
#with open('/home2/e0268-26/a4/model_recovered2.pkl','wb') as f:
#  pickle.dump(model_arima_recovered_fit2,f)
#print(model_arima_recovered_fit2.aic)
model_rec_italy=pickle.load(open('/home2/e0268-26/a4/model_recovered2.pkl','rb'))
predictions22=model_rec_italy.forecast(steps=step+1)[0]

#model_arima_death2=ARIMA(Z2,order=(9,2,3))
#model_arima_death_fit2=model_arima_death2.fit()
#with open('/home2/e0268-26/a4/model_death2.pkl','wb') as f:
#  pickle.dump(model_arima_death_fit2,f)
#print(model_arima_death_fit2.aic)
model_death_italy=pickle.load(open('/home2/e0268-26/a4/model_death2.pkl','rb'))
predictions32=model_death_italy.forecast(steps=step+1)[0]

# start_date=start
# end_date=ending

# while start_date <= end_date:
#     print (start_date.strftime("%Y-%m-%d\t"),end="")
#     print("\t",end="")
#     print(predictions12[t],end="")
#     print("\t",end="")
#     print(predictions22[t],end="")
#     print("\t",end="")
#     print(predictions32[t],end="")
#     print('\n')
#     t=t+1
#     start_date += delta

start_date=start
end_date=ending
#t=delta.days
t=m
while start_date <= end_date:
  output_Italy.write(str(start_date.strftime("%Y-%m-%d"))+'\t')
  output_Italy.write(str(predictions12[t])+'\t')
  # output_India.write(str("\t"))
  output_Italy.write(str(predictions22[t])+'\t')
  # output_India.write(str("\t"))
  output_Italy.write(str(predictions32[t])+'\n')
  # output_India.write(str("\n"))
  t=t+1
  start_date += delta

# #Belgium---------------------------------------------------------


#covid_belgium_confirmed=pd.read_csv('/home2/e0268-26/a4/data/Belgium.csv',index_col=0,sep='\t',usecols=[0,1])
#covid_belgium_recovered=pd.read_csv('/home2/e0268-26/a4/data/Belgium.csv',index_col=0,sep='\t',usecols=[0,2])
#covid_belgium_death=pd.read_csv('/home2/e0268-26/a4/data/Belgium.csv',index_col=0,sep='\t',usecols=[0,3])

#X3=covid_belgium_confirmed.values
#Y3=covid_belgium_recovered.values
#Z3=covid_belgium_death.values

#model_arima_confirmed3=ARIMA(X3,order=(0,2,1))
#model_arima_confirmed_fit3=model_arima_confirmed3.fit()
#with open('/home2/e0268-26/a4/model_confirmed3.pkl','wb') as f:
#  pickle.dump(model_arima_confirmed_fit3,f)
#print(model_arima_confirmed_fit3.aic)
model_con_belgium=pickle.load(open('/home2/e0268-26/a4/model_confirmed3.pkl','rb'))
predictions13=model_con_belgium.forecast(steps=step+1)[0]

#model_arima_recovered3=ARIMA(Y3,order=(9,2,2))
#model_arima_recovered_fit3=model_arima_recovered3.fit()
#with open('/home2/e0268-26/a4/model_recovered3.pkl','wb') as f:
#  pickle.dump(model_arima_recovered_fit3,f)
#print(model_arima_recovered_fit3.aic)
model_rec_belgium=pickle.load(open('/home2/e0268-26/a4/model_recovered3.pkl','rb'))
predictions23=model_rec_belgium.forecast(steps=step+1)[0]

#model_arima_death3=ARIMA(Z3,order=(0,2,7))
#model_arima_death_fit3=model_arima_death3.fit()
#with open('/home2/e0268-26/a4/model_death3.pkl','wb') as f:
#  pickle.dump(model_arima_death_fit3,f)
#print(model_arima_death_fit3.aic)
model_death_belgium=pickle.load(open('/home2/e0268-26/a4/model_death3.pkl','rb'))
predictions33=model_death_belgium.forecast(steps=step+1)[0]

# start_date=start
# end_date=ending

# while start_date <= end_date:
#     print (start_date.strftime("%Y-%m-%d\t"),end="")
#     print("\t",end="")
#     print(predictions13[t],end="")
#     print("\t",end="")
#     print(predictions23[t],end="")
#     print("\t",end="")
#     print(predictions33[t],end="")
#     print('\n')
#     t=t+1
#     start_date += delta

start_date=start
end_date=ending
#t=delta.days
t=m
while start_date <= end_date:
  output_Belgium.write(str(start_date.strftime("%Y-%m-%d"))+'\t')
  output_Belgium.write(str(predictions13[t])+'\t')
  # output_India.write(str("\t"))
  output_Belgium.write(str(predictions23[t])+'\t')
  # output_India.write(str("\t"))
  output_Belgium.write(str(predictions33[t])+'\n')
  # output_India.write(str("\n"))
  t=t+1
  start_date += delta

# US--------------------------------------------------------

#covid_us_confirmed=pd.read_csv('/home2/e0268-26/a4/data/US.csv',index_col=0,sep='\t',usecols=[0,1])
#covid_us_recovered=pd.read_csv('/home2/e0268-26/a4/data/US.csv',index_col=0,sep='\t',usecols=[0,2])
#covid_us_death=pd.read_csv('/home2/e0268-26/a4/data/US.csv',index_col=0,sep='\t',usecols=[0,3])

#X4=covid_us_confirmed.values
#Y4=covid_us_recovered.values
#Z4=covid_us_death.values

#model_arima_confirmed4=ARIMA(X4,order=(2,2,5))
#model_arima_confirmed_fit4=model_arima_confirmed4.fit()
#with open('/home2/e0268-26/a4/model_confirmed4.pkl','wb') as f:
#  pickle.dump(model_arima_confirmed_fit4,f)
#print(model_arima_confirmed_fit4.aic)
model_con_us=pickle.load(open('/home2/e0268-26/a4/model_confirmed4.pkl','rb'))
predictions14=model_con_us.forecast(steps=step+1)[0]

#model_arima_recovered4=ARIMA(Y4,order=(3,2,9))
#model_arima_recovered_fit4=model_arima_recovered4.fit()
#with open('/home2/e0268-26/a4/model_recovered4.pkl','wb') as f:
#  pickle.dump(model_arima_recovered_fit4,f)
#print(model_arima_recovered_fit4.aic)
model_rec_us=pickle.load(open('/home2/e0268-26/a4/model_recovered4.pkl','rb'))
predictions24=model_rec_us.forecast(steps=step+1)[0]

#model_arima_death4=ARIMA(Z4,order=(6,2,2))
#model_arima_death_fit4=model_arima_death4.fit()
#with open('/home2/e0268-26/a4/model_death4.pkl','wb') as f:
#  pickle.dump(model_arima_death_fit4,f)
#print(model_arima_death_fit4.aic)
model_death_us=pickle.load(open('/home2/e0268-26/a4/model_death4.pkl','rb'))
predictions34=model_death_us.forecast(steps=step+1)[0]

# start_date=start
# end_date=ending

# while start_date <= end_date:
#     print (start_date.strftime("%Y-%m-%d\t"),end="")
#     print("\t",end="")
#     print(predictions14[t],end="")
#     print("\t",end="")
#     print(predictions24[t],end="")
#     print("\t",end="")
#     print(predictions34[t],end="")
#     print('\n')
#     t=t+1
#     start_date += delta

start_date=start
end_date=ending
#t=delta.days
t=m
while start_date <= end_date:
  output_US.write(str(start_date.strftime("%Y-%m-%d"))+'\t')
  output_US.write(str(predictions14[t])+'\t')
  # output_India.write(str("\t"))
  output_US.write(str(predictions24[t])+'\t')
  # output_India.write(str("\t"))
  output_US.write(str(predictions34[t])+'\n')
  # output_India.write(str("\n"))
  t=t+1
  start_date += delta

# p=d=q=range(0,10)
# pdq=list(itertools.product(p,d,q))
# print(pdq)
# mini=10000
# par=[0,0,0]
# for param in pdq:
#   try:
#       model_arima=ARIMA(X,order=(param))
#       model_arima_fit=model_arima.fit()
#       # print(param,model_arima_fit.aic)
#       if(model_arima_fit.aic <mini):
#         mini=model_arima_fit.aic
#         par=param
#   except:
#       continue
# print(mini,par)


# print(type(predictions))
# print(predictions.shape)

# print((predictions))
# plt.plot(predictions)
