import pandas as pd
import numpy as np
from numpy import genfromtxt
import pickle
import sys
import xgboost as xgb
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split

#print("hello")
#data = pd.read_csv('/home2/e0268-26/a3/years.train',header=None)
#D = pd.DataFrame(data)
# # D.columns = ['year','attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute' ,  'attribute']
# # predict = D.iloc[:,0]
# pre = "year"
#X = np.array(data.drop([0],1))
#y = np.array(data[0])
# print(X[0:4])
# print(y[0:10])

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.04)

#xg_reg = xgb.XGBRegressor(learning_rate = 0.1, max_depth = 30 , alpha = 10, n_estimators = 200)

# print(type(x_test))

inputfile = open(sys.argv[1] , "r")
output = open(sys.argv[2], "w")
# I = int(input.readline())

#xg_reg.fit(x_train , y_train)


#with open('/home2/e0268-26/a3/model_year.pkl','wb') as f:
#     pickle.dump(xg_reg , f)
model = pickle.load(open("/home2/e0268-26/a3/model_year.pkl",'rb'))
# year_ = model.predict(input)

# preds = model.predict(x_test)
# print(type(preds))
# print(preds[0:5])
#with open('model_year.pkl','rb') as f:
#     preds = pickle.load(f)


I=int(inputfile.readline())
#print(I)
my_data = genfromtxt(inputfile, delimiter=',')

second_preds=model.predict(my_data)
# print(second_preds)
# print(second_preds.shape)
# print(np.around(second_preds))

for i in range(I):
    output.write(str(np.around(second_preds[i])))
    output.write("\n")

#np.savetxt(output,second_preds,fmt='%f')
#pd.DataFrame(second_preds).to_csv(output)
inputfile.close()
output.close()
#f.close()

# np.savetxt(output,second_pred,fmt='%f')

# rmse = np.sqrt(mean_squared_error(y_test, preds))
# print("RMSE: %f" % rmse)




