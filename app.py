import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("set2.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('dataset.csv')
X = dataset.iloc[:,1:10].values

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Female', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 1:2])
#Replacing missing data with the calculated mean value  
X[:, 1:2]= imputer.transform(X[:, 1:2])  

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Female', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 2:3])
#Replacing missing data with the calculated mean value  
X[:, 2:3]= imputer.transform(X[:, 2:3])

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 3:6])
#Replacing missing data with the calculated mean value  
X[:, 3:6]= imputer.transform(X[:, 3:6])

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'most_frequent', fill_value='none', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 6:8])
#Replacing missing data with the calculated mean value  
X[:, 6:8]= imputer.transform(X[:, 6:8])

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 8:9])
#Replacing missing data with the calculated mean value  
X[:, 8:9]= imputer.transform(X[:, 8:9])  

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:,1])

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:,2])



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict_note_authentication(CreditScore, Geography, Gender, Age, Tenure, Balance, HasCrCard, IsActiveMember, EstimatedSalary):
  output= model.predict(sc.transform([[CreditScore, Geography, Gender, Age, Tenure, Balance, HasCrCard, IsActiveMember, EstimatedSalary]]))
  print("Model has predicted",output)
  if output==[0]:
    prediction="Customer will leave..."
   

  if output==[1]:
    prediction="Customert will continue..."
    
    
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:black;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Experiment Deployment By Shruti Jain</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer continue or not...")
    Name = st.text_input('Enter name')
    CreditScore = st.number_input('Insert credit score',100,1000)
    Geography= st.number_input('Insert 0 for France 1 for Spain',0,1)
    Gender = st.number_input('Insert 0 for Male 1 for Female ',0,1)
    age = st.number_input('Insert a Age',18,70)
    Tenure = st.number_input('Insert Tenure',0,40)
    Balance = st.number_input('Insert Balance',0,130000)
    HasCrCard = st.number_input('Insert 0 for no 1 for yes for Credit Card',0,1)
    IsActiveMember = st.number_input('Insert 0 for no 1 for yes for active member',0,1)
    EstimatedSalary = st.number_input('InsertEstimated salery ',0,150000)
    
    result=""
    if st.button("Predict"):
      result=predict_note_authentication(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Shruti Jain")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
   
