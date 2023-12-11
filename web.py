import streamlit as st
import pandas as pd
import numpy as np
import pickle


with open('xgb_plan1.dat', 'rb') as f:
    model = pickle.load(f)

st.title('Wine Quality Classifier Web App')
st.write('This is a web app to classify the quality of your wine based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the classifier.')


age_input = st.selectbox('Age',('<55', '[55,65)', '[65,75)','≥75'))
gender_input = st.selectbox('Gender',('man', 'woman'))
historyOfDiabetesMellitus_input = st.selectbox('History Of Diabetes Mellitus',('no', 'yes'))
historyOfMyocardialInfarction_input = st.selectbox('History Of Myocardial Infarction',('no', 'yes'))
historyOfIschemicStroke_input = st.selectbox('History Of Ischemic Stroke',('no', 'yes'))
historyOfHeartFailure_input = st.selectbox('History Of Heart Failure',('no', 'yes'))
historyOfCOPD_input = st.selectbox('History Of COPD',('no', 'yes'))
historyOfRenalInsufficiency_input = st.selectbox('History Of Renal Insufficiency',('no', 'yes'))
heartRate_input = st.selectbox('Heart rate',('<60','[60,100)', '≥100'))
shockIndex_input = st.selectbox('Shock index',('<0.4', '[0.4,0.8)','≥0.8'))
acuteFailureOnAdmission_input = st.selectbox('Acute Failure On Admission',('no', 'yes'))
cardiacArrestOnAdmission_input = st.selectbox('Cardiac Arrest On Admission',('no', 'yes'))
stSegmentChange_input = st.selectbox('ST Segment Change',('no', 'yes'))


SEX = np.where(gender_input=='man',0,1)
MHDM = np.where(historyOfDiabetesMellitus_input=='no',0,1)
MHMI = np.where(historyOfMyocardialInfarction_input=='no',0,1)
MHHF = np.where(historyOfHeartFailure_input=='no',0,1)
zuzhong = np.where(historyOfIschemicStroke_input=='no',0,1)
MHCOPD = np.where(historyOfCOPD_input=='no',0,1)	
MHKF = np.where(historyOfRenalInsufficiency_input=='no',0,1)
MAHF = np.where(acuteFailureOnAdmission_input=='no',0,1)
MACA = np.where(cardiacArrestOnAdmission_input=='no',0,1)
stc = np.where(stSegmentChange_input=='no',0,1)
age_1 = np.where(age_input=='<55',1,0)
age_2 = np.where(age_input=='[55,65)',1,0)
age_3 = np.where(age_input=='[65,75)',1,0)
age_4 = np.where(age_input=='≥75',1,0)
si_1 = np.where(shockIndex_input=='<0.4',1,0)
si_2 = np.where(shockIndex_input=='[0.4,0.8)',1,0)
si_3 = np.where(shockIndex_input=='≥0.8',1,0)
bmp_1 = np.where(heartRate_input=='<60',1,0)
bmp_2 = np.where(heartRate_input=='[60,100)',1,0)
bmp_3 = np.where(heartRate_input=='≥100',1,0)


features = np.array([SEX,MHDM,MHMI,MHHF,zuzhong,MHCOPD,MHKF,MAHF,MACA,stc,age_1,age_2,age_3,age_4,si_1,si_2,si_3,bmp_1,bmp_2,bmp_3]).reshape(1,-1)


st.table(pd.DataFrame([{
'Age':age_input,
'Gender':gender_input,
'History Of Diabetes Mellitus':historyOfDiabetesMellitus_input,
'History Of Myocardial Infarction':historyOfMyocardialInfarction_input,
'History Of Ischemic Stroke':historyOfIschemicStroke_input,
'History Of Heart Failure':historyOfHeartFailure_input,
'History Of COPD':historyOfCOPD_input,
'History Of Renal Insufficiency':historyOfRenalInsufficiency_input,
'Heart rate':heartRate_input,
'Shock index':shockIndex_input,
'Acute Failure On Admission':acuteFailureOnAdmission_input,
'Cardiac Arrest On Admission':cardiacArrestOnAdmission_input,
'ST Segment Change':stSegmentChange_input
}]))  


if st.button('Predict'):
    prediction = model.predict_proba(features)[:,1]
    st.write(' Based on feature values, your risk score is : '+ str(int(prediction * 100)))