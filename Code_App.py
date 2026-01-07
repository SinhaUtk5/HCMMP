# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import streamlit as st
import pickle

import numpy as np
from io import BytesIO

import requests

# df = pd.read_csv("C:/Utilities/MMP Papers/HC inj code/MMP_HC_Data ALLwith SG_Clean.csv")  
# read a CSV file inside the 'data" folder next to 'app.py'
# df = pd.read_excel(...)  # will work for Excel files
html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h1 style="color:white;text-align:center;">MMP Calculator for Hydrocarbon Gas Injection) </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

st.header("Interaction of Phase-Behavior and Flow  ([IPB&F](https://biroldindoruk.engr.tamu.edu/)) Consortium, Texas A&M University.")

st.markdown('<style>h2{color: red;}</style>', unsafe_allow_html=True)
st.subheader("Product Description - Calculates the Minimum Miscibility Pressure (psia) for CH4 dominant hydrocarbon gas injection.")
st.markdown('<style>h2{color: red;}</style>', unsafe_allow_html=True)
st.subheader("[Download Input Template File.](https://drive.google.com/file/d/1HNyZjobmTEBcWfk0C2cmClQfahTONrX1/view?usp=sharing)")
# st.markdown("[Input Template File Link](https://drive.google.com/file/d/1HNyZjobmTEBcWfk0C2cmClQfahTONrX1/view?usp=sharing)",unsafe_allow_html=True)

pickle_in = open('finalized_MMP_HC_model2.pkl', 'rb')
my_model= pickle.load(pickle_in)
# print(my_model)


# df=pd.read_csv("App_test_MMP_HC.csv")
# IRIS0=IRIS
# row,col=np.shape(IRIS)
# row
def predict_MMP(Inp):
    result=my_model.predict(Inp)
    return result

def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')


    
    
# A=IRIS[:,0]
uploaded_file = st.file_uploader("Upload input csv file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    IRIS=np.array(df)
    H2S =(pow((IRIS[:,0 ]),1))
    print(H2S)
    Co2 = (IRIS[:,1 ])
    N2 = IRIS[:,2]
    
    C1 = IRIS[:,3 ]
    
    C2_C6 = IRIS[:,4 ]
    
    C7_plus = pow(IRIS[:,5 ],1)
    
    
    MW_Oil = IRIS[:,6 ]
    
    
    
    N2_gas = IRIS[:,7 ]
    
    
    H2S_gas = IRIS[:,8 ]
    
    CO2_gas = IRIS[:,9 ]
    
    CH4_gas = IRIS[:,10 ]
    
    C2Plus_gas = IRIS[:,11 ]
    
    MWC2Plus_gas = IRIS[:,12 ]
    
    GAS_MW = (N2_gas*28.0134+H2S_gas*34.1+CH4_gas*16.04+CO2_gas*44.01+C2Plus_gas*MWC2Plus_gas)/100
    
    
    mf_CO2=CO2_gas/100
    mf_H2S=H2S_gas/100
    
    GAS_Grav=GAS_MW/28.97
    
    uncTC_Rank=169.2+349.5*GAS_Grav-74*GAS_Grav*GAS_Grav
    Correction=120*(( mf_CO2+ mf_H2S)**0.9-(mf_CO2+ mf_H2S)**1.6)+15*((mf_H2S**0.5)-(mf_H2S**4))
    TC_GAS_F = uncTC_Rank-Correction-459.67
    
    unc_PC=756.8-131.07*GAS_Grav-3.6*GAS_Grav*GAS_Grav
    
    PC_PSIA = unc_PC*(TC_GAS_F+459.67)/(uncTC_Rank-mf_H2S*(1-mf_H2S)*Correction)
    
    
    
    T_Res_F= IRIS[:,13] 
    
    MWC7plus_oil = ((IRIS[:,14 ]))
    MMP = ((IRIS[:,16 ]))
    
    
    APPWeight_C7plus_Oil= MWC7plus_oil*C7_plus 
    Prox1=(C2_C6+H2S+Co2)/(MWC7plus_oil)/pow((( T_Res_F-32)/1.8+273),0.203)
    Prox2=(C2Plus_gas)*(MWC2Plus_gas)/100
    
    correction_factor=1
    
    SG_calc0=SG_calc0=1.106352054/(46.23006224/MWC7plus_oil+1.090283159)
    
    
    SG_calc1=0.134462445+0.214592184*SG_calc0+0.703011117*SG_calc0*SG_calc0+0.010846788*np.exp(SG_calc0)
    # SG_calc1=SG_calc0
    
    
    
    
    
    SG_calc2=np.array(((IRIS[:,15 ])), dtype=np.float64)
    SG_calc=np.array(SG_calc2)
    
    
    
    for index in range(0,len(SG_calc2)):
        
        if(np.isnan(SG_calc2[index])):
            SG_calc[index]=SG_calc1[index]

     
            
            
    print(SG_calc2)        
    #prox4 is KW
    API_calc=141.5/SG_calc-131.5
    KW_calc=4.5579*(MWC7plus_oil**0.15178)*(SG_calc**(-0.84573))
    
    X3=np.c_[H2S,N2, Co2, C1, C2_C6,C7_plus,MW_Oil,APPWeight_C7plus_Oil,MWC7plus_oil,\
              N2_gas, H2S_gas,CO2_gas,CH4_gas,C2Plus_gas,\
              TC_GAS_F,PC_PSIA,T_Res_F,Prox1,Prox2,GAS_MW,\
                SG_calc,KW_calc,API_calc]
    
    X2=pd.DataFrame(X3)
    
    X=np.array(X2)
    mean=np.array([ 7.49782383e-01,  1.88150259e-01,  2.50931088e+00,  2.94963057e+01,
        2.77320674e+01,  3.93243834e+01,  1.14484469e+02,  9.41773229e+03,
        2.30557071e+02,  8.59015544e-01,  2.18165803e+00,  4.28359585e+00,
        6.80401347e+01,  2.46355959e+01, -6.43291640e+01,  6.35375264e+02,
        2.23743003e+02,  4.13742243e-02,  9.84381443e+00,  2.36272474e+01,
        8.58336185e-01,  1.18343859e+01,  3.35047587e+01])
    var=np.array([4.43510728e+00, 8.75111432e-02, 4.51939588e+00, 2.03828200e+02,
       6.52173929e+01, 2.63273806e+02, 2.27667960e+03, 2.50357588e+07,
       9.11313633e+02, 2.32271717e+00, 3.64417506e+01, 2.05944166e+01,
       2.70772004e+02, 2.05713883e+02, 8.41120763e+02, 6.13225075e+02,
       1.19483411e+03, 2.15996195e-04, 3.52361854e+01, 1.66154062e+01,
       6.84661128e-04, 1.38515258e-02, 2.45358475e+01])
    X=(X-mean)/(var**0.5)
    
    
    if st.button("Predict MMP (Psia)"):
        output=np.array(predict_MMP(X))
        df.insert(17, "MMP_Pred(Psia)", output, True)
        st.write(df)
        csv = convert_df(df)
        st.download_button(
             label="Download data as CSV",
             data=csv,
             file_name='large_df.csv',
             mime='text/csv',
         )
st.subheader("Developed by by [Utkarsh Sinha](https://www.linkedin.com/in/utkarsh-sinha-ba398b75/) and [Dr. Birol Dindoruk](https://www.petro.uh.edu/faculty/dindoruk) [ based on the work in Ref:- Sinha U., Dindoruk B., & Soliman M. (2023). Physics guided data-driven model to estimate minimum miscibility pressure (MMP) for hydrocarbon gases. Geoenergy Science and Engineering, 211389.](https://www.sciencedirect.com/science/article/abs/pii/S294989102200077X)")
#st.subheader('[Ref.: Sinha U., Dindoruk B., & Soliman M. (2023). Physics guided data-driven model to estimate minimum miscibility pressure (MMP) for hydrocarbon gases. Geoenergy Science and Engineering, 211389.](https://www.sciencedirect.com/science/article/abs/pii/S294989102200077X)')

from PIL import Image
# Authors section
c1, c2 = st.columns(2)
with c1:
    img1 = load_square_image("birol.jpg") or load_square_image(
        "https://upload.wikimedia.org/wikipedia/commons/8/88/Placeholder_avatar.png"
    )
    if img1:
        st.image(img1, caption="Dr. Birol Dindoruk", width=150)
    st.markdown(
        """
        **Dr. Birol Dindoruk**<br>
        Professor<br>
        Harold Vance Department of Petroleum Engineering,<br>
        Texas A&M University
        """,
        unsafe_allow_html=True,
    )

with c2:
    img2 = load_square_image("utkarsh.jpg") or load_square_image(
        "https://upload.wikimedia.org/wikipedia/commons/8/88/Placeholder_avatar.png"
    )
    if img2:
        st.image(img2, caption="Utkarsh Sinha", width=150)
    st.markdown(
        """
        **Utkarsh Sinha**<br>
        Volunteer Research Associate <br>
        Interaction of Phase-Behavior and Flow (IPB&F) Consortium,<br>
        """,
        unsafe_allow_html=True,
    )


        
    #print('result===',result)


