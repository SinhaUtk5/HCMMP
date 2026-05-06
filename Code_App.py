# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image

# --------------------------
# App config
# --------------------------
st.set_page_config(page_title="HC MMP Calculator", layout="wide")

APP_DIR = Path(__file__).resolve().parent

# --- Top logos ---------------------------------------------------------------------
left_space, logo1, logo2, right_space = st.columns([1, 1, 1, 1])

with logo1:
    if os.path.exists("TAMU.png"):
        st.image("TAMU.png", width=200)

with logo2:
    if os.path.exists("IPBF.png"):
        st.image("IPBF.png", width=200)

# --------------------------
# Helpers
# --------------------------
def show_resized_image(img_name: str, target_height: int = 200) -> None:
    img_path = APP_DIR / img_name
    if not img_path.exists():
        st.warning(f"Missing image file: {img_name} (place it next to this app if you want it shown).")
        return
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    new_h = int(target_height)
    new_w = int(w * (new_h / h))
    st.image(img.resize((new_w, new_h)))


@st.cache_resource
def load_model() -> object:
    model_path = APP_DIR / "finalized_MMP_HC_model2.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Place 'finalized_MMP_HC_model2.pkl' next to Code_App.py"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def require_columns(df: pd.DataFrame, min_cols: int) -> None:
    if df.shape[1] < min_cols:
        raise ValueError(
            f"Input file has {df.shape[1]} columns, but this app expects at least {min_cols} columns "
            f"(based on the original positional indexing IRIS[:,0..16])."
        )


def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # Force everything to numeric where possible; non-numeric becomes NaN
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def predict_mmp_from_dataframe(df_raw: pd.DataFrame, model: object) -> pd.DataFrame:
    """
    This preserves your original column-by-position logic, but makes it robust:
    - numeric conversion
    - optional SG column handling
    - avoids crashing if some columns are missing by giving clear errors
    """

    df_num = to_numeric_df(df_raw)
    require_columns(df_num, 17)  # because you reference IRIS[:,16]

    IRIS = df_num.to_numpy(dtype=np.float64)

    # --- Original feature extraction (positional) ---
    H2S = np.power(IRIS[:, 0], 1)
    Co2 = IRIS[:, 1]
    N2 = IRIS[:, 2]
    C1 = IRIS[:, 3]
    C2_C6 = IRIS[:, 4]
    C7_plus = np.power(IRIS[:, 5], 1)
    MW_Oil = IRIS[:, 6]

    N2_gas = IRIS[:, 7]
    H2S_gas = IRIS[:, 8]
    CO2_gas = IRIS[:, 9]
    CH4_gas = IRIS[:, 10]
    C2Plus_gas = IRIS[:, 11]
    MWC2Plus_gas = IRIS[:, 12]

    GAS_MW = (N2_gas * 28.0134 + H2S_gas * 34.1 + CH4_gas * 16.04 + CO2_gas * 44.01 + C2Plus_gas * MWC2Plus_gas) / 100.0
    mf_CO2 = CO2_gas / 100.0
    mf_H2S = H2S_gas / 100.0
    GAS_Grav = GAS_MW / 28.97

    uncTC_Rank = 169.2 + 349.5 * GAS_Grav - 74.0 * GAS_Grav * GAS_Grav
    Correction = 120.0 * ((mf_CO2 + mf_H2S) ** 0.9 - (mf_CO2 + mf_H2S) ** 1.6) + 15.0 * ((mf_H2S ** 0.5) - (mf_H2S ** 4))
    TC_GAS_F = uncTC_Rank - Correction - 459.67

    unc_PC = 756.8 - 131.07 * GAS_Grav - 3.6 * GAS_Grav * GAS_Grav
    PC_PSIA = unc_PC * (TC_GAS_F + 459.67) / (uncTC_Rank - mf_H2S * (1 - mf_H2S) * Correction)

    T_Res_F = IRIS[:, 13]
    MWC7plus_oil = IRIS[:, 14]

    # IRIS[:,15] is SG (optional per-row), may be NaN
    # IRIS[:,16] is MMP (may exist / may be NaN); we do NOT need it for prediction

    APPWeight_C7plus_Oil = MWC7plus_oil * C7_plus
    Prox1 = (C2_C6 + H2S + Co2) / (MWC7plus_oil) / np.power(((T_Res_F - 32) / 1.8 + 273), 0.203)
    Prox2 = (C2Plus_gas) * (MWC2Plus_gas) / 100.0

    SG_calc0 = 1.106352054 / (46.23006224 / MWC7plus_oil + 1.090283159)
    SG_calc1 = 0.134462445 + 0.214592184 * SG_calc0 + 0.703011117 * SG_calc0 * SG_calc0 + 0.010846788 * np.exp(SG_calc0)

    SG_calc2 = np.array(IRIS[:, 15], dtype=np.float64)  # user-provided SG (may be NaN)
    SG_calc = SG_calc2.copy()
    nan_mask = np.isnan(SG_calc2)
    SG_calc[nan_mask] = SG_calc1[nan_mask]

    API_calc = 141.5 / SG_calc - 131.5
    KW_calc = 4.5579 * (MWC7plus_oil ** 0.15178) * (SG_calc ** (-0.84573))

    X3 = np.c_[
        H2S, N2, Co2, C1, C2_C6, C7_plus, MW_Oil, APPWeight_C7plus_Oil, MWC7plus_oil,
        N2_gas, H2S_gas, CO2_gas, CH4_gas, C2Plus_gas,
        TC_GAS_F, PC_PSIA, T_Res_F, Prox1, Prox2, GAS_MW,
        SG_calc, KW_calc, API_calc
    ]

    # --- Standardize exactly as you do ---
    mean = np.array([
        7.49782383e-01, 1.88150259e-01, 2.50931088e+00, 2.94963057e+01,
        2.77320674e+01, 3.93243834e+01, 1.14484469e+02, 9.41773229e+03,
        2.30557071e+02, 8.59015544e-01, 2.18165803e+00, 4.28359585e+00,
        6.80401347e+01, 2.46355959e+01, -6.43291640e+01, 6.35375264e+02,
        2.23743003e+02, 4.13742243e-02, 9.84381443e+00, 2.36272474e+01,
        8.58336185e-01, 1.18343859e+01, 3.35047587e+01
    ], dtype=np.float64)

    var = np.array([
        4.43510728e+00, 8.75111432e-02, 4.51939588e+00, 2.03828200e+02,
        6.52173929e+01, 2.63273806e+02, 2.27667960e+03, 2.50357588e+07,
        9.11313633e+02, 2.32271717e+00, 3.64417506e+01, 2.05944166e+01,
        2.70772004e+02, 2.05713883e+02, 8.41120763e+02, 6.13225075e+02,
        1.19483411e+03, 2.15996195e-04, 3.52361854e+01, 1.66154062e+01,
        6.84661128e-04, 1.38515258e-02, 2.45358475e+01
    ], dtype=np.float64)

    X = (X3 - mean) / np.sqrt(var)

    # --- Predict ---
    y_pred = model.predict(X)
    y_pred = np.asarray(y_pred).reshape(-1)

    out = df_raw.copy()
    out["MMP_Pred(Psia)"] = y_pred
    return out


# --------------------------
# UI Header
# --------------------------
st.title("Hydrocarbon Gas (HC) MMP Calculator")
st.markdown("A product of Interaction of Phase-Behavior and Flow (IPB&F) Consortium")
st.markdown("**Developed by** - Utkarsh Sinha and Dr. Birol Dindoruk")

st.markdown(
    "[Ref. - Sinha, U., Dindoruk, B., & Soliman, M. (2021).  Physics guided data-driven model to estimate minimum "
    "miscibility pressure (MMP) for hydrocarbon gases. Geoenergy Science and Engineering, 211389.](https://doi.org/10.1016/j.geoen.2022.211389)"
)
st.markdown("**Product Description:** Calculates the Minimum Miscibility Pressure (psia) for CH4 dominant hydrocarbon gas injection.")
st.markdown(
    "1) **Download the example input CSV template file**: "
    "[Click here](https://drive.google.com/file/d/1HNyZjobmTEBcWfk0C2cmClQfahTONrX1/view?usp=sharing)"
)

# --------------------------
# Contributor images / blocks
# --------------------------
c1, c2 = st.columns([1, 1])
with c1:
    show_resized_image("utkarsh.jpg", 180)
    st.markdown("**Utkarsh Sinha**  \nVolunteer Research Fellow  \nInteraction of Phase-Behavior and Flow (IPB&F) Consortium")
with c2:
    show_resized_image("birol.jpg", 180)
    st.markdown("**Dr. Birol Dindoruk**  \nProfessor  \nHarold Vance Department of Petroleum Engineering,  \nTexas A&M University")

st.divider()

# --------------------------
# Model load
# --------------------------
try:
    my_model = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()


# --------------------------
# File upload + prediction
# --------------------------
uploaded_file = st.file_uploader("Upload the input CSV file here", type=["csv"])

# Blue info banner exactly like your screenshot
if uploaded_file is None:
    st.info("Upload a CSV to enable prediction.")
    predict_disabled = True
else:
    predict_disabled = False

# Keep the predict button always visible, but disabled until file is uploaded
run_pred = st.button("Run prediction", disabled=predict_disabled)

if uploaded_file is not None:
    try:
        df_in = pd.read_csv(uploaded_file)
        st.write("Preview:", df_in.head())

        if run_pred:
            df_out = predict_mmp_from_dataframe(df_in, my_model)
            st.success("Prediction completed.")
            st.write(df_out)

            csv_bytes = convert_df(df_out)
            st.download_button(
                label="Download results as CSV",
                data=csv_bytes,
                file_name="HC_MMP_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error("Prediction failed. Check your CSV formatting and required columns.")
        st.exception(e)










