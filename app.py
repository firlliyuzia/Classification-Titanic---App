import streamlit as st
import pandas as pd
import pickle

# Page Setup
st.set_page_config(
    page_title="Titanic Survival Classification",
    page_icon="🚢",
    layout="centered"
)

st.title("Klasifikasi Keselamatan Penumpang Titanic")
st.write(
    "Aplikasi ini memprediksi kemungkinan keselamatan penumpang Titanic "
    "menggunakan model **AdaBoost** dengan akurasi sekitar **83%**."
)
st.divider()

# Load Trained Model
MODEL_PATH = "best_model_adaboost.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model tidak ditemukan. Pastikan file model berada di folder yang sama.")
    st.stop()

# Input Form
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Kelas Tiket (Pclass, 1 = Eksekutif, 2, = Bisnis, 3 = Ekonomi) ", ["pilih kelas tiket",1, 2, 3])
    name = st.text_input("Nama Penumpang (Wajib sertakan gelar, cth: Mr. John)")
    sex = st.selectbox("Jenis Kelamin", ["pilih jenis kelamin","male", "female"])
    age = st.number_input("Umur (Tahun)", min_value=0, max_value=100, step=1, format="%d")
    
with col2:
    sibsp = st.number_input("Jumlah Saudara/Pasangan (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Jumlah Orang Tua/Anak (Parch)", min_value=0, max_value=10, value=0, step=1)
    fare = st.number_input("Harga Tiket (Fare)", min_value=0.0, max_value=600.0, step=0.1)
    embarked = st.selectbox("Pelabuhan Keberangkatan (S=Southampton, C=Cherbourg, Q=Queenstown)", ["pilih Embarked","S", "C", "Q"])

st.divider()
# Preprocessing Function
def preprocess_input(name, pclass, sex, age, sibsp, parch, fare, embarked):
    # Ambil gelar dari nama
    if "," in name and "." in name:
        title = name.split(",")[1].split(".")[0].strip()
    else:
        title = "Mr"

    if title == "Mr":
        initial = 0
    elif title == "Mrs":
        initial = 1
    elif title in ["Miss", "Mlle", "Mme", "Ms"]:
        initial = 2
    elif title == "Master":
        initial = 3
    else:
        initial = 4

    # Age band
    if age <= 16:
        age_band = 0
    elif age <= 32:
        age_band = 1
    elif age <= 48:
        age_band = 2
    elif age <= 64:
        age_band = 3
    else:
        age_band = 4

    family_size = sibsp + parch
    alone = 1 if family_size == 0 else 0

    # Fare category
    if fare <= 7.91:
        fare_cat = 0
    elif fare <= 14.454:
        fare_cat = 1
    elif fare <= 31:
        fare_cat = 2
    else:
        fare_cat = 3

    sex_num = 0 if sex == "male" else 1
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    embarked_num = embarked_map.get(embarked, 0)

    data = pd.DataFrame(
        [[
            pclass,
            sex_num,
            sibsp,
            parch,
            embarked_num,
            initial,
            age_band,
            family_size,
            alone,
            fare_cat
        ]],
        columns=[
            "Pclass", "Sex", "SibSp", "Parch", "Embarked",
            "Initial", "Age_band", "Family_Size", "Alone", "Fare_cat"
        ]
    )

    return data
# Prediction Button
_, mid_col, _ = st.columns([1, 2, 1])

with mid_col:
    predict_btn = st.button(
        "PREDIKSI SEKARANG",
        type="primary",
        use_container_width=True
    )
# Prediction Result

if predict_btn:
    with st.spinner("Sedang memproses data..."):
        X_input = preprocess_input(
            name, pclass, sex, age, sibsp, parch, fare, embarked
        )

        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]

    st.markdown("")

    if prediction == 1:
        st.success("### Hasil: SELAMAT (Survived)")
        st.write(
            f"Probabilitas keselamatan: **{proba[1]*100:.1f}%**"
        )
        st.progress(proba[1])
    else:
        st.error("### Hasil: TIDAK SELAMAT (Not Survived)")
        st.write(
            f"Probabilitas tidak selamat: **{proba[0]*100:.1f}%**"
        )
        st.progress(proba[0])
        st.caption(
            "Catatan: Faktor gelar penumpang dan harga tiket cukup berpengaruh "
            "terhadap hasil prediksi."
        )
