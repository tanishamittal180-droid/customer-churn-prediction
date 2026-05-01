import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Wrong credentials")
    st.stop()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists("model.joblib"):

        np.random.seed(42)
        n = 2000

        df = pd.DataFrame({
            "billing": np.random.randint(100,1000,n),
            "tenure": np.random.randint(1,60,n),
            "usage": np.random.randint(5,200,n),
            "active_days": np.random.randint(1,30,n),
            "logins": np.random.randint(1,50,n),
            "session_time": np.random.randint(5,120,n),
            "tickets": np.random.randint(0,10,n),
            "sla": np.random.randint(0,5,n),
            "emails": np.random.randint(0,50,n),
            "clicks": np.random.randint(0,20,n),
            "last_payment": np.random.randint(0,60,n),
            "nps": np.random.randint(-10,10,n),
            "autopay": np.random.randint(0,2,n),
            "discount": np.random.randint(0,2,n),
            "devices": np.random.randint(1,5,n)
        })

        # engineered features
        df["engagement"] = df["active_days"]/30
        df["ctr"] = df["clicks"]/(df["emails"]+1)
        df["risk"] = df["billing"]/(df["tenure"]+1)

        df["churn"] = (
            (df["tickets"] > 5) |
            (df["sla"] > 2) |
            (df["tenure"] < 6) |
            (df["engagement"] < 0.3)
        ).astype(int)

        X = df.drop("churn", axis=1)
        y = df["churn"]

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        joblib.dump(model, "model.joblib")

    return joblib.load("model.joblib")

model = load_model()

# ---------------- UI ----------------
st.title("📊 Customer Churn Dashboard")

mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch Upload"])

# ---------------- SINGLE ----------------
if mode == "Single Prediction":

    st.header("Enter Customer Data")

    col1, col2, col3 = st.columns(3)

    billing = col1.number_input("Billing",100,1000,500)
    tenure = col1.number_input("Tenure",1,60,12)
    usage = col1.number_input("Usage",5,200,50)

    active = col2.number_input("Active Days",1,30,20)
    logins = col2.number_input("Logins",1,50,10)
    session = col2.number_input("Session Time",5,120,30)

    tickets = col3.number_input("Tickets",0,10,2)
    sla = col3.number_input("SLA",0,5,1)
    emails = col3.number_input("Emails",0,50,10)

    clicks = st.number_input("Clicks",0,20,5)
    last = st.number_input("Last Payment Days",0,60,10)
    nps = st.slider("NPS",-10,10,5)

    autopay = st.selectbox("Autopay",[0,1])
    discount = st.selectbox("Discount",[0,1])
    devices = st.number_input("Devices",1,5,2)

    if st.button("Predict"):

        df = pd.DataFrame([{
            "billing": billing,
            "tenure": tenure,
            "usage": usage,
            "active_days": active,
            "logins": logins,
            "session_time": session,
            "tickets": tickets,
            "sla": sla,
            "emails": emails,
            "clicks": clicks,
            "last_payment": last,
            "nps": nps,
            "autopay": autopay,
            "discount": discount,
            "devices": devices
        }])

        df["engagement"] = df["active_days"]/30
        df["ctr"] = df["clicks"]/(df["emails"]+1)
        df["risk"] = df["billing"]/(df["tenure"]+1)

        prob = model.predict_proba(df)[0][1]

        st.metric("Churn Probability", round(prob,2))

        # FEATURE IMPORTANCE GRAPH
        st.subheader("Feature Importance")

        imp = model.feature_importances_
        names = model.feature_names_in_

        imp_df = pd.DataFrame({"Feature": names, "Importance": imp})
        imp_df = imp_df.sort_values(by="Importance")

        fig, ax = plt.subplots()
        ax.barh(imp_df["Feature"], imp_df["Importance"])
        st.pyplot(fig)

# ---------------- BATCH ----------------
else:

    st.header("Upload CSV")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        df["engagement"] = df["active_days"]/30
        df["ctr"] = df["clicks"]/(df["emails"]+1)
        df["risk"] = df["billing"]/(df["tenure"]+1)

        probs = model.predict_proba(df)[:,1]
        df["churn_probability"] = probs

        df["risk_level"] = pd.cut(
            df["churn_probability"],
            bins=[0,0.25,0.5,1],
            labels=["Low","Medium","High"]
        )

        # KPIs
        col1,col2,col3 = st.columns(3)
        col1.metric("Customers", len(df))
        col2.metric("High Risk", (df["risk_level"]=="High").sum())
        col3.metric("Avg Risk", round(df["churn_probability"].mean(),2))

        # GRAPH 1
        st.subheader("Risk Distribution")
        fig, ax = plt.subplots()
        df["risk_level"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        # GRAPH 2
        st.subheader("Probability Histogram")
        fig, ax = plt.subplots()
        df["churn_probability"].plot(kind="hist", bins=20, ax=ax)
        st.pyplot(fig)

        # TABLE
        st.dataframe(df)

        st.download_button("Download Report", df.to_csv(index=False))
        