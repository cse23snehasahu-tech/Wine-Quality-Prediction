
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction System")
st.write(
    "Enter the wine's chemical properties "
    "to predict its quality using Machine Learning."
)

@st.cache_resource
def train_model():
    data = pd.read_csv("WineQT.csv")

    X = data.drop(columns=["quality", "Id"], errors="ignore")
    y = data["quality"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    return model, X, y

try:
    model, X, y = train_model()

    st.subheader("Enter Wine Properties")

    with st.form("wine_form"):
        values = {}

        for column in X.columns:
            minimum = float(X[column].min())
            maximum = float(X[column].max())
            default = float(X[column].median())

            values[column] = st.number_input(
                column.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=default,
                format="%.3f"
            )

        submitted = st.form_submit_button(
            "Predict Wine Quality"
        )

    if submitted:
        input_data = pd.DataFrame(
            [values],
            columns=X.columns
        )

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Wine Quality: {prediction}")

        st.caption(
            "This is a machine learning prediction, "
            "not a laboratory quality assessment."
        )

except Exception as error:
    st.error(f"Unable to load the project: {error}")
    st.info(
        "Make sure WineQT.csv is present "
        "in the project folder."
    )
