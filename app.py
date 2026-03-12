import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

st.set_page_config(layout="wide")

st.title("NextBuy Dashboard")

# ---------- LOAD DATA ----------

@st.cache_data
def load_data(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    return df

path = st.sidebar.text_input("Dataset path", "datasets/cleaned_dataset.csv")

fast_mode = st.sidebar.checkbox("Fast mode (load fewer rows)", True)

rows = None
if fast_mode:
    rows = st.sidebar.number_input("Rows", 50000, 5000000, 300000, step=50000)

df = load_data(path, rows)

days = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}

df["day_name"] = df["order_dow"].map(days)

day_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# ---------- KPIs ----------

orders = df["order_id"].nunique()
users = df["user_id"].nunique()
items = len(df)
reorder_rate = df["reordered"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Orders", orders)
c2.metric("Users", users)
c3.metric("Items", items)
c4.metric("Reorder rate", f"{reorder_rate:.2%}")

st.divider()

# ---------- HEATMAP ----------

st.subheader("Orders by day and hour")

heat = (
    df.groupby(["day_name", "order_hour_of_day"])["order_id"]
    .nunique()
    .reset_index(name="orders")
)

fig = px.density_heatmap(
    heat,
    x="order_hour_of_day",
    y="day_name",
    z="orders",
    color_continuous_scale="Blues",
    category_orders={"day_name": day_order}
)

st.plotly_chart(fig, use_container_width=True)

# ---------- TOP DEPARTMENTS ----------

st.subheader("Top departments")

dept = (
    df.groupby("department")["order_id"]
    .nunique()
    .reset_index(name="orders")
    .sort_values("orders", ascending=False)
    .head(10)
)

fig = px.bar(dept, x="department", y="orders")

st.plotly_chart(fig, use_container_width=True)

# ---------- BASKET SIZE ----------

st.subheader("Basket size distribution")

basket = (
    df.groupby("order_id")["product_id"]
    .count()
    .reset_index(name="basket_size")
)

fig = px.histogram(basket, x="basket_size", nbins=50)

st.plotly_chart(fig, use_container_width=True)

# ---------- PRODUCT SEARCH ----------

st.header("Product analysis")

product_name = st.text_input("Type product name")

if product_name:

    product_df = df[
        df["product_name"].str.contains(product_name, case=False, na=False)
    ]

    if len(product_df) > 0:

        purchases = len(product_df)
        reorder = product_df["reordered"].mean()

        c1, c2 = st.columns(2)

        c1.metric("Total purchases", purchases)
        c2.metric("Reorder rate", f"{reorder:.2%}")

        st.subheader("Orders by hour")

        fig = px.histogram(product_df, x="order_hour_of_day", nbins=24)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Orders by day")

        fig = px.histogram(product_df, x="day_name", nbins=7, category_orders={"day_name": day_order})

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("Product not found")

# ---------- USER ANALYSIS ----------

st.header("Customer analysis")

user_id = st.number_input("Enter user id", step=1)

if user_id:

    user_df = df[df["user_id"] == user_id]

    if len(user_df) > 0:

        orders = user_df["order_id"].nunique()

        basket = (
            user_df.groupby("order_id")["product_id"]
            .count()
        )

        avg_basket = basket.mean()

        avg_days = user_df["days_since_prior_order"].mean()

        c1, c2, c3 = st.columns(3)

        c1.metric("Orders", orders)
        c2.metric("Average basket", round(avg_basket, 2))
        c3.metric("Average reorder days", round(avg_days, 1))

        st.subheader("Orders by hour")

        hour = (
            user_df.groupby("order_hour_of_day")["order_id"]
            .nunique()
            .reset_index(name="orders")
        )

        fig = px.bar(hour, x="order_hour_of_day", y="orders")

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Orders by day")

        dow = (
            user_df.groupby("day_name")["order_id"]
            .nunique()
            .reset_index(name="orders")
        )

        fig = px.bar(dow, x="day_name", y="orders", category_orders={"day_name": day_order})

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("User not found")


# LOAD MODEL

model_package = joblib.load("reorder_model.pkl")

model = model_package["model"]
features = model_package["features"]
accuracy = model_package["accuracy"]
model_type = model_package["model_type"]


# BUILD FEATURES

def reorder_df(df):
    
    user_col = df.groupby('user_id').agg(
        user_total_orders = ('order_number', 'max'),
        user_moy_basket_size = ('product_id', 'count'),
        user_moy_reorder_days = ('days_since_prior_order', 'mean'),
    ).reset_index()

    # collones produit
    prod_col = df.groupby('product_id').agg(
        product_total_orders = ('order_id', 'count'),
        product_reorder_rate = ('reordered', 'mean'),
    ).reset_index()

    # Collones des 2
    fus_col = df.groupby(['user_id', 'product_id']).agg(
        fus_order_count = ('order_id', 'count'),
        fus_avg_cart_position = ('add_to_cart_order', 'mean'),
        fus_last_order = ('order_number', 'max'),
    ).reset_index()

    # Création de la nouvelle df
    df_new = (df[['user_id', 'product_id', 'reordered']]
                .merge(user_col, on='user_id', how='left')
                .merge(prod_col, on='product_id', how='left')
                .merge(fus_col, on=['user_id', 'product_id'], how='left')
                .drop_duplicates(subset=['user_id', 'product_id'])
                .reset_index(drop=True))
    return df_new


# MODEL PREDICTION

st.header("Reorder prediction model")

st.subheader("Model Information")

c1, c2 = st.columns(2)

c1.metric(
    "Model type",
    model_type
)

c2.metric(
    "Model accuracy",
    f"{accuracy:.2%}"
)
user_id_input = st.number_input("User ID", step=1)

product_name_input = st.text_input("Product name")

df_features = reorder_df(df)

if user_id_input and product_name_input:

    product_row = df[
        df["product_name"].str.contains(product_name_input, case=False, na=False)
    ]

    if len(product_row) > 0:

        product_id_input = product_row.iloc[0]["product_id"]

        row = df_features[
            (df_features["user_id"] == user_id_input) &
            (df_features["product_id"] == product_id_input)
        ]

        if len(row) > 0:

            X = row[features]

            probability = model.predict_proba(X)[0][1]

            st.metric(
                "Reorder probability",
                f"{probability:.1%}"
            )

        else:

            st.warning("This user has never bought this product before.")

    else:

        st.warning("Product not found.")




# ---------- DATA PREVIEW ----------

with st.expander("Show data"):
    st.dataframe(df.head(200))