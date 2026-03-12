# NextBuy - Analytics & Prediction

NextBuy is a **data analytics and machine learning project** built on a large grocery shopping dataset.
The objective is to transform raw transactional data into **business insights and predictive models**.

This project includes:

* Exploratory Data Analysis (EDA)
* Machine Learning prediction
* Interactive Streamlit dashboard

The dataset represents **millions of grocery orders** and allows analysis of customer shopping behavior.

---

# Project Structure

```
nextbuy/
│
├── datasets/
│   ├── aisles.csv
│   ├── departments.csv
│   ├── products.csv
│   ├── orders.csv
│   └── order_products.csv
│
├── notebook.ipynb        # Data exploration and model training
├── app.py                # Streamlit dashboard
├── reorder_model.pkl     # Trained ML model
├── requirements.txt      # Python dependencies
└── README.md
```

---

# Dataset Description

The dataset contains several tables describing customer orders.

### orders.csv

Information about every order.

| Column                 | Description                 |
| ---------------------- | --------------------------- |
| order_id               | unique order id             |
| user_id                | customer id                 |
| order_number           | order sequence for the user |
| order_dow              | day of week                 |
| order_hour_of_day      | hour of order               |
| days_since_prior_order | days since previous order   |

---

### order_products.csv

Products included in each order.

| Column            | Description              |
| ----------------- | ------------------------ |
| order_id          | order id         |
| product_id        | product id       |
| add_to_cart_order | position in cart         |
| reordered         | if product was reordered |

---

### products.csv

Product catalog.

| Column        | Description           |
| ------------- | --------------------- |
| product_id    | product id            |
| product_name  | name of product       |
| aisle_id      | aisle identifier      |
| department_id | department identifier |

---

### aisles.csv

Product aisle categories.

| Column        | Description           |
| ------------- | --------------------- |
| aisle_id      | aisle id              |
| aisle         | name of aisle         |

### departments.csv

Product department categories.

| Column        | Description           |
| ------------- | --------------------- |
| department_id | department id         |
| department    | name of department    |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/EpitechBachelorPromo2028/B-DAT-201-PAR-2-1-nextbuy-17.git
cd B-DAT-201-PAR-2-1-nextbuy-17
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# requirements.txt

Example dependencies:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
joblib
jupyter
```

---

# Run the Notebook

To explore the dataset and train the model:

```bash
jupyter notebook notebook.ipynb
```

The notebook contains:

* Data cleaning
* EDA
* Feature engineering
* Behavioral analysis
* Machine learning model training

---

# Machine Learning Model

The trained model is saved as:

```
reorder_model.pkl
```

---

# Run the Dashboard

Launch the interactive dashboard with Streamlit:

```bash
streamlit run app.py
```

---

# License

This project is created for **educational and portfolio purposes**.
