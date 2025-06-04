import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.stats import truncnorm

# Define feature lists
numeric_features = ['umur_pasien', 'showing_count', 'appointment_leadtime', 'showing_rate']
categorical_features = [
    'high_no_show_doctor',
    'pasien_lama_baru',
    'previous_no_show',
    'asuransi',
    'previous2_show',
    'previous2_no_show',
    'pertama_daftar'
]

# Hard-coded categorical options
categorical_options = {
    'pasien_lama_baru': ['Pasien Lama', 'Pasien Baru'],
    'asuransi': ['UMUM', 'BPJS Kesehatan', 'Lainnya'],
}

# Binary feature probabilities (based on dataset means)
binary_probs = {
    'high_no_show_doctor': 0.105194,
    'previous_no_show': 0.169187,
    'previous2_show': 0.325005,
    'previous2_no_show': 0.036160,
    'pertama_daftar': 0.396669
}

# Numeric feature distributions (mean, std, min, max)
numeric_stats = {
    'umur_pasien': {'mean': 34.512163, 'std': 15.859298, 'min': 0, 'max': 124},
    'showing_count': {'mean': 2.858207, 'std': 5.087039, 'min': 0, 'max': 40},
    'appointment_leadtime': {'mean': 2.644970, 'std': 3.148413, 'min': 0, 'max': 36},
    'showing_rate': {'mean': 0.358731, 'std': 0.356909, 'min': 0.0, 'max': 0.965517}
}

# Helper for truncated normal
def sample_truncnorm(mean, sd, low, high, size):
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)

@st.cache_data()
def load_model(model_path: str = 'lr.pkl') -> object:
    """
    Load and cache the trained logistic regression model pipeline.
    Ensure 'lr.pkl' is in the same directory or provide the correct path.
    """
    return joblib.load(model_path)

# Load model
model = load_model()

# App layout
st.title('Appointment No-Show Prediction & Simulation')
st.sidebar.header('Navigation')
mode = st.sidebar.radio('Select Page', ['Single Prediction', 'Simulation'])

# Function for single patient input
def get_single_input() -> pd.DataFrame:
    data = {}
    for feat in numeric_features:
        stats = numeric_stats[feat]
        if feat != 'showing_rate':
        #     min_v, max_v = stats['min'], stats['max']
            default, step, fmt = int(stats['mean']), 1, '%d'
        else:
        #     min_v, max_v = float(stats['min']), float(stats['max'])
            default, step, fmt = float(stats['mean']), 0.01, '%.2f'
        data[feat] = st.sidebar.number_input(
            label=feat.replace('_', ' ').title(),
            # min_value=min_v,
            # max_value=max_v,
            value=default,
            step=step,
            format=fmt
        )
    # Categorical inputs
    data['pasien_lama_baru'] = st.sidebar.selectbox(
        'Pasien Lama/Baru', categorical_options['pasien_lama_baru']
    )
    data['asuransi'] = st.sidebar.selectbox(
        'Asuransi', categorical_options['asuransi']
    )
    for feat in ['high_no_show_doctor', 'previous_no_show', 'previous2_show', 'previous2_no_show', 'pertama_daftar']:
        data[feat] = st.sidebar.selectbox(
            feat.replace('_', ' ').title(), [True, False]
        )
    return pd.DataFrame(data, index=[0])

# Single Prediction Page
if mode == 'Single Prediction':
    st.sidebar.subheader('Single Patient Details')
    input_df = get_single_input()
    st.subheader('Input Summary')
    st.write(input_df)

    # Main Predict button
    if st.button('Predict Single Patient'):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        st.subheader('Prediction')
        st.write('No-Show' if bool(pred) else 'Show')
        st.write({'Show': proba[0], 'No-Show': proba[1]})

# Simulation Page
if mode == 'Simulation':
    st.sidebar.subheader('Simulation Settings')
    total_sim = st.sidebar.number_input(
        'Total Patients to Simulate', min_value=1, value=100, step=1
    )
    st.subheader('Simulation Settings')
    st.write(f'Number of patients to simulate: **{total_sim}**')

    # Main Run Simulation button
    if st.button('Run Simulation'):
        size = int(total_sim)
        sim = pd.DataFrame()
        # Numeric features
        for feat, stats in numeric_stats.items():
            vals = sample_truncnorm(
                stats['mean'], stats['std'], stats['min'], stats['max'], size
            )
            if feat in ['umur_pasien', 'showing_count', 'appointment_leadtime']:
                sim[feat] = np.round(vals).astype(int)
            else:
                sim[feat] = np.round(vals, 2)
        # Categorical features
        for feat, p in binary_probs.items():
            sim[feat] = np.random.choice([True, False], size=size, p=[p, 1-p])
        sim['pasien_lama_baru'] = np.random.choice(
            categorical_options['pasien_lama_baru'], size=size,
            p=[0.96342648, 0.03657352]
        )
        sim['asuransi'] = np.random.choice(
            categorical_options['asuransi'], size=size,
            p=[0.77793445, 0.20838337, 0.01368218]
        )

        # Predict on all simulated
        preds = model.predict(sim)
        sim['predicted_no_show'] = preds.astype(bool)

        # Summary counts
        no_show_count = int(sim['predicted_no_show'].sum())
        show_count = size - no_show_count
        st.subheader('Simulation Results')
        st.write(f"- Predicted Show: **{show_count}**")
        st.write(f"- Predicted No-Show: **{no_show_count}**")
        st.write(f"- No-Show Rate: **{no_show_count/size:.2%}**")

        # Display patients predicted as no-show
        st.subheader('Patients Predicted as No-Show')
        sim_no_show = sim[sim['predicted_no_show']]
        st.dataframe(sim_no_show.reset_index(drop=True))
