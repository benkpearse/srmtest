import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import chisquare

# 1. Set Page Configuration
st.set_page_config(
    page_title="SRM Calculator | Bayesian Toolkit",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Page Title and Introduction
st.title("⚖️ Sample Ratio Mismatch (SRM) Calculator")
st.markdown(
    """
    This tool checks for a **Sample Ratio Mismatch (SRM)** in your A/B/n test results. 
    An SRM can indicate a problem with your test setup that could invalidate your results.
    """
)

# 3. Sidebar for All User Inputs
with st.sidebar:
    st.header("Experiment Setup")

    num_variants = st.number_input(
        "Number of Variants (including control)",
        min_value=2,
        max_value=10,
        value=2,
        step=1
    )

    st.subheader("Traffic Allocation")
    observed_counts = []
    expected_split = []

    # Create columns for a cleaner layout in the sidebar
    col1, col2 = st.columns(2)

    for i in range(num_variants):
        variant_name = f"Variant {chr(65 + i)}" # A, B, C...
        with col1:
            observed = st.number_input(
                f"Users in {variant_name}",
                min_value=0,
                value=10000,
                step=1,
                key=f"obs_{i}"
            )
            observed_counts.append(observed)
        with col2:
            split = st.number_input(
                f"Split % for {variant_name}",
                min_value=0.0,
                max_value=100.0,
                value=round(100/num_variants, 1),
                step=0.1,
                key=f"split_{i}"
            )
            expected_split.append(split)

    # Validate that the splits add up to 100%
    total_split = sum(expected_split)
    if not np.isclose(total_split, 100.0):
        st.warning(f"Total split must be 100%. Current total: {total_split:.1f}%")

    st.subheader("Settings")
    significance_level = st.slider(
        "Significance Level (α)",
        min_value=0.01,
        max_value=0.10,
        value=0.01,
        step=0.01,
        format="%.2f",
        help="The p-value threshold for detecting an SRM. 0.01 is a common, strict choice."
    )
    
    st.markdown("---")
    run_button = st.button("Check for SRM", type="primary", use_container_width=True)

# 4. Main Page for Displaying Outputs
st.markdown("---")

if run_button:
    # Final validation before running calculation
    if not np.isclose(total_split, 100.0):
        st.error("Cannot run calculation. Please ensure the total expected split adds up to 100%.")
    elif sum(observed_counts) == 0:
        st.error("Cannot run calculation. Please enter the observed user counts.")
    else:
        with st.spinner("Calculating..."):
            total_users = sum(observed_counts)
            
            # Prepare data for display and calculation
            expected_split_decimal = [s / 100.0 for s in expected_split]
            expected_counts = [s * total_users for s in expected_split_decimal]

            summary_data = {
                "Variant": [f"Variant {chr(65 + i)}" for i in range(num_variants)],
                "Observed Users": observed_counts,
                "Expected Split": [f"{s:.1f}%" for s in expected_split],
                "Expected Users": [f"{c:,.1f}" for c in expected_counts]
            }
            summary_df = pd.DataFrame(summary_data)
            
            st.subheader("Summary")
            st.dataframe(summary_df)

            # Perform Chi-square test
            chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)

            st.subheader("Results")
            col1, col2 = st.columns(2)
            col1.metric("Chi-Square Statistic", f"{chi2_stat:.4f}")
            col2.metric("p-value", f"{p_value:.4f}")

            if p_value < significance_level:
                st.error(
                    f"🚫 **SRM Detected.** The p-value ({p_value:.4f}) is less than your significance level ({significance_level}). "
                    "The observed traffic split is significantly different from what you expected."
                )
            else:
                st.success(
                    f"✅ **No SRM Detected.** The p-value ({p_value:.4f}) is greater than your significance level ({significance_level}). "
                    "The observed traffic split is consistent with your expectations."
                )

else:
    st.info("Adjust the parameters in the sidebar and click 'Check for SRM'.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ How to interpret these results"):
    st.markdown("""
    #### What is a Sample Ratio Mismatch (SRM)?
    An SRM occurs when the observed number of users in each variant is statistically different from the expected number. For example, in a 50/50 A/B test, you get 45% of users in A and 55% in B. This can indicate a bug in your randomization or tracking, which can invalidate your entire experiment.

    #### How does the Chi-Square Test work here?
    The Chi-square (χ²) goodness-of-fit test measures how well your observed traffic distribution fits the expected distribution. A large Chi-square statistic suggests a poor fit.

    #### How to Interpret the p-value
    The **p-value** tells you the probability of seeing a traffic split at least as imbalanced as yours, *assuming the tracking is working correctly*.
    - **A low p-value (e.g., < 0.01)** is a red flag. It means your result is very unlikely to be due to random chance, and you likely have a real problem (SRM).
    - **A high p-value (e.g., > 0.01)** means the observed imbalance is likely just due to normal random variation.

    #### What should I do if I find an SRM?
    **Do not trust the results of the experiment.** You should immediately pause the test, investigate the root cause of the allocation issue (e.g., faulty randomization logic, redirects, tracking pixel errors), fix it, and restart the experiment from scratch.
    """)
