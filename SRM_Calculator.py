import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import chi2, chisquare
import altair as alt

# 1. Set Page Configuration
st.set_page_config(
    page_title="SRM Calculator | Bayesian Toolkit",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- REWRITTEN ALTAIR PLOTTING FUNCTION ---
def plot_srm_altair_chart(chi2_stat, p_value, df, significance_level):
    """
    Generates an interpretable plot of the Chi-square distribution using Altair.
    This version uses alt.layer() for robust rendering.
    """
    critical_value = chi2.ppf(1 - significance_level, df)
    x_max = max(critical_value * 2, chi2.ppf(0.999, df))
    x = np.linspace(0, x_max, 500)
    
    # Create a DataFrame for plotting the distribution curve
    source = pd.DataFrame({
        'x': x,
        'pdf': chi2.pdf(x, df)
    })

    # Base chart for the distribution line
    line = alt.Chart(source).mark_line().encode(
        x=alt.X('x:Q', title='Chi-Square Statistic (χ²)'),
        y=alt.Y('pdf:Q', title='Probability Density')
    )

    # Shaded rejection region
    rejection_region = alt.Chart(source).mark_area(opacity=0.5, color='salmon').encode(
        x='x:Q',
        y='pdf:Q'
    ).transform_filter(
        alt.datum.x >= critical_value
    )

    # Create a DataFrame for the vertical lines to add tooltips
    rules_df = pd.DataFrame({
        'x': [critical_value, chi2_stat],
        'color': ['darkred', 'black'],
        'strokeDash': [[3, 3], [0, 0]], # Dashed for critical, solid for observed
        'label': [f'Critical Value (α={significance_level})', 'Observed Statistic']
    })

    rules = alt.Chart(rules_df).mark_rule(size=2).encode(
        x='x:Q',
        color=alt.Color('color:N', scale=None),
        strokeDash=alt.StrokeDash('strokeDash:N', scale=None),
        tooltip=['label', alt.Tooltip('x:Q', format='.2f')]
    )
    
    # Combine all layers into a single chart
    chart = alt.layer(
        line, rejection_region, rules
    ).properties(
        title="Chi-Square Test for Sample Ratio Mismatch"
    ).interactive()

    return chart


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
        min_value=2, max_value=10, value=2, step=1
    )

    st.subheader("Traffic Allocation")
    
    split_mode = st.radio(
        "Expected Traffic Split",
        ["Assume Equal Split", "Enter Custom Split"],
        horizontal=True,
        help="Choose 'Equal' for a standard test or 'Custom' for uneven splits (e.g., 90/10)."
    )

    observed_counts = []
    expected_split = []
    variant_names = []

    if split_mode == "Assume Equal Split":
        st.caption("Enter the observed user counts for each variant.")
        for i in range(num_variants):
            if i == 0:
                variant_name = "Control"
            else:
                variant_name = f"Variant {i}"
            variant_names.append(variant_name)
            observed = st.number_input(
                f"Users in {variant_name}",
                min_value=0, value=10000, step=1, key=f"obs_{i}"
            )
            observed_counts.append(observed)
        expected_split = [100.0 / num_variants] * num_variants

    else: # Custom Split
        st.caption("Enter observed counts and the expected split percentage for each variant.")
        col1, col2 = st.columns(2)
        for i in range(num_variants):
            if i == 0:
                variant_name = "Control"
            else:
                variant_name = f"Variant {i}"
            variant_names.append(variant_name)
            with col1:
                observed = st.number_input(
                    f"Users in {variant_name}",
                    min_value=0, value=10000, step=1, key=f"obs_{i}"
                )
                observed_counts.append(observed)
            with col2:
                split = st.number_input(
                    f"Split %",
                    min_value=0.0, max_value=100.0, value=round(100/num_variants, 1), step=0.1, key=f"split_{i}"
                )
                expected_split.append(split)
        
        total_split = sum(expected_split)
        if not np.isclose(total_split, 100.0):
            st.warning(f"Total split must be 100%. Current total: {total_split:.1f}%")

    st.subheader("Settings")
    significance_level = st.slider(
        "Significance Level (α)",
        min_value=0.01, max_value=0.10, value=0.01, step=0.01, format="%.2f",
        help="The p-value threshold for detecting an SRM. 0.01 is a common, strict choice."
    )
    
    st.markdown("---")
    run_button = st.button("Check for SRM", type="primary", use_container_width=True)

# 4. Main Page for Displaying Outputs
st.markdown("---")

if run_button:
    if split_mode == "Enter Custom Split" and not np.isclose(sum(expected_split), 100.0):
        st.error("Cannot run calculation. Please ensure the total custom split adds up to 100%.")
    elif sum(observed_counts) == 0:
        st.error("Cannot run calculation. Please enter the observed user counts.")
    else:
        with st.spinner("Calculating..."):
            total_users = sum(observed_counts)
            
            expected_split_decimal = [s / 100.0 for s in expected_split]
            expected_counts = [s * total_users for s in expected_split_decimal]

            summary_data = {
                "Variant": variant_names, # Use the generated names
                "Observed Users": observed_counts,
                "Expected Split": [f"{s:.1f}%" for s in expected_split],
                "Expected Users": [f"{c:,.1f}" for c in expected_counts]
            }
            summary_df = pd.DataFrame(summary_data)
            
            st.subheader("Summary")
            st.dataframe(summary_df)

            chi2_stat, p_value = chisquare(f_obs=observed_counts, f_exp=expected_counts)
            df = num_variants - 1 # Degrees of freedom

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
            
            st.subheader("Visualization")
            chart = plot_srm_altair_chart(chi2_stat, p_value, df, significance_level)
            st.altair_chart(chart, use_container_width=True)

else:
    st.info("Adjust the parameters in the sidebar and click 'Check for SRM'.")

# 5. Explanations Section
st.markdown("---")
with st.expander("ℹ️ How to interpret these results"):
    st.markdown("""
    #### What is a Sample Ratio Mismatch (SRM)?
    An SRM occurs when the observed number of users in each variant is statistically different from the expected number. For example, in a 50/50 A/B test, you get 45% of users in A and 55% in B. This can indicate a bug in your randomization or tracking, which can invalidate your entire experiment.

    #### How to Interpret the Visualization
    The plot shows the Chi-square (χ²) distribution for your test setup. This curve represents the range of outcomes you'd expect to see due to normal random chance if your tracking were working perfectly.
    - The **red dotted line** shows the **Critical Value**. If your result is to the right of this line, it's statistically significant.
    - The **black solid line** is your test's actual result (the "Observed Statistic").
    - The **shaded red area** is the "Rejection Region." If your observed statistic falls in this area, you have a clear SRM.
    
    If your observed statistic is far to the right of the critical value, it means your result was highly unlikely to have occurred by chance, signaling a probable SRM.

    #### What should I do if I find an SRM?
    **Do not trust the results of the experiment.** You should immediately pause the test, investigate the root cause of the allocation issue (e.g., faulty randomization logic, redirects, tracking pixel errors), fix it, and restart the experiment from scratch.
    """)
