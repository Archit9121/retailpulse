"""RetailPulse dashboard: inventory optimization recommendations."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import streamlit as st  # noqa: E402

from src.dashboard.data_loader import load_inventory_recommendations  # noqa: E402
from src.optimization.inventory import (  # noqa: E402
    HOLDING_COST_RATE,
    LEAD_TIME_DAYS,
    ORDER_COST,
    SERVICE_LEVEL,
    simulate_reorder_trigger,
)

st.set_page_config(page_title="Inventory — RetailPulse", page_icon="📦", layout="wide")
st.title("📦 Inventory Optimization")

recommendations = load_inventory_recommendations()
if recommendations is None:
    st.info("Run `python -m src.optimization.inventory` to generate recommendations.")
    st.stop()

with st.expander("Assumptions for optimization"):
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Lead time | {LEAD_TIME_DAYS} days | 
| Service level | {SERVICE_LEVEL:.0%} | 
| Order cost | £{ORDER_COST:.0f}/PO | 
| Holding cost rate | {HOLDING_COST_RATE:.0%}/year of unit price |

""")

st.subheader("ABC tiers")
tier_summary = (
    recommendations.groupby("tier")
    .agg(n_products=("tier", "size"), total_revenue=("total_revenue", "sum"))
    .reindex(["A", "B"])
)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tier A products", f"{tier_summary.loc['A', 'n_products']:,.0f}")
with col2:
    st.metric("Tier B products", f"{tier_summary.loc['B', 'n_products']:,.0f}")
with col3:
    st.metric("Total optimized revenue", f"£{tier_summary['total_revenue'].sum():,.0f}")

st.divider()
st.subheader("Reorder recommendations")

col1, col2, col3 = st.columns(3)
with col1:
    tier_filter = st.multiselect("Tier", options=["A", "B"], default=["A", "B"])
with col2:
    search = st.text_input("Search description or stock code")
with col3:
    sort_by = st.selectbox("Sort by", ["total_revenue", "reorder_point", "eoq", "safety_stock"])

filtered = recommendations[recommendations["tier"].isin(tier_filter)]
if search:
    mask = filtered["description"].str.contains(search, case=False, na=False) | filtered[
        "stock_code"
    ].str.contains(search, case=False, na=False)
    filtered = filtered[mask]
filtered = filtered.sort_values(sort_by, ascending=False)

display_cols = [
    "stock_code",
    "description",
    "tier",
    "daily_demand_mean",
    "safety_stock",
    "reorder_point",
    "eoq",
]
st.dataframe(
    filtered[display_cols]
    .head(200)
    .style.format(
        {
            "daily_demand_mean": "{:.1f}",
            "safety_stock": "{:.0f}",
            "reorder_point": "{:.0f}",
            "eoq": "{:.0f}",
        }
    ),
    width="stretch",
)


st.download_button(
    "Download recommendations (CSV)",
    filtered[display_cols].to_csv(index=False),
    file_name="inventory_recommendations.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Reorder trigger")


product_options = (filtered["stock_code"] + " — " + filtered["description"]).tolist()
if st.session_state.get("inventory_product_selector") not in product_options:
    st.session_state["inventory_product_selector"] = product_options[0] if product_options else None

selected = None
if not product_options:
    st.info("No products match the current filters.")
else:
    selected = st.selectbox("Product", options=product_options, key="inventory_product_selector")

if selected:
    stock_code = selected.split(" — ")[0]
    product = recommendations[recommendations["stock_code"] == stock_code].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reorder point", f"{product['reorder_point']:.0f} units")
        st.metric("Safety stock", f"{product['safety_stock']:.0f} units")
    with col2:
        st.metric("EOQ", f"{product['eoq']:.0f} units")
        st.metric("Avg daily demand", f"{product['daily_demand_mean']:.1f} units")

    current_stock = st.slider(
        "Current stock level",
        0,
        int(max(product["reorder_point"] * 2, 100)),
        int(product["reorder_point"]),
    )
    result = simulate_reorder_trigger(current_stock, product["reorder_point"], product["eoq"])

    if result["should_reorder"]:
        st.error(
            f"🔴 Below reorder point ({product['reorder_point']:.0f}). "
            f"Recommended order: {result['recommended_order_qty']:.0f} units."
        )
    else:
        st.success(f"🟢 Above reorder point ({product['reorder_point']:.0f}). No order needed yet.")
