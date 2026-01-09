import streamlit as st

def metric_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for c, (k, v) in zip(cols, items):
        c.metric(k, v)

def profile_card(profile: dict):
    with st.container(border=True):
        st.markdown(f"### {profile['column']}")
        st.caption(f"Type: {profile['dtype']}")

        metric_row([
            ("Missing %", f"{profile['missing_pct']:.2f}%"),
            ("Unique", str(profile["unique"])),
        ])

        if "mean" in profile:
            metric_row([
                ("Mean", f"{profile['mean']:.3f}"),
                ("Median", f"{profile['median']:.3f}"),
                ("Min", f"{profile['min']:.3f}"),
                ("Max", f"{profile['max']:.3f}"),
            ])
        else:
            st.markdown("**Top values**")
            if profile.get("top_values"):
                for k, v in profile["top_values"].items():
                    st.write(f"- {k}: {v}")
            else:
                st.write("N/A")
