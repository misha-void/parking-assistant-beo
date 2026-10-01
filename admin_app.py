"""
Admin approval UI for parking reservations.
Run with: streamlit run admin_app.py

Lists pending reservation requests with their slot details and lets an admin
confirm or refuse each one. Confirming records the booking via the MCP server.
"""

import streamlit as st

from parking_assistant.admin.admin_agent import list_pending_reservations, decide_reservation

st.set_page_config(page_title="Parking Admin", page_icon="🛂", layout="centered")
st.title("🛂 Reservation Approvals")
st.caption("Review pending requests and confirm or refuse them.")

pending = list_pending_reservations()

if not pending:
    st.success("No pending reservations. 🎉")
else:
    for r in pending:
        with st.container(border=True):
            st.markdown(f"**#{r['id']} — {r['name']}**")
            st.write(f"🚗 {r['car_number']}  ·  📍 {r['location']}")
            st.write(f"📅 {r['start']}  →  {r['end']}")
            confirm_col, refuse_col = st.columns(2)
            if confirm_col.button("✅ Confirm", key=f"confirm-{r['id']}", use_container_width=True):
                st.success(decide_reservation(r["id"], "confirm"))
                st.rerun()
            if refuse_col.button("❌ Refuse", key=f"refuse-{r['id']}", use_container_width=True):
                st.warning(decide_reservation(r["id"], "refuse"))
                st.rerun()

if st.button("🔄 Refresh"):
    st.rerun()
