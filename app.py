"""Streamlit chat UI for the parking assistant.

Run with: streamlit run app.py
"""

import asyncio
import uuid

import streamlit as st

from parking_assistant.admin.admin_agent import get_reservation_status
from parking_assistant.service import process_chat_message
from parking_assistant.session_registry import get_reservation

# Statuses worth interrupting the user about, and how to phrase them.
DECIDED_STATUSES = {"confirmed": "confirmed ✅", "cancelled": "refused ❌"}

st.set_page_config(page_title="Parking Assistant", page_icon="🚗", layout="centered")
st.title("🚗 Parking Lot Assistant")
st.caption("Ask me about parking in Belgrade, or book a spot.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "announced_statuses" not in st.session_state:
    # reservation id -> the status already announced, so each decision is shown once
    st.session_state.announced_statuses = {}


@st.fragment(run_every=3)
def announce_decisions() -> None:
    """Tell the user when their reservation is decided, without them having to ask.

    The admin app is a separate process, so the database is the only shared channel.
    """
    reservation_id = get_reservation(st.session_state.session_id)
    if not reservation_id:
        return

    status = get_reservation_status(reservation_id)
    if status not in DECIDED_STATUSES:
        return
    if st.session_state.announced_statuses.get(reservation_id) == status:
        return

    st.session_state.announced_statuses[reservation_id] = status
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            f"🔔 Update: reservation #{reservation_id} was "
            f"{DECIDED_STATUSES[status]} by the administrator."
        ),
    })
    st.rerun(scope="app")


announce_decisions()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = asyncio.run(
                process_chat_message(
                    user_message=prompt,
                    session_id=st.session_state.session_id,
                    chat_history=st.session_state.messages[:-1],
                )
            )
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
