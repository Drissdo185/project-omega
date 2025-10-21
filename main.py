# app/ui/streamlit_app.py
import os
import json
import re
import streamlit as st
from app.chat.manager import process_user_message
from app.storage.json_store import _load_index

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(page_title="AI PDF QA – Strict HR/IT Assistant", layout="wide")

# -------------------------------------------------
# Initialize Session State
# -------------------------------------------------
defaults = {
    "history": [],
    "contexts_history": [],
    "sources_history": [],
    "call_log": [],
    "user_input": "",
    "clear_input_flag": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------------------------
# Sidebar — Conversation Timeline
# -------------------------------------------------
# in app/ui/streamlit_app.py — replace Sidebar Conversation Timeline block with this

st.sidebar.header("📜 Conversation Timeline")

log_path = os.path.join("logs", "call_log.json")

# try to read file from disk every render (keeps UI in sync with manager writes)
disk_call_log = []
if os.path.exists(log_path):
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            disk_call_log = json.load(f) or []
            if not isinstance(disk_call_log, list):
                disk_call_log = []
    except Exception as e:
        st.sidebar.error(f"Failed to read call_log.json: {e}")
        disk_call_log = []

# ensure session_state.call_log exists and merge (disk entries are authoritative)
st.session_state.setdefault("call_log", [])

# merge: keep disk_call_log as the source of truth;
# but if session_state has extra entries not yet on disk, append them and persist.
# (this handles rare race cases)
# Build a simple set of timestamps to detect duplicates
disk_ts = set(e.get("timestamp") for e in disk_call_log)
merged = list(disk_call_log)  # start from disk

# append any in-memory entries that aren't in disk
for e in list(st.session_state.get("call_log", [])):
    if e.get("timestamp") not in disk_ts:
        merged.append(e)
        disk_ts.add(e.get("timestamp"))

# save merged back into session_state
st.session_state["call_log"] = merged

# display most recent entries (reverse order)
if st.session_state.call_log:
    for entry in reversed(st.session_state.call_log[-50:]):
        ts = entry.get("timestamp", "")
        fn = entry.get("function", "") or "N/A"
        args = entry.get("args", {})
        result = entry.get("result", "")
        st.sidebar.markdown(f"**{fn}** — `{ts}`")
        st.sidebar.caption(f"Args: {args}\nResult: {result}")
        st.sidebar.markdown("---")
else:
    st.sidebar.info("No function calls recorded yet.")

if st.sidebar.button("🧹 Clear Timeline"):
    # clear both in-memory and disk
    st.session_state.call_log = []
    try:
        if os.path.exists(log_path):
            os.remove(log_path)
    except Exception as e:
        st.sidebar.error(f"Could not remove log file: {e}")
    st.rerun()


# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("💬 AI PDF QA — Strict HR / IT / Other Assistant")
st.caption("Answers strictly from ingested PDF context with numbered [1][2][3] listings.")

# -------------------------------------------------
# Render Chat Messages (modern UX)
# -------------------------------------------------
assistant_idx = -1  # track assistant messages separately

for i, turn in enumerate(st.session_state.history):
    # ---------- USER MESSAGE ----------
    if turn["role"] == "user":
        st.markdown(
            f"""
            <div style='display:flex; justify-content:flex-end; margin:6px 0;'>
                <div style='background:linear-gradient(135deg, #00b37e, #009970);
                color:white; padding:10px 14px; border-radius:16px 16px 4px 16px;
                max-width:75%; font-size:15px; box-shadow:0 2px 4px rgba(0,0,0,0.2);'>
                    <b>({turn.get('folder','')})</b> {turn['text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- ASSISTANT MESSAGE ----------
    elif turn["role"] == "assistant":
        assistant_idx += 1
        answer = turn["text"]

        if answer.strip().lower() == "i don't know":
            st.markdown(
                f"""
                <div style='display:flex; justify-content:flex-start; margin:6px 0;'>
                    <div style='background:#f1f3f4; color:#222; padding:10px 14px;
                    border-radius:16px 16px 16px 4px; max-width:80%;
                    font-size:15px; box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
                        🤔 I don't know
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # format context output
            safe_html = (
                answer.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
                .replace("Sources:", "<hr><b>Sources:</b><br>")
            )
            for n, color in zip(range(1, 6),
                ["#2E8B57", "#1E90FF", "#FF8C00", "#DA70D6", "#A52A2A"]
            ):
                safe_html = safe_html.replace(f"[{n}]", f"<b style='color:{color};'>[{n}]</b>")

            st.markdown(
                f"""
                <div style='display:flex; justify-content:flex-start; margin:6px 0;'>
                    <div style='background-color:#fefefe; color:#111;
                    padding:12px 16px; border-radius:16px 16px 16px 4px;
                    max-width:85%; font-family:Segoe UI, sans-serif;
                    line-height:1.6; font-size:15px;
                    box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
                        {safe_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------- SOURCES ----------
            if assistant_idx < len(st.session_state.sources_history):
                sources = st.session_state.sources_history[assistant_idx]
                if sources:
                    st.markdown(
                        f"""
                        <div style='margin:6px 0 16px 20px; font-size:13px;
                        font-family:monospace; color:#555;'>
                            📚 <b>Sources:</b><br>
                            {"".join([
                                f"<div style='margin-left:12px;'>[{i+1}] <code>{os.path.basename(s.get('doc_path',''))}</code> — <span style=\"color:#777;\">{s.get('doc_path','')}</span></div>"
                                for i, s in enumerate(sources)
                            ])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------- AUTO-SCROLL TO BOTTOM ----------
st.markdown("<div id='chat_end'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <script>
        var chatDiv = document.getElementById('chat_end');
        if (chatDiv) {
            chatDiv.scrollIntoView({behavior: 'smooth'});
        }
    </script>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Input Section (safe clear)
# -------------------------------------------------
st.markdown("---")

query = st.text_input(
    "Ask a question (auto HR / IT / Other detection):",
    value=st.session_state.get("user_input", ""),
    key="chat_input_value",
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("📤 Send", use_container_width=True):
        q = (st.session_state.get("chat_input_value") or "").strip()
        if q:
            process_user_message(st.session_state, q, top_k=3)
            st.session_state["user_input"] = ""
            st.session_state["clear_input_flag"] = True
            st.rerun()
        else:
            st.warning("Please enter a question.")

# Safe clear for Streamlit ≥ 1.37
if st.session_state.get("clear_input_flag", False):
    if "chat_input_value" in st.session_state:
        del st.session_state["chat_input_value"]
    st.session_state["clear_input_flag"] = False

with col2:
    if st.button("🧹 Clear Chat", use_container_width=True):
        for key in ["history", "contexts_history", "sources_history"]:
            st.session_state[key] = []
        st.rerun()

# -------------------------------------------------
# Indexed Files Viewer
# -------------------------------------------------
st.markdown("---")
st.subheader("📂 Indexed Files")

index = _load_index()
if index:
    for entry in sorted(index, key=lambda x: x.get("created_at", ""), reverse=True)[:30]:
        filename = os.path.basename(entry["path"])
        label = entry.get("label", "Unknown")
        num_sections = entry.get("num_sections", 0)
        st.markdown(f"- **{filename}** ({label}) — {num_sections} sections")
else:
    st.info("No indexed files found. Upload PDFs to start ingestion.")
