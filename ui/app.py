from pathlib import Path
import csv
import logging
import sys
import tempfile

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="CLG | AI Prescription Assistant",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #183642;
            --muted: #60777f;
            --line: #d7e6e8;
            --surface: #ffffff;
            --wash: #f1f7f8;
            --teal: #087f82;
            --teal-dark: #075b67;
            --mint: #d9f1eb;
            --amber: #a36218;
            --rose: #b14c55;
        }

        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
        .stApp { background: linear-gradient(135deg, #f4f9fa 0%, #f8fbfc 54%, #eef7f4 100%); }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label,
        .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {
            color: var(--ink) !important;
        }
        .stApp [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
        .stApp [data-testid="stAlert"] p { color: inherit !important; }
        .stApp [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebar"] section { background: #123e4a !important; border-right: 0; }
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [role="radio"] p,
        [data-testid="stSidebar"] [role="radiogroup"] label { color: #f4fbfb !important; -webkit-text-fill-color: #f4fbfb !important; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { font-family: 'Space Grotesk', sans-serif; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #b9d8d8 !important; -webkit-text-fill-color: #b9d8d8 !important; }
        [data-testid="stSidebar"] [role="radiogroup"] label { border-radius: 8px; padding: .45rem .55rem; transition: background .2s ease; }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.1); }
        [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] { background: #1b6d75; }
        [data-testid="stSidebar"] [role="radiogroup"] div[role="radio"] { accent-color: #8fe0c6; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
        h1 { font-size: 2.4rem; line-height: 1.05; }
        h2 { font-size: 1.45rem; }
        .eyebrow { color: var(--teal); font-size: .76rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
        .hero { padding: 1.7rem 1.9rem; background: linear-gradient(115deg, #d9f1ec 0%, #f4fbfc 70%); border: 1px solid #c9e3e3; border-radius: 12px; margin-bottom: 1.25rem; box-shadow: 0 10px 28px rgba(18, 62, 74, .06); }
        .hero p { color: var(--muted); font-size: 1.03rem; max-width: 720px; margin-bottom: 0; }
        .metric { background: rgba(255,255,255,.88); border: 1px solid var(--line); border-radius: 10px; padding: 1.1rem; min-height: 112px; box-shadow: 0 6px 18px rgba(18, 62, 74, .04); }
        .metric-label { color: var(--muted); font-size: .79rem; text-transform: uppercase; letter-spacing: .08em; }
        .metric-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; margin-top: .4rem; }
        .panel { background: rgba(255,255,255,.92); border: 1px solid var(--line); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 6px 18px rgba(18, 62, 74, .04); }
        .step { display: flex; gap: .8rem; align-items: flex-start; padding: .85rem 0; border-bottom: 1px solid #edf2f1; }
        .step:last-child { border-bottom: 0; }
        .step-number { display: grid; place-items: center; flex: 0 0 28px; height: 28px; border-radius: 50%; background: var(--mint); color: var(--teal-dark); font-weight: 700; }
        .step-title { font-weight: 700; }
        .step-copy { color: var(--muted); font-size: .9rem; margin-top: .15rem; }
        .status-ok { color: var(--teal-dark); font-weight: 700; }
        .status-warn { color: var(--amber); font-weight: 700; }
        .status-bad { color: var(--rose); font-weight: 700; }
        .disclaimer { color: var(--muted); font-size: .78rem; border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 2rem; }
        [data-testid="stFileUploaderDropzone"] { background: #ffffff; border: 1px dashed #79aeb0; border-radius: 10px; }
        [data-testid="stFileUploaderDropzone"] p, [data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }
        [data-testid="stFileUploaderDropzone"] button { background: #e5f3f1; border: 1px solid #9bcac5; color: var(--teal-dark); }
        [data-testid="stFileUploaderDropzone"] button:hover { border-color: var(--teal); color: var(--teal-dark); }
        .stButton > button { border-radius: 7px; font-weight: 700; border-color: #b8d4d5; }
        .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); color: white; box-shadow: 0 5px 12px rgba(8, 127, 130, .2); }
        .stButton > button[kind="primary"]:hover { background: var(--teal-dark); border-color: var(--teal-dark); }
        [data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card_metric(label, value):
    st.markdown(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def prescription_dict(prescription):
    return prescription.model_dump() if hasattr(prescription, "model_dump") else prescription


@st.cache_data
def load_medical_information():
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "final_medlineplus.csv"
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        return list(csv.DictReader(dataset_file))


def run_analysis(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded_file.getbuffer())
            temporary_path = temporary_file.name

        from src.pipeline.prescription_pipeline import process_prescription

        result = None
        for attempt in range(3):
            result = process_prescription(temporary_path)
            if result is not None:
                break

        return result
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def render_header(title, eyebrow, description):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.write(description)


def render_dashboard(result):
    st.markdown(
        '<div class="hero"><div class="eyebrow">CLG medical AI</div>'
        '<h1>AI Prescription Assistant</h1>'
        '<p>Analyze prescriptions, verify medicines, and look up medical information from the project datasets.</p></div>',
        unsafe_allow_html=True,
    )

    medicine_count = len(result["prescription"].medicines) if result else 0
    verified_count = len(result["verified_medicines"]) if result else 0
    unverified_count = len(result["unverified_medicines"]) if result else 0
    columns = st.columns(3)
    with columns[0]:
        card_metric("Extracted medicines", medicine_count)
    with columns[1]:
        card_metric("Verified matches", verified_count)
    with columns[2]:
        card_metric("Needs review", unverified_count)

    st.subheader("How the current pipeline works")
    st.markdown(
        """
        <div class="panel">
          <div class="step"><div class="step-number">1</div><div><div class="step-title">Prescription image</div><div class="step-copy">Select a local image for analysis.</div></div></div>
          <div class="step"><div class="step-number">2</div><div><div class="step-title">Vision extraction</div><div class="step-copy">The existing Hugging Face VLM returns structured prescription data.</div></div></div>
          <div class="step"><div class="step-number">3</div><div><div class="step-title">Medicine verification</div><div class="step-copy">The existing MedicineVerifier checks extracted names against the local dataset.</div></div></div>
          <div class="step"><div class="step-number">4</div><div><div class="step-title">Medical information lookup</div><div class="step-copy">Search the bundled MedlinePlus dataset for uses, precautions, and side effects.</div></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not result:
        st.info("No analysis in this session yet. Open Prescription Analysis to begin.")


def render_analysis():
    render_header("Prescription Analysis", "Primary workflow", "Upload a prescription image and send it through the existing extraction and verification pipeline.")
    uploaded_file = st.file_uploader("Prescription image", type=["jpg", "jpeg", "png", "webp"], help="Use a clear image of one prescription.")

    if uploaded_file:
        left, right = st.columns([1, 1.2])
        with left:
            st.image(uploaded_file, caption=uploaded_file.name, width="stretch")
        with right:
            st.markdown('<div class="panel"><strong>Ready to analyze</strong><br><span style="color:#64777c">The image will be passed directly to the existing prescription pipeline.</span></div>', unsafe_allow_html=True)
            if st.button("Analyze prescription", type="primary", width="stretch"):
                try:
                    with st.spinner("Processing prescription..."):
                        result = run_analysis(uploaded_file)
                    if result is None:
                        st.session_state.analysis_result = None
                        st.warning("The vision model returned an incomplete response. Please try again or upload a clearer image.")
                    else:
                        st.session_state.analysis_result = result
                        st.success("Prescription processed successfully.")
                except (FileNotFoundError, ValueError):
                    st.error("The image could not be processed. Please choose a valid prescription image.")
                except Exception as error:
                    logger.exception("Prescription analysis failed: %s", error)
                    st.error("Something went wrong while processing the prescription. Please try again.")

    result = st.session_state.get("analysis_result")
    if result:
        render_extracted_prescription(result["prescription"])


def render_extracted_prescription(prescription):
    st.subheader("Extracted prescription")
    data = prescription_dict(prescription)
    details = [
        ("Patient", data.get("patient_name")),
        ("Date", data.get("prescription_date")),
        ("Doctor", data.get("doctor_name")),
        ("Clinic", data.get("clinic_name")),
        ("Global instruction", data.get("global_instruction")),
    ]
    columns = st.columns(3)
    for index, (label, value) in enumerate(details):
        with columns[index % 3]:
            st.markdown(f"**{label}**\n\n{value or 'Not extracted'}")

    medicine_rows = []
    for medicine in data.get("medicines", []):
        medicine_rows.append({
            "Medicine": medicine.get("medicine_name") or "Not extracted",
            "Strength": medicine.get("strength") or "-",
            "Frequency": medicine.get("frequency") or "-",
            "Quantity": medicine.get("quantity") or "-",
            "Duration": medicine.get("duration") or "-",
            "Instruction": medicine.get("instruction") or "-",
        })
    if medicine_rows:
        st.dataframe(medicine_rows, width="stretch", hide_index=True)
    else:
        st.info("No medicines were extracted from this prescription.")


def render_verification():
    render_header("Medicine Verification", "Dataset matching", "Review the actual matches returned by MedicineVerifier, including the matching strategy and dataset records.")
    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Run a prescription analysis first to view verification results.")
        return

    verified = result["verified_medicines"]
    unverified = result["unverified_medicines"]
    if verified:
        for medicine in verified:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f"### {medicine.get('extracted_name') or 'Unnamed medicine'}")
            columns = st.columns(3)
            with columns[0]:
                st.caption("Matched medicine")
                st.write(medicine.get("verified_name") or "Not available")
            with columns[1]:
                st.caption("Match type")
                st.write(medicine.get("match_type") or "Not available")
            with columns[2]:
                st.caption("Status")
                st.markdown('<span class="status-ok">Verified</span>', unsafe_allow_html=True)
            records = medicine.get("database_records")
            if records:
                with st.expander("View matched dataset record"):
                    st.json(records)
            st.markdown('</div>', unsafe_allow_html=True)

    if unverified:
        st.subheader("Needs review")
        for medicine in unverified:
            name = medicine.get("extracted_name") or "Unnamed medicine"
            reason = medicine.get("reason") or "No match returned."
            st.warning(f"{name}: {reason}")
        st.caption("Medicine could not be confidently matched with the available dataset. This does not indicate that it is unsafe.")


def render_medical_information():
    render_header("Medical Information", "Local knowledge dataset", "Look up medicine information from the bundled MedlinePlus dataset. This is a dataset search, not a generated medical answer.")
    query = st.text_input("Search medicine information", placeholder="Enter a medicine name, for example amoxicillin")
    search_clicked = st.button("Search information", type="primary")

    if not search_clicked and not query:
        st.info("Enter a medicine name to search the available medical information dataset.")
        return
    if not query.strip():
        st.warning("Enter a medicine name before searching.")
        return

    normalized_query = " ".join(query.lower().split())
    matches = [
        record for record in load_medical_information()
        if normalized_query in " ".join(record.get("medicine_name", "").lower().split())
    ]
    if not matches:
        st.warning("No matching medical information was found in the available dataset.")
        return

    st.success(f"Found {len(matches)} matching record(s).")
    for record in matches[:10]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader(record.get("medicine_name") or "Unnamed medicine")
        columns = st.columns(3)
        with columns[0]:
            st.markdown("**Uses**")
            st.write(record.get("uses") or "Not available")
        with columns[1]:
            st.markdown("**Precautions**")
            st.write(record.get("precautions") or "Not available")
        with columns[2]:
            st.markdown("**Side effects**")
            st.write(record.get("side_effects") or "Not available")
        st.caption(f"Source: {record.get('source') or 'Dataset record'}")
        st.markdown('</div>', unsafe_allow_html=True)

    if len(matches) > 10:
        st.caption(f"Showing the first 10 of {len(matches)} matching records.")


def main():
    inject_styles()
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    with st.sidebar:
        st.markdown("# MedicaAI")
        st.caption("Prescription intelligence workspace")
        page = st.radio("Navigate", ["Dashboard", "Prescription Analysis", "Medicine Verification", "Medical Information"], label_visibility="collapsed")

    if page == "Dashboard":
        render_dashboard(st.session_state.analysis_result)
    elif page == "Prescription Analysis":
        render_analysis()
    elif page == "Medicine Verification":
        render_verification()
    elif page == "Medical Information":
        render_medical_information()


if __name__ == "__main__":
    main()