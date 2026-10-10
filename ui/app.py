from pathlib import Path
import csv
import logging
import sys
import tempfile

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.prescription_extractor import ExtractionProviderError
from src.recommendation.medicine_recommender import (
    TIER_POSSIBLE_EQUIVALENT,
    TIER_SAME_PRODUCT,
    TIER_WEAKER_MATCH,
    MedicineRecommendationEngine,
)
from src.rag.medical_information_answerer import (
    generate_medical_information_answer,
)
from src.rag.medical_information_retriever import retrieve_medical_information


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
        .status-candidate { color: var(--amber); font-weight: 700; }
        .status-uncertain { color: #7656a6; font-weight: 700; }
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


def medical_information_dataset_path():
    return Path(__file__).resolve().parents[1] / "data" / "final_medlineplus.csv"


def normalize_medical_information_name(value):
    return " ".join((value or "").casefold().split())


def _explicit_record_aliases(record):
    """Return aliases only when the dataset record explicitly supplies them."""

    aliases = record.get("aliases") or ""
    if isinstance(aliases, str):
        return [alias for alias in aliases.split("|") if alias.strip()]
    if isinstance(aliases, (list, tuple)):
        return [alias for alias in aliases if isinstance(alias, str)]
    return []


def search_medical_information(records, query, limit=10):
    """Search supplied records without inferring medicine-name equivalence."""

    normalized_query = normalize_medical_information_name(query)
    if not normalized_query:
        return {"match_type": "none", "records": [], "total_matches": 0}

    exact_matches = [
        record
        for record in records
        if normalize_medical_information_name(record.get("medicine_name"))
        == normalized_query
    ]
    if exact_matches:
        return {
            "match_type": "exact",
            "records": exact_matches[:limit],
            "total_matches": len(exact_matches),
        }

    alias_matches = [
        record
        for record in records
        if normalized_query
        in {
            normalize_medical_information_name(alias)
            for alias in _explicit_record_aliases(record)
        }
    ]
    if alias_matches:
        return {
            "match_type": "explicit_alias",
            "records": alias_matches[:limit],
            "total_matches": len(alias_matches),
        }

    fallback_matches = [
        record
        for record in records
        if normalized_query
        in normalize_medical_information_name(record.get("medicine_name"))
    ]
    return {
        "match_type": "substring" if fallback_matches else "none",
        "records": fallback_matches[:limit],
        "total_matches": len(fallback_matches),
    }


@st.cache_data
def load_medical_information(dataset_path=None):
    dataset_path = Path(dataset_path) if dataset_path else medical_information_dataset_path()
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
        return list(csv.DictReader(dataset_file))


def answer_medical_information_question(question, records):
    """Run the separate retrieval and local-answer stages for a submitted question."""

    retrieval_result = retrieve_medical_information(question, records)
    return retrieval_result, generate_medical_information_answer(question, retrieval_result)


@st.cache_resource
def get_recommendation_engine():
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "cleaned_medical_drug_master.csv"
    )
    return MedicineRecommendationEngine(str(dataset_path))


def run_analysis(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded_file.getbuffer())
            temporary_path = temporary_file.name

        from src.pipeline.prescription_pipeline import process_prescription

        return process_with_retry(
            process_prescription,
            temporary_path,
        )
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)


def process_with_retry(process_prescription, image_path):
    """Retry only explicitly transient extraction failures."""

    last_error = None

    for _ in range(3):
        try:
            result = process_prescription(image_path)
        except ExtractionProviderError as error:
            if not error.retryable:
                raise
            last_error = error
            continue

        if result is not None:
            return result

    if last_error is not None:
        raise last_error

    return None


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
    review_count = (
        len(result.get("candidate_medicines", []))
        + len(result.get("uncertain_medicines", []))
        + len(result["unverified_medicines"])
    ) if result else 0
    columns = st.columns(3)
    with columns[0]:
        card_metric("Extracted medicines", medicine_count)
    with columns[1]:
        card_metric("Exact matches", verified_count)
    with columns[2]:
        card_metric("Needs review", review_count)

    st.subheader("How the current pipeline works")
    st.markdown(
        """
        <div class="panel">
          <div class="step"><div class="step-number">1</div><div><div class="step-title">Prescription image</div><div class="step-copy">Select a local image for analysis.</div></div></div>
          <div class="step"><div class="step-number">2</div><div><div class="step-title">Vision extraction</div><div class="step-copy">Local Ollama runs qwen2.5vl:3b and returns structured prescription data.</div></div></div>
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
                except ExtractionProviderError as error:
                    st.session_state.analysis_result = None
                    st.error(error.message)
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
    render_header("Medicine Verification", "Dataset matching", "Review the dataset match status, matching evidence, and any details that need confirmation.")
    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Run a prescription analysis first to view verification results.")
        return

    groups = [
        ("Exact Matches", result["verified_medicines"], "status-ok", "Exact Match"),
        ("Candidate Matches", result.get("candidate_medicines", []), "status-candidate", "Candidate Match"),
        ("Uncertain Matches", result.get("uncertain_medicines", []), "status-uncertain", "Uncertain"),
        ("Unmatched Medicines", result["unverified_medicines"], "status-bad", "Unmatched"),
    ]

    for heading, medicines, status_class, status_label in groups:
        if not medicines:
            continue

        st.subheader(heading)
        for medicine in medicines:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown(f"### {medicine.get('extracted_name') or 'Unnamed medicine'}")
            columns = st.columns(4)
            with columns[0]:
                st.caption("Matched medicine")
                st.write(medicine.get("verified_name") or "Not available")
            with columns[1]:
                st.caption("Match type")
                st.write(medicine.get("match_type") or "Not available")
            with columns[2]:
                st.caption("Status")
                st.markdown(
                    f'<span class="{status_class}">{status_label}</span>',
                    unsafe_allow_html=True,
                )
            with columns[3]:
                st.caption("Candidates")
                st.write(medicine.get("candidate_count", 0))
            details = st.columns(3)
            with details[0]:
                st.caption("Strength check")
                st.write(medicine.get("strength_check") or "Not checked")
            with details[1]:
                st.caption("Dosage-form check")
                st.write(medicine.get("dosage_form_check") or "Not checked")
            with details[2]:
                st.caption("Release check")
                st.write(medicine.get("release_check") or "Not checked")
            st.caption(medicine.get("message") or medicine.get("reason") or "No explanation returned.")
            records = medicine.get("database_records")
            if records:
                with st.expander("View matched dataset records"):
                    st.json(records)
            st.markdown('</div>', unsafe_allow_html=True)

    st.caption("An exact match is an automated dataset match without detected conflicts. Candidate and uncertain results need human review.")


def render_medical_information():
    render_header("Medical Information", "Local knowledge dataset", "Look up medicine information from the bundled MedlinePlus dataset. This is a dataset search, not a generated medical answer.")
    query = st.text_input("Search medicine information", placeholder="Enter a medicine name, for example amoxicillin")
    search_clicked = st.button("Search information", type="primary")

    if not search_clicked and not query:
        st.info("Enter a medicine name to search the available medical information dataset.")
    elif not query.strip():
        st.warning("Enter a medicine name before searching.")
    else:
        try:
            lookup = search_medical_information(load_medical_information(), query)
        except (FileNotFoundError, OSError, UnicodeError, csv.Error):
            st.error(
                "Medical information data is unavailable. Check that "
                "data/final_medlineplus.csv is present and readable."
            )
        else:
            matches = lookup["records"]
            if not matches:
                st.warning("No matching medical information was found in the available dataset.")
            else:
                if lookup["match_type"] == "exact":
                    st.success(f"Found {lookup['total_matches']} exact matching record(s).")
                elif lookup["match_type"] == "explicit_alias":
                    st.info(
                        "Found an explicitly supported alias match. Confirm the medicine "
                        "name before relying on this information."
                    )
                else:
                    st.warning(
                        "Showing broader name matches only. Confirm the medicine name before "
                        "relying on this information."
                    )

                for record in matches:
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

                if lookup["total_matches"] > len(matches):
                    st.caption(
                        f"Showing the first {len(matches)} of {lookup['total_matches']} matching records."
                    )

    st.divider()
    st.subheader("Ask a dataset-grounded question")
    st.caption(
        "A local model summarizes only matching MedlinePlus dataset fields. "
        "It is not medically verified advice."
    )
    rag_question = st.text_input(
        "Question about a medicine",
        placeholder="For example: What side effects can abacavir cause?",
        key="rag_medical_information_question_input",
    )
    if st.button("Answer from local dataset evidence", key="rag_medical_information_submit"):
        if not rag_question.strip():
            st.warning("Enter a medicine-specific question before requesting an answer.")
        else:
            try:
                records = load_medical_information()
                retrieval_result, answer_result = answer_medical_information_question(
                    rag_question,
                    records,
                )
                st.session_state.rag_medical_information_submitted_question = rag_question
                st.session_state.rag_medical_information_retrieval = retrieval_result
                st.session_state.rag_medical_information_answer = answer_result
            except (FileNotFoundError, OSError, UnicodeError, csv.Error):
                st.error(
                    "Medical information data is unavailable. Check that "
                    "data/final_medlineplus.csv is present and readable."
                )
            except Exception:
                logger.exception("Medical information question processing failed")
                st.error("The local medical-information request could not be completed.")

    answer_result = st.session_state.get("rag_medical_information_answer")
    submitted_question = st.session_state.get(
        "rag_medical_information_submitted_question"
    )
    if not answer_result or not submitted_question:
        return

    st.caption(f"Submitted question: {submitted_question}")
    status = answer_result.get("status")
    if status == "answered":
        st.markdown("**Dataset-grounded answer**")
        st.write(answer_result.get("answer") or "No answer was returned.")
    elif status in {"ambiguous", "insufficient_evidence"}:
        st.info(answer_result.get("answer") or "The available dataset cannot answer this question.")
    else:
        st.error(answer_result.get("answer") or "Local medical-information generation failed.")
        if answer_result.get("error"):
            st.caption(answer_result["error"])

    sources = answer_result.get("sources") or []
    if sources:
        st.markdown("**Retrieved sources**")
        for source in sources:
            if isinstance(source, dict):
                st.caption(
                    f"{source.get('medicine_name') or 'Medicine'} — "
                    f"{source.get('source') or 'Dataset record'}"
                )


def render_recommendations():
    render_header(
        "Medicine Recommendations",
        "Dataset comparison",
        "Review possible equivalent product candidates using structured "
        "dataset attributes. These results are not substitution advice.",
    )
    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Run a prescription analysis before requesting recommendations.")
        return

    source_options = []
    for item in result.get("verified_medicines", []):
        if item.get("verification_status") != "exact_match":
            continue
        records = item.get("database_records") or []
        if len(records) != 1:
            continue
        record = records[0]
        label = (
            f"{item.get('extracted_name') or 'Unnamed medicine'} — "
            f"{record.get('drug_name') or 'Dataset product'}"
        )
        source_options.append((label, record))

    if not source_options:
        st.info(
            "Recommendations require an exact dataset match with one resolved "
            "source product. Candidate and uncertain matches are not eligible."
        )
        exact_count = sum(
            item.get("verification_status") == "exact_match"
            for item in result.get("verified_medicines", [])
        )
        st.caption(
            "Current analysis state: "
            f"{exact_count} exact match(es), {len(source_options)} resolved "
            "recommendation source(s). Run the prescription analysis again "
            "after verification changes to refresh this session's result."
        )
        return

    labels = [option[0] for option in source_options]
    selected_label = st.selectbox("Exact matched source product", labels)
    selected_record = next(
        record for label, record in source_options if label == selected_label
    )

    if st.button("Find possible equivalent products", type="primary"):
        try:
            st.session_state.recommendation_result = (
                get_recommendation_engine().recommend_equivalents(selected_record)
            )
            st.session_state.recommendation_source_setid = selected_record.get("SETID")
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Recommendation data could not be loaded: {error}")
            return

    recommendation = st.session_state.get("recommendation_result")
    if not recommendation:
        return
    if st.session_state.get("recommendation_source_setid") != selected_record.get("SETID"):
        st.info("Select Find possible equivalent products for the selected source.")
        return

    st.caption(recommendation["message"])
    if recommendation["recommendation_status"] != "ready":
        st.warning(recommendation["message"])
        return

    groups = [
        (
            "Same product records",
            TIER_SAME_PRODUCT,
            "These records identify the same normalized product.",
        ),
        (
            "Possible equivalent products",
            TIER_POSSIBLE_EQUIVALENT,
            "All available ingredient, strength, form, and release checks align.",
        ),
        (
            "Weaker matches requiring review",
            TIER_WEAKER_MATCH,
            "These are not substitutes. Review missing information and conflicts.",
        ),
    ]
    for heading, tier, description in groups:
        candidates = [
            candidate for candidate in recommendation["candidates"]
            if candidate["recommendation_tier"] == tier
        ]
        if not candidates:
            continue
        st.subheader(heading)
        st.caption(description)
        for candidate in candidates:
            record = candidate["record"]
            st.markdown(
                f"**{record.get('drug_name') or 'Unnamed dataset product'}**"
            )
            if candidate["match_reasons"]:
                st.write("Matches: " + " ".join(candidate["match_reasons"]))
            if candidate["missing_attributes"]:
                st.warning("Missing: " + " ".join(candidate["missing_attributes"]))
            if candidate["conflicts"]:
                st.warning("Conflicts: " + " ".join(candidate["conflicts"]))
            with st.expander("View dataset record"):
                st.json(record)


def main():
    inject_styles()
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "recommendation_result" not in st.session_state:
        st.session_state.recommendation_result = None
    if "recommendation_source_setid" not in st.session_state:
        st.session_state.recommendation_source_setid = None

    with st.sidebar:
        st.markdown("# MedicaAI")
        st.caption("Prescription intelligence workspace")
        page = st.radio("Navigate", ["Dashboard", "Prescription Analysis", "Medicine Verification", "Medicine Recommendations", "Medical Information"], label_visibility="collapsed")

    if page == "Dashboard":
        render_dashboard(st.session_state.analysis_result)
    elif page == "Prescription Analysis":
        render_analysis()
    elif page == "Medicine Verification":
        render_verification()
    elif page == "Medicine Recommendations":
        render_recommendations()
    elif page == "Medical Information":
        render_medical_information()


if __name__ == "__main__":
    main()
