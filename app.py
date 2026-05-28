import gc
import time
from typing import Dict, Tuple

import streamlit as st
import torch

# Workaround for Streamlit file watcher inspecting torch.classes.
torch.classes.__path__ = []

# Import directly from the auto-model modules to avoid
# a top-level transformers export issue in the deployed environment.
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Xiaomi AIoT Product Intelligence - Customer Reviews Analytics",
    page_icon="📱",
    layout="centered",
)


# =========================================================
# Fine-tuned model IDs uploaded to Hugging Face
# =========================================================
CATEGORY_MODEL_ID = "hanlongyang/amazon-category-distilbart-finetuned"
SENTIMENT_MODEL_ID = "hanlongyang/amazon-sentiment-bertweet-finetuned"
MAX_LENGTH = 128


# =========================================================
# Proposed internal routing map
# =========================================================
CATEGORY_TO_TEAM: Dict[str, str] = {
    "Electronics": "Electronics / Smart Devices Team",
    "Home": "Smart Home Appliances Team",
    "Home Entertainment": "Home Entertainment Team",
    "Office Products": "Office & Productivity Products Team",
    "PC": "PC & Computing Products Team",
    "Watches": "Wearables / Watch Products Team",
}


# =========================================================
# Model inference functions
# =========================================================
def predict_category(review_text: str) -> Tuple[str, float]:
    """
    Load the fine-tuned category model, predict one review,
    and then release the model to reduce memory usage.
    """

    tokenizer = AutoTokenizer.from_pretrained(CATEGORY_MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        CATEGORY_MODEL_ID,
        low_cpu_mem_usage=True,
    )

    model.eval()

    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_id = int(torch.argmax(probabilities, dim=-1).item())
        confidence = float(probabilities[0][predicted_id].item())

    category = model.config.id2label[predicted_id]

    del outputs
    del probabilities
    del inputs
    del model
    del tokenizer
    gc.collect()

    return category, confidence


def predict_sentiment(review_text: str) -> Tuple[str, float]:
    """
    Load the fine-tuned sentiment model, predict one review,
    and then release the model to reduce memory usage.
    """

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_ID,
        low_cpu_mem_usage=True,
    )

    model.eval()

    inputs = tokenizer(
        review_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_id = int(torch.argmax(probabilities, dim=-1).item())
        confidence = float(probabilities[0][predicted_id].item())

    sentiment = model.config.id2label[predicted_id]

    del outputs
    del probabilities
    del inputs
    del model
    del tokenizer
    gc.collect()

    return sentiment, confidence


def determine_action(category: str, sentiment: str) -> Tuple[str, str, str]:
    """
    Apply the customer service routing rule.
    """

    if sentiment == "Positive":
        return (
            "No",
            "No routing required",
            "Positive feedback detected. No immediate departmental follow-up is required.",
        )

    target_team = CATEGORY_TO_TEAM.get(
        category,
        "Relevant Product Team",
    )

    if sentiment == "Negative":
        return (
            "Yes",
            target_team,
            "Negative feedback detected. Please route this review for priority investigation.",
        )

    return (
        "Yes",
        target_team,
        "Neutral feedback detected. Please route this review for further review and monitoring.",
    )


@st.cache_data(show_spinner=False)
def process_review(review_text: str) -> Dict[str, object]:
    """
    Run the complete review-processing pipeline.
    Only the final lightweight result is cached, not the large models.
    """

    start_time = time.perf_counter()

    category, category_confidence = predict_category(review_text)
    sentiment, sentiment_confidence = predict_sentiment(review_text)

    action_required, route_to, decision_message = determine_action(
        category,
        sentiment,
    )

    runtime = time.perf_counter() - start_time

    return {
        "category": category,
        "category_confidence": category_confidence,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "action_required": action_required,
        "route_to": route_to,
        "decision_message": decision_message,
        "runtime": runtime,
    }


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Application Workflow")

    st.write("1. Enter one English customer review")
    st.write("2. Click **Process**")
    st.write("3. Predict product category")
    st.write("4. Predict customer sentiment")
    st.write("5. Route Neutral / Negative feedback")

    st.markdown("---")
    st.subheader("Decision Rule")

    st.write("🔴 Negative → Priority investigation")
    st.write("🟡 Neutral → Further review")
    st.write("🟢 Positive → No immediate follow-up")

    st.markdown("---")
    st.subheader("Product Routing Categories")

    for category, team in CATEGORY_TO_TEAM.items():
        st.markdown(f"**{category}**")
        st.caption(team)


# =========================================================
# Main page
# =========================================================
st.title("📱 Xiaomi AIoT Product Intelligence - Customer Reviews Analytics")

st.write(
    "A customer review triage tool for Xiaomi customer service staff. "
    "The application automatically identifies the related product category "
    "and customer sentiment, then determines whether the review should be "
    "routed to a proposed product team for follow-up."
)

st.info(
    "Prototype limitation: The current models were trained on labelled "
    "English product reviews. Please enter one English customer review."
)

st.markdown("### Enter Customer Review")


review_text = st.text_area(
    label="Customer review:",
    value="",
    height=160,
    placeholder="Paste one English customer review here...",
)

process_button = st.button(
    "Process",
    type="primary",
    use_container_width=True,
)


# =========================================================
# Process button action
# =========================================================
if process_button:
    cleaned_review = review_text.strip()

    if not cleaned_review:
        st.warning("Please enter a customer review before clicking Process.")

    else:
        try:
            with st.spinner(
                "Processing review. The first analysis may take longer while models are downloaded..."
            ):
                result = process_review(cleaned_review)

            st.markdown("---")
            st.markdown("## Processing Result")

            st.metric(
                label="Category",
                value=result["category"],
            )

            st.markdown("### Customer Service Decision")

            # Select banner colours according to sentiment result
            if result["sentiment"] == "Positive":
                banner_background = "#DFF3E6"
                banner_text_colour = "#137333"

            elif result["sentiment"] == "Neutral":
                banner_background = "#FFF5D6"
                banner_text_colour = "#8A5A00"

            else:
                banner_background = "#FDE2E1"
                banner_text_colour = "#B42318"

            # Display action decision with larger and bolder sentiment result
            st.markdown(
                f"""
                <div style="
                    background-color: {banner_background};
                    color: {banner_text_colour};
                    padding: 18px 20px;
                    border-radius: 8px;
                    margin-bottom: 18px;
                ">
                    <span style="
                        font-size: 16px;
                        font-weight: 500;
                    ">
                        Action Required:
                    </span>
                    <span style="
                        font-size: 24px;
                        font-weight: 700;
                        margin-left: 6px;
                    ">
                        {result['action_required']} (Sentiment: {result['sentiment']})
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result["sentiment"] == "Positive":
                st.write(result["decision_message"])

            elif result["sentiment"] == "Neutral":
                st.write(f"**Route To:** {result['route_to']}")
                st.write(result["decision_message"])

            else:
                st.write(f"**Route To:** {result['route_to']}")
                st.write(result["decision_message"])

        except Exception as error:
            st.error(
                "The review could not be processed. "
                "Please check the deployment logs and verify that both "
                "Hugging Face model repositories are publicly accessible."
            )
            st.exception(error)
