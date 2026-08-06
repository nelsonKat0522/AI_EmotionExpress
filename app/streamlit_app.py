from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from emoexpress.emotion_classifier import (
    load_emotion_resources,
)
from emoexpress.topic_classifier import (
    load_topic_resources,
)
from emoexpress.rag import (
    load_vector_store,
)
from emoexpress.pipeline import (
    run_emoexpress_pipeline,
)


st.set_page_config(
    page_title="EmoExpress",
    page_icon="💙",
    layout="wide",
)


@st.cache_resource
def initialize_resources():
    """
    Load models and the vector store once for the
    Streamlit application session.
    """

    (
        emotion_tokenizer,
        emotion_model,
        device,
    ) = load_emotion_resources()

    (
        topic_tokenizer,
        topic_model,
    ) = load_topic_resources()

    vector_store = load_vector_store()

    return {
        "emotion_tokenizer": (
            emotion_tokenizer
        ),
        "emotion_model": emotion_model,
        "topic_tokenizer": (
            topic_tokenizer
        ),
        "topic_model": topic_model,
        "device": device,
        "vector_store": vector_store,
    }


st.title("💙 EmoExpress")

st.write(
    "Share what you are experiencing and receive "
    "emotion-aware support grounded in trusted resources."
)

st.warning(
    "EmoExpress provides general emotional support. "
    "It is not a replacement for professional medical, "
    "mental-health, legal, or emergency services."
)

try:
    with st.spinner(
        "Loading EmoExpress resources..."
    ):
        resources = initialize_resources()

except Exception as error:
    st.error(
        "EmoExpress could not load its models or "
        f"knowledge base: {error}"
    )
    st.stop()


user_story = st.text_area(
    "Tell EmoExpress what you are experiencing:",
    height=180,
    placeholder=(
        "For example: I have applied for several jobs, "
        "but I keep getting rejected and am starting "
        "to lose confidence."
    ),
)

left_column, right_column = st.columns(2)

with left_column:
    generate_image_output = st.checkbox(
        "Generate a supportive image",
        value=True,
    )

with right_column:
    retrieval_k = st.selectbox(
        "Number of knowledge passages",
        options=[3, 4, 5],
        index=2,
    )


run_button = st.button(
    "Express My Feelings",
    type="primary",
    use_container_width=True,
)


if run_button:
    if len(user_story.strip()) < 5:
        st.error(
            "Please enter a more detailed story."
        )

    else:
        try:
            with st.spinner(
                "Analyzing your story and preparing support..."
            ):
                result = run_emoexpress_pipeline(
                    user_story=user_story,
                    emotion_tokenizer=resources[
                        "emotion_tokenizer"
                    ],
                    emotion_model=resources[
                        "emotion_model"
                    ],
                    topic_tokenizer=resources[
                        "topic_tokenizer"
                    ],
                    topic_model=resources[
                        "topic_model"
                    ],
                    device=resources[
                        "device"
                    ],
                    vector_store=resources[
                        "vector_store"
                    ],
                    retrieval_k=retrieval_k,
                    generate_image_output=(
                        generate_image_output
                    ),
                )

            st.success(
                "Your EmoExpress result is ready."
            )

            # Emotion and topic overview
            emotion_column, topic_column = (
                st.columns(2)
            )

            with emotion_column:
                st.subheader(
                    "Detected Emotions"
                )

                emotion_dataframe = (
                    pd.DataFrame(
                        result[
                            "emotion_predictions"
                        ]
                    )
                )

                emotion_dataframe[
                    "score"
                ] = emotion_dataframe[
                    "score"
                ].map(
                    lambda value: (
                        f"{value:.2%}"
                    )
                )

                st.dataframe(
                    emotion_dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

            with topic_column:
                st.subheader(
                    "Predicted Topic"
                )

                topic_result = result[
                    "topic_result"
                ]

                topic_display = (
                    topic_result["topic"]
                    .replace("_", " ")
                    .title()
                )

                st.metric(
                    label="Knowledge-base topic",
                    value=topic_display,
                )

                st.caption(
                    "Confidence: "
                    f"{topic_result['confidence']:.2%}"
                )

            generated_response = result[
                "generated_response"
            ]

            # Main response
            st.subheader(
                "Supportive Response"
            )

            st.write(
                generated_response[
                    "empathetic_response"
                ]
            )

            st.subheader(
                "Practical Recommendations"
            )

            for index, recommendation in enumerate(
                generated_response[
                    "recommendations"
                ],
                start=1,
            ):
                st.markdown(
                    f"**{index}.** "
                    f"{recommendation['recommendation']}"
                )

                source_title = (
                    recommendation.get(
                        "source_title"
                    )
                )

                page = recommendation.get(
                    "page"
                )

                if source_title:
                    st.caption(
                        f"Source: {source_title}, "
                        f"page {page}"
                    )

            # Final caption
            st.subheader(
                "Encouraging Message"
            )

            st.success(
                generated_response[
                    "caption"
                ]
            )

            # Final image
            image_result = result[
                "generated_image"
            ]

            final_image_path = (
                image_result.get(
                    "final_path"
                )
                or image_result.get(
                    "path"
                )
            )

            if final_image_path:
                st.subheader(
                    "Your Personalized Image"
                )

                st.image(
                    final_image_path,
                    use_container_width=True,
                )

            elif generate_image_output:
                st.warning(
                    "The supportive text was created, "
                    "but image generation was unsuccessful."
                )

                if image_result.get(
                    "error"
                ):
                    with st.expander(
                        "Image error details"
                    ):
                        st.code(
                            image_result[
                                "error"
                            ]
                        )

            # Sources
            with st.expander(
                "Knowledge Sources"
            ):
                sources = result[
                    "retrieval"
                ]["sources"]

                if sources:
                    source_dataframe = (
                        pd.DataFrame(
                            sources
                        )
                    )

                    available_columns = [
                        column
                        for column in [
                            "rank",
                            "title",
                            "page",
                            "topic",
                            "relevance_score",
                        ]
                        if column
                        in source_dataframe.columns
                    ]

                    st.dataframe(
                        source_dataframe[
                            available_columns
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                else:
                    st.write(
                        "No directly relevant sources "
                        "were retrieved."
                    )

            # Technical details
            with st.expander(
                "Pipeline Details"
            ):
                st.write(
                    "Retrieval strategy:",
                    result[
                        "retrieval"
                    ]["strategy"],
                )

                timing_dataframe = (
                    pd.DataFrame(
                        [
                            result["timing"]
                        ]
                    )
                )

                st.dataframe(
                    timing_dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as error:
            st.error(
                "EmoExpress could not complete "
                f"the request: {error}"
            )