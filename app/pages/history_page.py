from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from emoexpress.history_manager import (
    clear_history,
    delete_history_record,
    load_history,
)


# =========================================================
# Helper functions
# =========================================================

def format_created_at(
    created_at: str,
) -> str:
    try:
        created_datetime = datetime.fromisoformat(
            created_at
        )

        return created_datetime.strftime(
            "%B %d, %Y at %I:%M %p"
        )

    except (
        TypeError,
        ValueError,
    ):
        return created_at or "Unknown date"


def format_created_at_short(
    created_at: str,
) -> str:
    try:
        created_datetime = datetime.fromisoformat(
            created_at
        )

        return created_datetime.strftime(
            "%b %d, %Y"
        )

    except (
        TypeError,
        ValueError,
    ):
        return created_at or "Unknown date"


def format_topic(
    topic: str,
) -> str:
    return (
        str(topic)
        .replace("_", " ")
        .title()
    )


def create_story_preview(
    story: str,
    maximum_length: int = 160,
) -> str:
    story = str(story).strip()

    if len(story) <= maximum_length:
        return story

    return (
        story[:maximum_length].rstrip()
        + "..."
    )


# =========================================================
# Load history
# =========================================================

history = load_history()


if "selected_history_id" not in st.session_state:
    st.session_state[
        "selected_history_id"
    ] = None


# =========================================================
# DETAIL VIEW
# =========================================================

selected_history_id = st.session_state[
    "selected_history_id"
]


if selected_history_id:

    selected_record = next(
        (
            record
            for record in history
            if record.get("id")
            == selected_history_id
        ),
        None,
    )

    if selected_record is None:

        st.session_state[
            "selected_history_id"
        ] = None

        st.rerun()


    # -----------------------------------------------------
    # Back button
    # -----------------------------------------------------

    if st.button(
        "← Back to History"
    ):
        st.session_state[
            "selected_history_id"
        ] = None

        st.rerun()


    topic = format_topic(
        selected_record.get(
            "topic",
            "general_support",
        )
    )

    created_at = format_created_at(
        selected_record.get(
            "created_at",
            "",
        )
    )


    # -----------------------------------------------------
    # Article heading
    # -----------------------------------------------------

    st.title(topic)

    st.caption(
        f"🕒 {created_at}"
    )


    # -----------------------------------------------------
    # Generated image
    # -----------------------------------------------------

    image_path = selected_record.get(
        "image_path"
    )

    if (
        image_path
        and Path(image_path).exists()
    ):

        image_column, spacer_column = (
            st.columns(
                [1.4, 1]
            )
        )

        with image_column:

            st.image(
                image_path,
                width="stretch",
            )


    # -----------------------------------------------------
    # Story
    # -----------------------------------------------------

    st.subheader(
        "Your Story"
    )

    st.write(
        selected_record.get(
            "user_story",
            "",
        )
    )


    st.divider()


    # -----------------------------------------------------
    # Emotion + topic
    # -----------------------------------------------------

    emotion_column, topic_column = (
        st.columns(2)
    )


    with emotion_column:

        st.subheader(
            "Detected Emotions"
        )

        emotion_predictions = (
            selected_record.get(
                "emotion_predictions",
                [],
            )
        )

        if emotion_predictions:

            emotion_dataframe = (
                pd.DataFrame(
                    emotion_predictions
                )
            )

            if (
                "score"
                in emotion_dataframe.columns
            ):

                emotion_dataframe[
                    "score"
                ] = emotion_dataframe[
                    "score"
                ].map(
                    lambda value:
                    f"{float(value):.2%}"
                )

            st.dataframe(
                emotion_dataframe,
                hide_index=True,
                width="stretch",
            )


    with topic_column:

        st.subheader(
            "Topic"
        )

        st.write(
            topic
        )

        confidence = (
            selected_record.get(
                "topic_confidence"
            )
        )

        if confidence is not None:

            st.caption(
                "Confidence: "
                f"{float(confidence):.2%}"
            )


    # -----------------------------------------------------
    # Supportive response
    # -----------------------------------------------------

    st.subheader(
        "Supportive Response"
    )

    st.write(
        selected_record.get(
            "empathetic_response",
            "",
        )
    )


    # -----------------------------------------------------
    # Recommendations
    # -----------------------------------------------------

    recommendations = (
        selected_record.get(
            "recommendations",
            [],
        )
    )

    if recommendations:

        st.subheader(
            "Recommendations"
        )

        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):

            st.markdown(
                f"**{index}.** "
                f"{recommendation.get('recommendation', '')}"
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

                source_text = (
                    f"Source: {source_title}"
                )

                if page is not None:

                    source_text += (
                        f", page {page}"
                    )

                st.caption(
                    source_text
                )


    # -----------------------------------------------------
    # Caption
    # -----------------------------------------------------

    caption = selected_record.get(
        "caption",
        "",
    )

    if caption:

        st.subheader(
            "Encouraging Message"
        )

        st.success(
            caption
        )


    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    with st.expander(
        "Knowledge Sources"
    ):

        sources = selected_record.get(
            "sources",
            [],
        )

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
                hide_index=True,
                width="stretch",
            )

        else:

            st.write(
                "No saved source information."
            )


    # -----------------------------------------------------
    # Delete
    # -----------------------------------------------------

    st.divider()

    delete_confirmation = st.checkbox(
        "Confirm deletion of this story",
        key=(
            "confirm_delete_"
            + selected_history_id
        ),
    )

    if st.button(
        "Delete Story",
        disabled=(
            not delete_confirmation
        ),
    ):

        deleted = delete_history_record(
            selected_history_id
        )

        if deleted:

            st.session_state[
                "selected_history_id"
            ] = None

            st.success(
                "Story deleted."
            )

            st.rerun()


# =========================================================
# NEWSPAPER / BLOG LIST VIEW
# =========================================================

else:

    st.title(
        "🕘 History"
    )

    st.write(
        "Browse your previous EmoExpress stories."
    )


    # -----------------------------------------------------
    # Empty history
    # -----------------------------------------------------

    if not history:

        st.info(
            "No stories have been generated yet."
        )

        st.page_link(
            "pages/generate_page.py",
            label="Generate your first story",
            icon="✨",
            width="stretch",
        )

        st.stop()


    # -----------------------------------------------------
    # Topic filter
    # -----------------------------------------------------

    topic_options = sorted(
        {
            str(
                record.get(
                    "topic",
                    "general_support",
                )
            )
            for record in history
        }
    )


    selected_topic = st.selectbox(
        "Filter by topic",
        options=[
            "All topics",
            *topic_options,
        ],
        format_func=lambda topic: (
            "All topics"
            if topic == "All topics"
            else format_topic(topic)
        ),
    )


    # -----------------------------------------------------
    # Clear history
    # -----------------------------------------------------

    clear_confirmation = st.checkbox(
        "Confirm clear all history"
    )

    if st.button(
        "Clear All History",
        disabled=(
            not clear_confirmation
        ),
    ):

        clear_history()

        st.success(
            "History was cleared."
        )

        st.rerun()


    st.divider()


    # -----------------------------------------------------
    # Apply filter
    # -----------------------------------------------------

    if selected_topic == "All topics":

        filtered_history = history

    else:

        filtered_history = [
            record
            for record in history
            if record.get(
                "topic",
                "general_support",
            )
            == selected_topic
        ]


    st.caption(
        f"{len(filtered_history)} saved "
        f"{'story' if len(filtered_history) == 1 else 'stories'}"
    )


    if not filtered_history:

        st.info(
            "No saved stories match this topic."
        )

        st.stop()


    # =====================================================
    # Newspaper-style posts
    # =====================================================

    for record in filtered_history:

        record_id = record.get(
            "id",
            ""
        )

        topic = format_topic(
            record.get(
                "topic",
                "general_support",
            )
        )

        created_at = (
            format_created_at_short(
                record.get(
                    "created_at",
                    "",
                )
            )
        )

        story = record.get(
            "user_story",
            "",
        )

        preview = create_story_preview(
            story
        )

        image_path = record.get(
            "image_path"
        )


        # -------------------------------------------------
        # One newspaper post
        # -------------------------------------------------

        with st.container(
            border=True
        ):

            image_column, content_column = (
                st.columns(
                    [1, 2.7],
                    vertical_alignment="center",
                )
            )


            # ---------------------------------------------
            # Thumbnail
            # ---------------------------------------------

            with image_column:

                if (
                    image_path
                    and Path(
                        image_path
                    ).exists()
                ):

                    st.image(
                        image_path,
                        width=180,
                    )

                else:

                    st.markdown(
                        "🖼️ *No image*"
                    )


            # ---------------------------------------------
            # Article summary
            # ---------------------------------------------

            with content_column:

                st.caption(
                    f"{created_at}"
                )

                st.subheader(
                    topic
                )

                st.write(
                    preview
                )


                emotion_predictions = (
                    record.get(
                        "emotion_predictions",
                        [],
                    )
                )

                if emotion_predictions:

                    emotion_names = [
                        emotion.get(
                            "emotion",
                            "",
                        )
                        for emotion
                        in emotion_predictions
                    ]

                    st.caption(
                        "Emotions: "
                        + " · ".join(
                            emotion_names
                        )
                    )


                if st.button(
                    "Read Story →",
                    key=(
                        "read_story_"
                        + record_id
                    ),
                ):

                    st.session_state[
                        "selected_history_id"
                    ] = record_id

                    st.rerun()