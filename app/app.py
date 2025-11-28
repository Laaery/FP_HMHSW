from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import os
from joblib import load
from plotly import express as px
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances

APP_DIR = Path(__file__).parent.resolve()
st.set_page_config(
    page_title="Heavy-Metal Hazardous Waste Source Predictor",
    page_icon="🧮",
    layout="wide"
)


@st.cache_resource
def load_models_and_data():
    model_paths = list(APP_DIR.glob("model_*.pkl"))
    model_files = [f.name for f in model_paths]

    if not model_files:
        st.error("❌ No model files found! Please include files like 'model_RandomForest.pkl'")
        return {}, None, [], None

    progress_bar = st.progress(0)
    status_text = st.empty()

    models = {}
    for i, (file_path, file_name) in enumerate(zip(model_paths, model_files)):
        name = file_name.replace('model_', '').replace('.pkl', '')
        try:
            model = load(file_path)
            status_text.text(f"Loading {file_name}...")
            time.sleep(0.1)

            if "xgb" in name.lower() or "xgboost" in name.lower():
                if hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    booster.set_param('device', 'cpu')
                    current_tree_method = booster.attr('tree_method')
                    if current_tree_method and 'gpu' in current_tree_method:
                        booster.set_param('tree_method', 'hist')
                    elif not current_tree_method:
                        booster.set_param('tree_method', 'hist')
                    try:
                        gpu_id = booster.attr('gpu_id')
                        if gpu_id is not None:
                            booster.set_attr(gpu_id=None)
                    except:
                        pass
            models[name] = model
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file_name}: {e}")
        progress_bar.progress((i + 1) / len(model_files))

    time.sleep(0.3)
    progress_bar.empty()
    status_text.text(f"✅ Loaded {len(models)} model(s).")

    try:
        label_encoder = load(APP_DIR / 'label_encoder.pkl')
    except Exception as e:
        st.error(f"❌ Failed to load label_encoder.pkl: {e}")
        label_encoder = None

    try:
        feature_names = load(APP_DIR / 'feature_names.pkl')
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        elif not isinstance(feature_names, (list, tuple)):
            feature_names = list(feature_names)
    except Exception as e:
        st.error(f"❌ Failed to load feature_names.pkl: {e}")
        feature_names = []

    phase_vector_df = None
    csv_path = APP_DIR / 'phase_vector_for_app.csv'
    try:
        if csv_path.exists():
            phase_vector_df = pd.read_csv(csv_path)
            required_cols = ['Source', 'Index']
            missing_cols = [col for col in required_cols if col not in phase_vector_df.columns]
            if missing_cols:
                st.warning(f"phase_vector_for_app.csv missing required columns: {missing_cols}")
                phase_vector_df = None
            else:
                if 'Description' not in phase_vector_df.columns:
                    phase_vector_df['Description'] = 'No description available'
        else:
            st.info("phase_vector_for_app.csv not found. Similarity matching will be disabled.")
    except Exception as e:
        st.warning(f"Failed to load phase_vector_for_app.csv: {e}")

    return models, label_encoder, feature_names, phase_vector_df


def find_top_k_matches_with_description(df, query_vector, target_sources, k=3):
    feature_cols = [col for col in df.columns if col not in ['Source', 'Index', 'Description']]
    if len(query_vector) != len(feature_cols):
        raise ValueError(f"Query vector length {len(query_vector)} doesn't match feature columns {len(feature_cols)}")

    query_array = np.array(query_vector).reshape(1, -1)

    results = {}

    for source in target_sources:
        source_data = df[df['Source'] == source]
        if len(source_data) == 0:
            results[source] = []
            continue

        source_features = source_data[feature_cols].values
        source_indices = source_data['Index'].tolist()
        source_descriptions = source_data['Description'].tolist()

        similarities = cosine_similarity(query_array, source_features)[0]

        index_desc_sim_triples = list(zip(source_indices, source_descriptions, similarities))
        index_desc_sim_triples.sort(key=lambda x: x[2], reverse=True)

        results[source] = index_desc_sim_triples[:k]

    return results


def create_similarity_visualization(df, query_vector, all_matches, user_vector_name="Your Input"):
    try:
        import plotly.graph_objects as go
        from sklearn.manifold import MDS

        feature_cols = [col for col in df.columns if col not in ['Source', 'Index', 'Description']]

        points_data = []
        labels = []
        colors = []
        sources = []

        points_data.append(query_vector)
        labels.append(user_vector_name)
        colors.append('red')
        sources.append('User Input')

        color_palette = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        source_color_map = {}

        for i, (source, matches) in enumerate(all_matches.items()):
            if source not in source_color_map:
                source_color_map[source] = color_palette[i % len(color_palette)]

            for match in matches:
                index_val, description, similarity = match
                match_row = df[(df['Source'] == source) & (df['Index'] == index_val)]
                if len(match_row) > 0:
                    match_features = match_row[feature_cols].iloc[0].values
                    points_data.append(match_features)
                    labels.append(f"{index_val}")
                    colors.append(source_color_map[source])
                    sources.append(source)

        if len(points_data) <= 1:
            return None

        points_array = np.array(points_data)

        distance_matrix = cosine_distances(points_array)  # shape: (n, n)


        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            random_state=42,
            max_iter=1500,
            eps=1e-9
        )
        points_2d = mds.fit_transform(distance_matrix)

        fig = go.Figure()

        unique_sources = list(set(sources))
        for source in unique_sources:
            source_indices = [i for i, s in enumerate(sources) if s == source]
            source_x = [points_2d[i, 0] for i in source_indices]
            source_y = [points_2d[i, 1] for i in source_indices]
            source_labels = [labels[i] for i in source_indices]

            if source == 'User Input':
                fig.add_trace(go.Scatter(
                    x=source_x,
                    y=source_y,
                    mode='markers+text',
                    marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='black')),
                    text=source_labels,
                    textposition="top center",
                    name=source,
                    hovertemplate='<b>%{text}</b><br>Source: User Input<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=source_x,
                    y=source_y,
                    mode='markers+text',
                    marker=dict(size=10, color=source_color_map[source]),
                    text=source_labels,
                    textposition="top center",
                    name=source,
                    hovertemplate='<b>%{text}</b><br>Source: ' + source + '<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ))

        fig.update_layout(
            title="Similarity Matching Visualization (MDS 2D)",
            xaxis_title="MDS Component 1",
            yaxis_title="MDS Component 2",
            width=600,
            height=600,
            margin=dict(l=40, r=40, t=60, b=60),
            legend=dict(
                x=0.99,
                y=0.99,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='Gray',
                borderwidth=1
            )
        )

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )

        return fig

    except Exception as e:
        st.warning(f"Visualization error: {str(e)}")
        return None


if 'selected_phases' not in st.session_state:
    st.session_state.selected_phases = []

# ======================
# SIDEBAR: Settings Panel
# ======================
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    models, label_encoder, feature_names_reduced, phase_vector_df = load_models_and_data()

    if not models or not label_encoder or not feature_names_reduced:
        st.error("Required files missing. App cannot run.")
        st.stop()

    model_names = list(models.keys())
    default_index = model_names.index('LogisticRegression') if 'LogisticRegression' in model_names else 0
    model_name = st.selectbox("Model", model_names, index=default_index)
    model = models[model_name]

    similarity_matching_available = phase_vector_df is not None
    k_matches = 3
    if similarity_matching_available:
        k_matches = st.slider("Top K matches", 1, 5, 3, help="Matches per source")
        st.info("Similarity matching: ✅ Enabled")
    else:
        st.warning("Similarity matching: ❌ Disabled")

    # Reuse logic to get full_feature_names
    if similarity_matching_available:
        full_feature_names = [col for col in phase_vector_df.columns if col not in ['Source', 'Index', 'Description']]
    else:
        full_feature_names = feature_names_reduced

# ======================
# MAIN: Header
# ======================
st.title("🔍 Trace Heavy-Metal Waste to Its Source")
st.markdown(
    "<p style='text-align: center; color: #666; margin-bottom: 2rem;'>"
    "Enter mineral phases to identify the source of heavy-metal hazardous solid waste"
    "</p>",
    unsafe_allow_html=True
)


# ======================
# SEARCH-LIKE INPUT WITH EMBEDDED ADD BUTTON
# ======================
st.markdown("#### Enter mineral phases from your sample")

input_col, add_col, clear_col, predict_col = st.columns([5, 1, 1, 1])

with input_col:
    phase_input = st.selectbox(
        "Mineral phase",
        options=full_feature_names,
        index=None,
        placeholder="🔍 Type to search phases...",
        key="phase_input_unique",
        label_visibility="collapsed"
    )

# Add button: only show if something is selected
with add_col:
    if phase_input:
        if st.button("➕", key="add_phase_btn", help="Add this phase", use_container_width=True):
            if phase_input not in st.session_state.selected_phases:
                st.session_state.selected_phases.append(phase_input)
                del st.session_state.phase_input_unique  # Clear selection
                st.rerun()
            else:
                st.toast(f"⚠️ '{phase_input}' already added", icon="ℹ️")
    else:
        # Show empty space to keep layout stable
        st.empty()

# Clear button
with clear_col:
    if st.button("🗑️", key="clear_all_btn", help="Clear all phases", use_container_width=True):
        st.session_state.selected_phases = []
        st.rerun()

with predict_col:
    if st.button("🔎 Predict", key="predict_btn", help="Predict source", use_container_width=True):
        predict_btn = True
    else:
        predict_btn = False


# Display selected phases as compact tags
if st.session_state.selected_phases:
    st.markdown("<br>", unsafe_allow_html=True)
    tags = " ".join([f"<span style='background:#e8f5e9; color:#1b5e20; padding:6px 10px; border-radius:6px; margin:4px;'>{p}</span>"
                     for p in st.session_state.selected_phases])
    st.markdown(f"<div>{tags}</div>", unsafe_allow_html=True)


# ======================
# PREDICTION LOGIC (triggered by Predict button)
# ======================
if predict_btn:
    if not st.session_state.selected_phases:
        st.warning("Please add at least one mineral phase.")
    else:
        # Build feature vector
        X_reduced = np.zeros((1, len(feature_names_reduced)))
        for phase in st.session_state.selected_phases:
            if phase in feature_names_reduced:
                idx = feature_names_reduced.index(phase)
                X_reduced[0, idx] = 1

        try:
            # Predict
            pred_class = model.predict(X_reduced)[0]
            pred_proba = model.predict_proba(X_reduced)[0]
            predicted_source = label_encoder.inverse_transform([pred_class])[0]
            confidence = pred_proba[pred_class]

            # ======================
            # IMPROVED RESULT DISPLAY
            # ======================
            st.markdown("### 🎯 Prediction Result")

            # Highlighted source badge
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #e0f7e7, #c8e6c9);
                    padding: 1.2rem;
                    border-radius: 12px;
                    text-align: center;
                    border: 2px solid #4caf50;
                    margin: 1rem 0;
                ">
                    <h3 style="color: #1b5e20; margin:0;">{predicted_source}</h3>
                    <p style="margin:0.5rem 0; color: #2e7d32;">
                        <b>Confidence:</b> {confidence:.2%}
                    </p>
                    <p style="margin:0; font-size:0.9em; color: #555;">
                        Model: <code>{model_name}</code>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Prepare probability data
            proba_data = []
            for i, cls in enumerate(label_encoder.classes_):
                source_name = label_encoder.inverse_transform([i])[0]
                proba_data.append({"Source": source_name, "Probability": pred_proba[i]})
            proba_df = pd.DataFrame(proba_data).sort_values(by="Probability", ascending=False)

            # Two-column layout for plots
            plot_col, viz_col = st.columns(2)

            with plot_col:
                with st.container(border=True):
                    st.markdown("#### 📊 Prediction Probabilities")
                    fig = px.bar(
                        proba_df,
                        x='Source',
                        y='Probability',
                        color='Probability',
                        color_continuous_scale='teal',
                        text=proba_df['Probability'].round(4)
                    )
                    fig.update_traces(textposition='outside')
                    fig.update_layout(
                        height=400,
                        margin=dict(t=40, b=120, l=40, r=20),
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with viz_col:
                with st.container(border=True):
                    if similarity_matching_available:
                        st.markdown("#### 🌐 Similar Samples (MDS)")
                        query_vector_full = [1 if f in st.session_state.selected_phases else 0 for f in
                                             full_feature_names]
                        top3_indices = np.argsort(pred_proba)[::-1][:3]
                        top3_sources = [label_encoder.inverse_transform([i])[0] for i in top3_indices]
                        available_sources = [s for s in top3_sources if s in phase_vector_df['Source'].unique()]

                        if available_sources:
                            similarity_results = find_top_k_matches_with_description(
                                phase_vector_df, query_vector_full, available_sources, k=k_matches
                            )
                            all_matches_for_viz = {s: similarity_results.get(s, []) for s in available_sources}
                            viz_fig = create_similarity_visualization(
                                phase_vector_df, query_vector_full, all_matches_for_viz, "Your Sample"
                            )
                            if viz_fig:
                                viz_fig.update_layout(height=400, margin=dict(t=40, b=40, l=20, r=20))
                                st.plotly_chart(viz_fig, use_container_width=True)
                            else:
                                st.info("MDS visualization unavailable.")
                        else:
                            st.info("No reference samples for top sources.")
                    else:
                        st.markdown("#### 🌐 Similarity Matching")
                        st.info("Reference data not available.")

            # Optional detailed table
            with st.expander("📋 Full Prediction Probabilities"):
                proba_df_display = proba_df.copy()
                proba_df_display["Probability"] = proba_df_display["Probability"].map("{:.4f}".format)
                st.dataframe(proba_df_display, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)
