import streamlit as st
import numpy as np
import pandas as pd
import os
from joblib import load
from plotly import express as px
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances

st.set_page_config(
    page_title="Heavy-Metal Hazardous Waste Source Predictor",
    page_icon="🧮",
    layout="centered"
)


@st.cache_resource
def load_models_and_data():
    model_files = [f for f in os.listdir() if f.startswith('model_') and f.endswith('.pkl')]

    if not model_files:
        st.error("❌ No model files found! Please include files like 'model_RandomForest.pkl'")
        return {}, None, [], None

    progress_bar = st.progress(0)
    status_text = st.empty()

    models = {}
    for i, file in enumerate(model_files):
        name = file.replace('model_', '').replace('.pkl', '')
        try:
            model = load(file)
            status_text.text(f"Loading {file}...")
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
            st.warning(f"⚠️ Failed to load {file}: {e}")
        progress_bar.progress((i + 1) / len(model_files))

    time.sleep(0.3)
    progress_bar.empty()
    status_text.text(f"✅ Loaded {len(models)} model(s).")

    try:
        label_encoder = load('label_encoder.pkl')
    except Exception as e:
        st.error(f"❌ Failed to load label_encoder.pkl: {e}")
        label_encoder = None

    try:
        feature_names = load('feature_names.pkl')
        if isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()
        elif not isinstance(feature_names, (list, tuple)):
            feature_names = list(feature_names)
    except Exception as e:
        st.error(f"❌ Failed to load feature_names.pkl: {e}")
        feature_names = []

    phase_vector_df = None
    try:
        if os.path.exists('phase_vector_for_app.csv'):
            phase_vector_df = pd.read_csv('phase_vector_for_app.csv')
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

st.title("📊 Heavy-Metal Hazardous Waste Source Predictor")
st.markdown("""
Predict the **source** of heavy metal hazardous waste based on its **mineral phase fingerprint**.
Enter mineral phases below — suggestions will appear as you type.
""")

models, label_encoder, feature_names_reduced, phase_vector_df = load_models_and_data()

if not models:
    st.stop()
if not label_encoder:
    st.error("Label encoder is required. App cannot run.")
    st.stop()
if not feature_names_reduced:
    st.error("Feature names (phases) are missing. App cannot run.")
    st.stop()


similarity_matching_available = phase_vector_df is not None

if similarity_matching_available:
    full_feature_names = [col for col in phase_vector_df.columns if col not in ['Source', 'Index', 'Description']]
    missing_in_full = set(feature_names_reduced) - set(full_feature_names)
    if missing_in_full:
        st.warning(f"Some reduced features not found in full dataset: {missing_in_full}")
else:
    full_feature_names = feature_names_reduced


st.subheader("🔍 Enter Mineral Phases Fingerprint")

phase_input = st.selectbox(
    "Start typing to search for phases...",
    options=full_feature_names,
    index=None,
    placeholder="Type or select a mineral phase...",
    key="phase_input_unique"
)

if st.button("➕ Add Phase"):
    phase_value = st.session_state.phase_input_unique
    if phase_value:
        if phase_value not in st.session_state.selected_phases:
            st.session_state.selected_phases.append(phase_value)
            st.success(f"Added: **{phase_value}**")
            del st.session_state.phase_input_unique
            st.rerun()
        else:
            st.warning(f"⚠️ '{phase_value}' is already in the list.")
    else:
        st.warning("Please select a phase first.")


if st.session_state.selected_phases:
    st.write("**Selected Phases:**")
    cols = st.columns(min(len(st.session_state.selected_phases), 5))
    for idx, phase in enumerate(st.session_state.selected_phases):
        col_idx = idx % 5
        with cols[col_idx]:
            if st.button(f"❌ {phase}", key=f"remove_{idx}"):
                st.session_state.selected_phases.remove(phase)
                st.rerun()
else:
    st.info("No phases added yet. Please add at least one.")


st.markdown("---")
if st.button("🗑️ Clear All Phases"):
    st.session_state.selected_phases = []
    st.rerun()

st.subheader("🧠 Select Model")

model_names = list(models.keys())
default_index = 0
if 'LogisticRegression' in model_names:
    default_index = model_names.index('LogisticRegression')
elif 'logistic' in model_names:
    default_index = model_names.index('logistic')

model_name = st.selectbox(
    "Choose a trained model:",
    options=model_names,
    index=default_index
)
model = models[model_name]


if similarity_matching_available:
    st.markdown("---")
    st.subheader("🔍 Similarity Matching Settings")

    k_matches = st.slider(
        "Number of top similar Index matches to show per Source:",
        min_value=1,
        max_value=10,
        value=3,
        help="How many similar Index matches to display for each of the top 3 predicted Sources"
    )


if st.button("🚀 Predict Source", type="primary"):
    if len(st.session_state.selected_phases) == 0:
        st.warning("Please add at least one mineral phase.")
    else:

        X_reduced = np.zeros((1, len(feature_names_reduced)))
        for phase in st.session_state.selected_phases:
            if phase in feature_names_reduced:
                idx = feature_names_reduced.index(phase)
                X_reduced[0, idx] = 1


        try:
            pred_class = model.predict(X_reduced)[0]
            pred_proba = model.predict_proba(X_reduced)[0]
            predicted_source = label_encoder.inverse_transform([pred_class])[0]

            st.success(f"**Predicted Source: `{predicted_source}`** (using `{model_name.upper()}`)")

            proba_data = []
            for i, cls in enumerate(label_encoder.classes_):
                source_name = label_encoder.inverse_transform([i])[0]
                proba_data.append({"Source": source_name, "Probability": pred_proba[i]})

            proba_df = pd.DataFrame(proba_data).sort_values(by="Probability", ascending=False)

            proba_df_display = proba_df.copy()
            proba_df_display["Probability"] = proba_df_display["Probability"].map("{:.4f}".format)

            st.markdown("### 📊 Prediction Probabilities (Sorted by Confidence)")
            st.dataframe(proba_df_display, use_container_width=True)

            fig = px.bar(
                proba_df_display,
                x='Source',
                y='Probability',
                title="Prediction Probabilities",
                labels={'Probability': 'Probability', 'Source': 'Source'},
                color='Probability',
                color_continuous_scale='Sunset',
                text='Probability'
            )

            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.update_layout(
                uniformtext_minsize=8,
                uniformtext_mode='hide',
                xaxis_tickangle=-45,
                height=500,
                margin=dict(l=20, r=20, t=60, b=100)
            )
            st.plotly_chart(fig, use_container_width=True)

            if similarity_matching_available:
                st.markdown("---")
                st.subheader("🔍 Similar Index Matches")

                try:
                    query_vector_full = []
                    for feature in full_feature_names:
                        if feature in st.session_state.selected_phases:
                            query_vector_full.append(1)
                        else:
                            query_vector_full.append(0)


                    top3_indices = np.argsort(pred_proba)[::-1][:3]
                    top3_sources = [label_encoder.inverse_transform([i])[0] for i in top3_indices]

                    available_sources = []
                    for source in top3_sources:
                        if source in phase_vector_df['Source'].unique():
                            available_sources.append(source)
                        else:
                            st.info(
                                f"Source '{source}' not found in phase_vector_for_app.csv, skipping similarity matching for this source.")

                    if not available_sources:
                        st.warning(
                            "None of the top 3 predicted sources are available in the similarity matching dataset.")
                    else:
                        similarity_results = find_top_k_matches_with_description(
                            phase_vector_df,
                            query_vector_full,
                            available_sources,
                            k=k_matches
                        )

                        all_matches_for_viz = {}


                        total_matches = 0
                        for source in available_sources:
                            matches = similarity_results.get(source, [])
                            all_matches_for_viz[source] = matches
                            total_matches += len(matches)

                            if matches:
                                st.markdown(f"### 🔸 Source: **{source}**")
                                match_data = []
                                for idx, (index_val, description, similarity) in enumerate(matches):
                                    match_data.append({
                                        "Rank": idx + 1,
                                        "Index": index_val,
                                        "Description": description,
                                        "Similarity": f"{similarity:.4f}"
                                    })

                                match_df = pd.DataFrame(match_data)
                                st.dataframe(match_df, use_container_width=True, hide_index=True)
                            else:
                                st.info(f"No matches found for Source: {source}")

                        if total_matches > 0:
                            st.markdown("### 📈 Similarity Matching Visualization")
                            viz_fig = create_similarity_visualization(
                                phase_vector_df,
                                query_vector_full,
                                all_matches_for_viz,
                                user_vector_name="Your Input"
                            )

                            if viz_fig is not None:
                                st.plotly_chart(viz_fig, use_container_width=True)
                            else:
                                st.info("Visualization not available due to technical limitations.")

                except Exception as e:
                    st.error(f"Similarity matching error: {str(e)}")
                    st.exception(e)
            else:
                st.info("Similarity matching data (phase_vector_for_app.csv) not available.")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.exception(e)
