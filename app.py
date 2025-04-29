import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# ---------- Use Streamlit Secrets Instead of dotenv ----------
client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# ---------- Load Data ----------
df = pd.read_csv("Cover_Crop_Simulated_data_Transformed.csv")

goal_columns = [
    'Cashcrop compatibility', 'Erosion fighter', 'Good grazing',
    'Grain harvest', 'Lasting residue', 'Mechanical forage',
    'Nitrogen source', 'Quick growth', 'Soil builder', 'Weed fighter'
]

# Unique cash crops
cash_crops = set()
for entry in df['Target Cash Crops']:
    for crop in entry.split(','):
        cash_crops.add(crop.strip())
cash_crop_options = sorted(list(cash_crops))

# ---------- AI Recommendation Function ----------
def get_ai_summary(recommendation_table, user_goals, user_crops):
    table_text = recommendation_table.to_csv(index=False)
    prompt = f"""
You are a professional agricultural advisor.
A farmer is rotating the following crops: {', '.join(user_crops)} and has selected these farming goals: {', '.join(user_goals)}.
Based on the following cover crop data, recommend the best one or two options and explain why clearly using agronomic reasoning.

Cover crop data:
{table_text}

Respond in a helpful, professional tone starting with: Based on your farming goals, we recommend...
    """
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a smart farm advisor helping farmers choose the best cover crops using expert agronomic knowledge."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Cover Crop Selection", layout="centered")

# ---------- OSU Branding ----------
st.markdown("""
    <div style='background-color: #bb0000; padding: 20px;'>
        <h1 style='color: white; text-align: center;'>The Ohio State University</h1>
        <h3 style='color: white; text-align: center;'>Soil, Water & Bioenergy Research Team<br>OSU South Centers</h3>
    </div>
""", unsafe_allow_html=True)

# Display cover crop image without warning
st.image("Cover crop image.png", use_container_width=True)

st.markdown("""
    <div style='background-color: #f2f2f2; padding: 25px; border-radius: 10px; margin-top: 20px;'>
        <h2 style='text-align: center; color: #bb0000;'>Cover Crop Selection Guide</h2>
""", unsafe_allow_html=True)

selected_goals = st.multiselect("What are your farming goals?", goal_columns)
selected_cash_crops = st.multiselect("What cash crops do you want to rotate?", options=cash_crop_options)

if "ai_response" not in st.session_state:
    st.session_state.ai_response = ""

if st.button("Get Cover Crop Recommendations"):
    if not selected_goals or not selected_cash_crops:
        st.warning("Please select at least one goal and one or more cash crops.")
    else:
        filtered = df.copy()
        for goal in selected_goals:
            filtered = filtered[filtered[goal] == 'Yes']
        crop_pattern = '|'.join([crop.strip() for crop in selected_cash_crops])
        filtered = filtered[filtered['Target Cash Crops'].str.contains(crop_pattern, case=False)]

        if filtered.empty:
            st.error("No matching cover crops found.")
        else:
            st.success(f"Found {len(filtered)} matching cover crop(s).")
            display_cols = [
                'Cover Crop', 'Planting Months', 'TerminationMonths',
                'SeedCostPerAcre', 'TerminationCostPerAcre', 'Notes'
            ]
            st.dataframe(filtered[display_cols].reset_index(drop=True))

            st.markdown("""<h3 style='color: #bb0000;'>Smart Recommendation from AI Agronomist</h3>""", unsafe_allow_html=True)
            with st.spinner("Analyzing cover crops..."):
                try:
                    st.session_state.ai_response = get_ai_summary(
                        filtered[display_cols], selected_goals, selected_cash_crops)
                    st.write(st.session_state.ai_response)
                except Exception as e:
                    st.error(f"AI Error: {e}")

if st.session_state.ai_response:
    if st.button("🔊 Speak Recommendation Again"):
        st.components.v1.html(f"""
            <script>
                var msg = new SpeechSynthesisUtterance({st.session_state.ai_response!r});
                msg.lang = "en-US";
                msg.pitch = 1;
                msg.rate = 1;
                msg.volume = 1;
                var voices = window.speechSynthesis.getVoices();
                msg.voice = voices.find(v => v.name.includes('Female') || v.name.includes('Google')) || voices[0];
                window.speechSynthesis.speak(msg);
            </script>
        """, height=0)

st.markdown("</div>", unsafe_allow_html=True)

