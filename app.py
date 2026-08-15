"""
app.py
واجهة Streamlit لمساعد Fantasy Premier League الذكي.
تشغيل: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from fpl_data import build_players_dataframe, filter_candidates
from ai_advisor import get_ai_recommendation
from fixtures import build_fixture_difficulty_table, get_team_difficulty_summary_text

st.set_page_config(page_title="مساعد Fantasy Premier League", layout="wide")

st.title("⚽ مساعد Fantasy Premier League الذكي")
st.caption("تحليل بيانات حية من FPL API + توصيات مدعومة بالذكاء الاصطناعي")

# ---------- الشريط الجانبي: الإعدادات ----------
with st.sidebar:
    st.header("الإعدادات")
    api_key = st.text_input("OpenAI API Key", type="password")
    budget = st.number_input("الميزانية المتبقية (مليون)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
    position_filter = st.selectbox("المركز المطلوب تحليله", ["الكل", "GKP", "DEF", "MID", "FWD"])
    st.markdown("---")
    st.subheader("تشكيلتك الحالية")
    current_squad_input = st.text_area(
        "أدخل أسماء لاعبيك (اسم واحد بكل سطر)",
        placeholder="Salah\nHaaland\n...",
        height=150,
    )

# ---------- تحميل بيانات اللاعبين ----------
@st.cache_data(ttl=3600)  # تحديث كل ساعة، عشان ما نضرب الـ API بكثرة
def load_data():
    return build_players_dataframe()

with st.spinner("جاري تحميل بيانات اللاعبين من FPL..."):
    try:
        players_df = load_data()
    except Exception as e:
        st.error(f"تعذر جلب البيانات من FPL API: {e}")
        st.stop()

st.success(f"تم تحميل بيانات {len(players_df)} لاعب بنجاح.")

# ---------- عرض جدول اللاعبين ----------
st.subheader("📊 بيانات اللاعبين")
display_df = players_df if position_filter == "الكل" else players_df[players_df["position"] == position_filter]
st.dataframe(
    display_df[["full_name", "team", "position", "price_million", "total_points", "form", "chance_of_playing_next_round", "news"]],
    use_container_width=True,
    height=350,
)

st.markdown("---")

# ---------- تحليل صعوبة المباريات القادمة ----------
st.subheader("📅 صعوبة المباريات القادمة")

@st.cache_data(ttl=3600)
def load_fixture_difficulty(num_gw):
    return build_fixture_difficulty_table(num_gameweeks=num_gw)

num_gw = st.slider("عدد الجولات القادمة للتحليل", min_value=2, max_value=8, value=5)

try:
    fixture_df = load_fixture_difficulty(num_gw)
    st.dataframe(fixture_df, use_container_width=True, height=300)
    fixture_summary_text = get_team_difficulty_summary_text(num_gameweeks=num_gw)
except Exception as e:
    st.warning(f"تعذر تحميل جدول صعوبة المباريات: {e}")
    fixture_summary_text = None

st.markdown("---")

# ---------- طلب التوصية من الذكاء الاصطناعي ----------
st.subheader("🤖 اسأل المساعد")
user_question = st.text_area(
    "اكتب سؤالك (مثال: من أفضل مهاجم بسعر أقل من 8 مليون هذا الأسبوع؟ أو: اقترح تبديل لمدافعي)",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("احصل على التوصية", type="primary", use_container_width=True)

if ask_button:
    if not api_key:
        st.warning("الرجاء إدخال OpenAI API Key في الشريط الجانبي أولاً.")
    elif not user_question.strip():
        st.warning("الرجاء كتابة سؤال.")
    else:
        pos_arg = None if position_filter == "الكل" else position_filter
        candidates = filter_candidates(players_df, position=pos_arg, max_price=None, top_n=40)

        squad_list = [s.strip() for s in current_squad_input.splitlines() if s.strip()]

        with st.spinner("المساعد يحلل البيانات..."):
            try:
                answer = get_ai_recommendation(
                    api_key=api_key,
                    user_question=user_question,
                    candidates_df=candidates,
                    current_squad=squad_list,
                    budget_remaining=budget,
                    fixture_difficulty_text=fixture_summary_text,
                )
                st.markdown("### التوصية")
                st.write(answer)
            except Exception as e:
                st.error(f"حدث خطأ أثناء الاتصال بالنموذج: {e}")
