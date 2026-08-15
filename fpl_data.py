"""
fpl_data.py
جلب ومعالجة بيانات Fantasy Premier League من الـ API الرسمي (مجاني، بدون مفتاح).
"""

import requests
import pandas as pd

BASE_URL = "https://fantasy.premierleague.com/api/"


def get_bootstrap_data() -> dict:
    """
    يجلب البيانات الأساسية: كل اللاعبين، الفرق، وقواعد اللعبة (نقاط، تسعير، إلخ).
    هذا الـ endpoint الأهم — كل شيء تقريبًا يبدأ منه.
    """
    resp = requests.get(BASE_URL + "bootstrap-static/", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_fixtures() -> list:
    """جدول المباريات القادمة (مفيد لتحليل صعوبة المباريات القادمة لكل نادي)."""
    resp = requests.get(BASE_URL + "fixtures/", timeout=15)
    resp.raise_for_status()
    return resp.json()


def build_players_dataframe() -> pd.DataFrame:
    """
    يحوّل بيانات bootstrap-static إلى DataFrame نظيف وجاهز للتحليل،
    مع دمج اسم النادي واسم المركز بدل الأرقام (IDs) الخام.
    """
    data = get_bootstrap_data()

    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])[["id", "name", "short_name"]]
    positions = pd.DataFrame(data["element_types"])[["id", "singular_name", "singular_name_short"]]

    # دمج اسم النادي
    players = players.merge(
        teams, left_on="team", right_on="id", suffixes=("", "_team")
    )
    # دمج اسم المركز (حارس/مدافع/وسط/مهاجم)
    players = players.merge(
        positions, left_on="element_type", right_on="id", suffixes=("", "_pos")
    )

    players["price_million"] = players["now_cost"] / 10.0
    players["full_name"] = players["first_name"] + " " + players["second_name"]

    # الأعمدة المهمة فقط للتحليل والإرسال للنموذج اللغوي
    cols = [
        "full_name", "web_name", "name", "singular_name_short",
        "price_million", "total_points", "form", "points_per_game",
        "selected_by_percent", "ict_index", "minutes",
        "chance_of_playing_next_round", "news", "status",
    ]
    df = players[cols].rename(columns={
        "name": "team",
        "singular_name_short": "position",
        "web_name": "short_name",
    })

    # ترتيب حسب الأداء الحالي (form) كنقطة انطلاق منطقية
    df["form"] = pd.to_numeric(df["form"], errors="coerce")
    df = df.sort_values("form", ascending=False).reset_index(drop=True)
    return df


def filter_candidates(
    df: pd.DataFrame,
    position: str = None,
    max_price: float = None,
    exclude_injured: bool = True,
    top_n: int = 40,
) -> pd.DataFrame:
    """
    يفلتر قائمة اللاعبين المرشحين قبل إرسالها للنموذج اللغوي.
    هذا مهم جدًا: ما نرسل 600+ لاعب للـ AI، نرسل فقط أفضل عينة ذات صلة
    (يوفر تكلفة، ويقلل هلوسة النموذج، ويسرّع الاستجابة).
    """
    out = df.copy()

    if position:
        out = out[out["position"] == position]

    if max_price:
        out = out[out["price_million"] <= max_price]

    if exclude_injured:
        # نستبعد من هم مستبعدون تمامًا (status = 'u' unavailable) لكن نُبقي "مشكوك فيهم"
        # عشان الـ AI يشوفهم ويقرر بنفسه بناءً على chance_of_playing
        out = out[out["status"] != "u"]

    return out.sort_values(["form", "total_points"], ascending=False).head(top_n)


if __name__ == "__main__":
    # اختبار سريع من سطر الأوامر
    df = build_players_dataframe()
    print(df.head(15))
    print(f"\nعدد اللاعبين الكلي: {len(df)}")
