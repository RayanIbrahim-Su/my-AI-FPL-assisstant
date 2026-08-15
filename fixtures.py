"""
fixtures.py
تحليل صعوبة المباريات القادمة لكل نادي، بالاعتماد على مؤشر FDR
(Fixture Difficulty Rating) الرسمي من FPL API (يتراوح من 1 = سهل جدًا إلى 5 = صعب جدًا).

الفكرة: لاعب فورمه جيد لكن ناديه القادم يواجه 3 مباريات صعبة متتالية
قد يكون خيارًا أسوأ من لاعب فورمه أقل لكن مبارياته القادمة سهلة.
"""

import requests
import pandas as pd

BASE_URL = "https://fantasy.premierleague.com/api/"


def get_fixtures_raw() -> list:
    """يجلب كل المباريات (الملعوبة والقادمة) من الـ API الرسمي."""
    resp = requests.get(BASE_URL + "fixtures/", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_teams_lookup() -> pd.DataFrame:
    """يجلب أسماء الأندية (نحتاجها لأن fixtures يرجع فقط team IDs)."""
    resp = requests.get(BASE_URL + "bootstrap-static/", timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data["teams"])[["id", "name", "short_name"]]


def build_fixture_difficulty_table(num_gameweeks: int = 5) -> pd.DataFrame:
    """
    يبني جدولاً لكل نادٍ يوضّح:
    - متوسط صعوبة أقرب N مباريات قادمة (avg_difficulty)
    - قائمة الخصوم القادمين مع رمز (H) لملعب أو (A) لخارج الملعب
    - تصنيف عام: سهلة / متوسطة / صعبة

    كلما كان avg_difficulty أقل، كانت المباريات القادمة أسهل (فرصة جيدة
    لضم لاعبين من هذا النادي أو الاحتفاظ بهم).
    """
    fixtures = get_fixtures_raw()
    teams = get_teams_lookup()
    team_id_to_name = dict(zip(teams["id"], teams["short_name"]))

    # نبقي فقط المباريات القادمة (غير الملعوبة)
    upcoming = [f for f in fixtures if not f.get("finished") and f.get("event") is not None]
    upcoming_df = pd.DataFrame(upcoming)

    rows = []
    for team_id, team_name in team_id_to_name.items():
        # مباريات هذا الفريق كمضيف أو كضيف
        home_matches = upcoming_df[upcoming_df["team_h"] == team_id].copy()
        home_matches["opponent"] = home_matches["team_a"].map(team_id_to_name)
        home_matches["difficulty"] = home_matches["team_h_difficulty"]
        home_matches["venue"] = "H"

        away_matches = upcoming_df[upcoming_df["team_a"] == team_id].copy()
        away_matches["opponent"] = away_matches["team_h"].map(team_id_to_name)
        away_matches["difficulty"] = away_matches["team_a_difficulty"]
        away_matches["venue"] = "A"

        team_matches = pd.concat([home_matches, away_matches])
        team_matches = team_matches.sort_values("event").head(num_gameweeks)

        if team_matches.empty:
            continue

        avg_difficulty = round(team_matches["difficulty"].mean(), 2)
        opponents_str = ", ".join(
            f"{row.opponent}({row.venue})" for row in team_matches.itertuples()
        )

        if avg_difficulty <= 2.4:
            label = "سهلة"
        elif avg_difficulty <= 3.4:
            label = "متوسطة"
        else:
            label = "صعبة"

        rows.append({
            "team": team_name,
            "avg_difficulty": avg_difficulty,
            "difficulty_label": label,
            "next_opponents": opponents_str,
            "num_fixtures_analyzed": len(team_matches),
        })

    result = pd.DataFrame(rows).sort_values("avg_difficulty").reset_index(drop=True)
    return result


def get_team_difficulty_summary_text(num_gameweeks: int = 5) -> str:
    """
    يبني ملخصًا نصيًا مختصرًا لجدول الصعوبة، جاهزًا للحقن مباشرة
    ضمن رسالة النموذج اللغوي (بدل إرسال JSON كامل).
    """
    df = build_fixture_difficulty_table(num_gameweeks=num_gameweeks)
    lines = [f"تحليل صعوبة أقرب {num_gameweeks} مباريات لكل نادٍ (1=سهل جدًا، 5=صعب جدًا):"]
    for row in df.itertuples():
        lines.append(
            f"- {row.team}: متوسط الصعوبة {row.avg_difficulty} ({row.difficulty_label}) "
            f"| الخصوم: {row.next_opponents}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    df = build_fixture_difficulty_table(num_gameweeks=5)
    print(df.to_string(index=False))
