"""
ai_advisor.py
الربط مع نموذج لغوي (GPT-4 عبر OpenAI API) لتحليل بيانات اللاعبين
وتقديم توصيات التشكيلة والتبديلات.
"""

import json
import pandas as pd
from openai import OpenAI


SYSTEM_PROMPT = """أنت محلل خبير في لعبة Fantasy Premier League (FPL).
قواعد اللعبة التي يجب الالتزام بها بدقة:
- الميزانية الكلية: 100 مليون جنيه إسترليني (افتراضيًا، إلا إذا ذُكر غير ذلك).
- التشكيلة تتكوّن من 15 لاعبًا: 2 حارس مرمى، 5 مدافعين، 5 لاعبي وسط، 3 مهاجمين.
- الحد الأقصى للاعبين من نفس النادي: 3 لاعبين.
- التشكيلة الأساسية (11 لاعب) يجب أن تحتوي على الأقل: حارس واحد، 3 مدافعين، مهاجم واحد.
- انتبه لحقل chance_of_playing_next_round: القيمة أقل من 75 تعني شكًا في المشاركة،
  و0 أو None مع status غير متاح تعني اللاعب مستبعد تقريبًا.
- افضّل اللاعبين ذوي form عالٍ ونقاط لكل مباراة (points_per_game) جيدة، مع مراعاة السعر (قيمة مقابل السعر).
- إذا توفّر لك تحليل صعوبة المباريات القادمة (Fixture Difficulty)، اعتبره عاملاً مهمًا:
  لاعب فورمه جيد لكن ناديه يواجه مباريات صعبة متتالية قد يكون خيارًا أضعف من لاعب
  فورمه أقل قليلاً لكن مبارياته القادمة سهلة. اذكر هذا في تبريرك عند توفر البيانات.

عند الرد:
- قدّم توصياتك بشكل منظم وواضح.
- اذكر السبب المختصر وراء كل اقتراح (الفورم، السعر، صعوبة المباريات، حالة الإصابة).
- إذا طُلب منك اقتراح تبديل (Transfer)، وازن بين تحسين النقاط المتوقعة وتكلفة التبديل.
- لا تخترع بيانات غير موجودة في القائمة المرسلة لك. استخدم فقط ما هو متوفر.
"""


def _players_to_json(df: pd.DataFrame) -> str:
    """يحوّل جدول اللاعبين المرشحين إلى JSON مختصر لتقليل حجم الطلب."""
    records = df.to_dict(orient="records")
    return json.dumps(records, ensure_ascii=False, default=str)


def get_ai_recommendation(
    api_key: str,
    user_question: str,
    candidates_df: pd.DataFrame,
    current_squad: list[str] = None,
    budget_remaining: float = None,
    fixture_difficulty_text: str = None,
    model: str = "gpt-4o",
) -> str:
    """
    يرسل سؤال المستخدم + بيانات اللاعبين المرشحين + تشكيلته الحالية
    + تحليل صعوبة المباريات القادمة إلى GPT-4 ويرجع التوصية كنص.
    """
    client = OpenAI(api_key=api_key)

    context_parts = [f"بيانات اللاعبين المرشحين (JSON):\n{_players_to_json(candidates_df)}"]

    if fixture_difficulty_text:
        context_parts.append(f"\n{fixture_difficulty_text}")

    if current_squad:
        context_parts.append(f"\nالتشكيلة الحالية للمستخدم: {', '.join(current_squad)}")

    if budget_remaining is not None:
        context_parts.append(f"\nالميزانية المتبقية: {budget_remaining} مليون")

    context_parts.append(f"\nسؤال المستخدم: {user_question}")

    user_message = "\n".join(context_parts)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content
