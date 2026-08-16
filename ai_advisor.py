"""
ai_advisor.py
الربط مع نموذج Google Gemini (مجاني عبر Google AI Studio) لتحليل بيانات اللاعبين
وتقديم توصيات التشكيلة والتبديلات.

يستخدم مكتبة google-genai الحديثة (وليست google-generativeai القديمة المتوقفة).
"""

import json
import pandas as pd
from google import genai
from google.genai import types


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

لديك إمكانية البحث في جوجل (Google Search) للحصول على معلومات حية. استخدمها تحديدًا من أجل:
- آخر أخبار الإصابات أو الشكوك الطبية للاعبين المذكورين في السؤال (أحدث من البيانات الرقمية المتوفرة لديك).
- توقعات وتحليلات مواقع متخصصة مثل Fantasy Football Scout أو FPL Statistics أو LiveFPL بخصوص لاعبين معينين أو الجولة القادمة.
- أي تشكيلة أساسية متوقعة (Predicted Lineup) قد تؤثر على فرصة لعب لاعب معين.
لا تستخدم البحث لأشياء موجودة أصلاً في بيانات JSON المرسلة إليك (كالسعر أو النقاط الكلية) — اعتمدي عليها مباشرة لتوفير الوقت.

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
    model: str = "gemini-flash-latest",
) -> str:
    """
    يرسل سؤال المستخدم + بيانات اللاعبين المرشحين + تشكيلته الحالية
    + تحليل صعوبة المباريات القادمة إلى Gemini ويرجع التوصية كنص.

    نستخدم "gemini-flash-latest" (وليس اسم إصدار محدد) عشان يبقى الكود
    شغّال تلقائيًا حتى لو غيّرت جوجل النموذج الأساسي مستقبلاً — هذا alias
    يشير دائمًا لأحدث نموذج Flash مستقر متاح.
    """
    client = genai.Client(api_key=api_key)

    context_parts = [f"بيانات اللاعبين المرشحين (JSON):\n{_players_to_json(candidates_df)}"]

    if fixture_difficulty_text:
        context_parts.append(f"\n{fixture_difficulty_text}")

    if current_squad:
        context_parts.append(f"\nالتشكيلة الحالية للمستخدم: {', '.join(current_squad)}")

    if budget_remaining is not None:
        context_parts.append(f"\nالميزانية المتبقية: {budget_remaining} مليون")

    context_parts.append(f"\nسؤال المستخدم: {user_question}")

    user_message = "\n".join(context_parts)

    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    answer_text = response.text

    # نستخرج قائمة المصادر الفعلية التي بحث فيها النموذج (شفافية للمستخدم)
    sources = []
    try:
        grounding = response.candidates[0].grounding_metadata
        if grounding and grounding.grounding_chunks:
            for chunk in grounding.grounding_chunks:
                if chunk.web:
                    sources.append(f"- [{chunk.web.title}]({chunk.web.uri})")
    except (AttributeError, IndexError):
        pass

    if sources:
        answer_text += "\n\n---\n**المصادر التي استند إليها البحث:**\n" + "\n".join(sources)

    return answer_text
