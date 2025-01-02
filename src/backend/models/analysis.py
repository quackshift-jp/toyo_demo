import os
import json
from typing import Literal
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# モデルの設定
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"response_mime_type": "application/json"},
)


def analyze_with_gemini(
    image_bytes: bytes,
    analysis_type: Literal[
        "visual_analysis",
        "color_analysis",
        "overall_impression",
    ],
) -> str:
    """Geminiを使用して広告分析を実行"""
    prompts = {
        "visual_analysis": """
        以下の視覚的な要素（レイアウト、注目ポイント、視線の流れなど）について分析してください。
        専門家の視点で分析し、具体的な数値やエビデンスを含めて説明してください。
        また、日本語で解説してほしい。

        JSON形式で返答してください：
        {
            "key_points": ["主要なポイント1", "主要なポイント2", ...],
            "attention_areas": ["注目エリア1", "注目エリア2", ...],
            "attention_flow": {
                "first_view": "最初に目が行く場所",
                "second_view": "次に目が行く場所",
                "final_view": "最後に目が行く場所"
            },
            "effectiveness_score": 0-100の数値,
            "element_scores": {
                "layout": 0-100の数値,
                "hierarchy": 0-100の数値,
                "visibility": 0-100の数値
            },
            "recommendations": ["改善提案1", "改善提案2", ...]
        }
        """,
        "color_analysis": """
        以下の広告分析テキストから、色使いについて詳細に分析してください。
        色彩心理学の観点から、各色の効果や印象も含めて説明してください。

        JSON形式で返答してください：
        {
            "dominant_colors": [
                {"color": "色名", "percentage": 数値, "psychological_effect": "心理的効果"},
                ...
            ],
            "color_scheme": {
                "type": "配色タイプ",
                "effectiveness": 0-100の数値,
                "harmony_description": "調和の説明"
            },
            "psychological_effects": ["効果1", "効果2", ...],
            "target_audience_impact": {
                "age_groups": ["対象年齢層への効果"],
                "gender_appeal": ["性別ごとの訴求力"],
                "cultural_factors": ["文化的な影響"]
            },
            "color_harmony_score": 0-100の数値,
            "suggestions": ["提案1", "提案2", ...]
        }
        """,
        "overall_impression": """
        以下の広告分析テキストから、全体的な印象を総合的に分析してください。
        マーケティング効果や消費者心理の観点から深く分析してください。

        JSON形式で返答してください：
        {
            "impressions": [
                {"aspect": "側面", "score": 0-100の数値, "description": "詳細説明"},
                ...
            ],
            "target_audience": {
                "primary": ["主要ターゲット1", "主要ターゲット2"],
                "secondary": ["副次ターゲット1", "副次ターゲット2"],
                "engagement_level": 0-100の数値
            },
            "strengths": ["強み1", "強み2", ...],
            "weaknesses": ["弱み1", "弱み2", ...],
            "market_fit": {
                "score": 0-100の数値,
                "reasons": ["理由1", "理由2"]
            },
            "overall_score": 0-100の数値,
            "future_potential": ["将来性1", "将来性2"]
        }
        """,
    }
    response = model.generate_content([prompts[analysis_type], image_bytes])
    return json.loads(response.text)


def analyze_marketing_strategy(image_bytes: bytes):
    """マーケティング戦略の分析を実行"""
    prompt = """
    以下のマーケティング戦略について包括的に分析してください。
    マーケティング4P、消費者行動分析、競合分析の観点から詳細な分析と実践的な示唆を提供してください。
    また、日本語で解説してほしい。

    JSON形式で返答してください：
    {
        "marketing_4p": {
            "product": {
                "current_status": "現状の分析",
                "competitive_position": "競合との比較",
                "suggestions": ["提案1", "提案2"]
            },
            "price": {
                "current_status": "現状の分析",
                "market_positioning": "市場での位置づけ",
                "suggestions": ["提案1", "提案2"]
            },
            "place": {
                "current_status": "現状の分析",
                "channel_effectiveness": "チャネルの有効性",
                "suggestions": ["提案1", "提案2"]
            },
            "promotion": {
                "current_status": "現状の分析",
                "communication_effectiveness": "コミュニケーション効果",
                "suggestions": ["提案1", "提案2"]
            }
        },
        "consumer_journey": {
            "awareness": {
                "score": 0-100の数値,
                "touchpoints": ["接点1", "接点2"],
                "insights": ["インサイト1", "インサイト2"]
            },
            "consideration": {
                "score": 0-100の数値,
                "decision_factors": ["要因1", "要因2"],
                "insights": ["インサイト1", "インサイト2"]
            },
            "purchase": {
                "score": 0-100の数値,
                "triggers": ["トリガー1", "トリガー2"],
                "insights": ["インサイト1", "インサイト2"]
            }
        },
        "competitive_analysis": {
            "market_position": "市場での位置づけ",
            "unique_selling_points": ["USP1", "USP2"],
            "threat_level": 0-100の数値,
            "opportunities": ["機会1", "機会2"]
        },
        "actionable_insights": [
            {
                "insight": "示唆1",
                "priority": 0-100の数値,
                "expected_impact": "期待される効果"
            },
            ...
        ],
        "next_steps": [
            {
                "action": "次のステップ1",
                "timeline": "実施時期",
                "expected_outcome": "期待される結果"
            },
            ...
        ]
    }
    """
    response = model.generate_content([prompt, image_bytes])
    return json.loads(response.text)
