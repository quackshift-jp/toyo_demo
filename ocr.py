# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import google.generativeai as genai
import PyPDF2
import io
import json
import os
from dotenv import load_dotenv
import fitz
from PIL import Image
import io
import tempfile

# .envファイルの読み込み
load_dotenv()

# 設定
st.set_page_config(
    page_title="分析ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# スタイル設定
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container {
        margin-top: -2em;
    }
    .streamlit-expanderHeader {
        font-size: 1.2em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API-KEYの設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error(
        "APIキーが設定されていません。.envファイルにGEMINI_API_KEYを設定してください。"
    )
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# モデルの設定
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"response_mime_type": "application/json"},
)


def extract_text_from_pdf(pdf_file):
    """PDFからテキストを抽出"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"PDFの解析中にエラーが発生しました: {str(e)}")
        return None


def display_pdf_images(image_bytes):
    """PDFから抽出した画像を表示"""
    if image_bytes and len(image_bytes) > 0:  # 画像が存在することを確認
        st.divider()
        st.subheader("📸 分析対象の画像")

        # 画像の表示サイズ設定
        display_width = st.slider(
            "画像の表示サイズ",
            min_value=300,
            max_value=1200,
            value=800,
            step=100,
            key="image_size_slider",  # ユニークなキーを追加
        )

        for idx, img_bytes in enumerate(image_bytes):
            try:
                image = Image.open(io.BytesIO(img_bytes))
                st.image(image, caption=f"画像 {idx + 1}", width=display_width)

                with st.expander(f"画像 {idx + 1} の詳細情報"):
                    st.write(f"元のサイズ: {image.size[0]} x {image.size[1]}")
                    st.write(f"フォーマット: {image.format}")
                    st.write(f"モード: {image.mode}")
            except Exception as e:
                st.error(f"画像 {idx + 1} の表示中にエラーが発生しました: {str(e)}")
    else:
        st.info("画像が見つかりませんでした")


def display_analysis_images(image_bytes, title="📸 分析対象画像"):
    """分析に使用した画像を表示（image_infoの順序で表示）"""
    if image_bytes and len(image_bytes) > 0:
        st.divider()
        st.subheader(title)

        # 表示する画像の情報
        image_info = [
            {
                "page": 3,
                "number": 2,
                "index": 1,
            },  # indexは画像のインデックス（0から始まる）
            {"page": 1, "number": 1, "index": 0},
        ]

        # 2つのカラムを作成
        cols = st.columns(2)

        # image_infoの順序で画像を表示
        for idx, info in enumerate(image_info):
            with cols[idx]:
                try:
                    image = Image.open(io.BytesIO(image_bytes[info["index"]]))
                    st.image(
                        image,
                        caption=f"ページ: {info['page']}, 画像番号: {info['number']}",
                        use_column_width=True,
                    )
                except Exception as e:
                    st.error(f"画像の表示中にエラーが発生しました: {str(e)}")
    else:
        st.info("分析対象の画像が見つかりませんでした")


# PDFから画像を抽出する関数の修正
def extract_image_bytes(pdf_file) -> list[bytes]:
    """PDFから画像をバイト列のリストとして抽出"""
    extract_bytes = []
    try:
        # アップロードされたファイルを一時的に保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_file.getvalue())
            temp_path = temp_pdf.name

        # 保存したファイルを開く
        pdf_doc = fitz.open(temp_path)

        for page in pdf_doc:
            for image in page.get_images():
                xref = image[0]
                base_image = pdf_doc.extract_image(xref)
                if base_image:
                    extract_bytes.append(base_image["image"])

        # クリーンアップ
        pdf_doc.close()
        os.unlink(temp_path)
        return extract_bytes
    except Exception as e:
        st.error(f"画像の抽出中にエラーが発生しました: {str(e)}")
        return []


def analyze_with_gemini(text, analysis_type):
    """Geminiを使用して広告分析を実行"""
    prompts = {
        "visual_analysis": """
        以下の広告分析テキストから、視覚的な要素（レイアウト、注目ポイント、視線の流れなど）について分析してください。
        専門家の視点で分析し、具体的な数値やエビデンスを含めて説明してください。

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

        分析テキスト:
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

        分析テキスト:
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
        
        分析テキスト:
        """,
    }

    try:
        response = model.generate_content(prompts[analysis_type] + text)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Geminiの分析中にエラーが発生しました: {str(e)}")
        return None


def analyze_marketing_strategy(text):
    """マーケティング戦略の分析を実行"""
    prompt = """
    以下の広告分析テキストから、マーケティング戦略について包括的に分析してください。
    マーケティング4P、消費者行動分析、競合分析の観点から詳細な分析と実践的な示唆を提供してください。

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

    分析テキスト:
    """

    try:
        response = model.generate_content(prompt + text)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"マーケティング分析中にエラーが発生しました: {str(e)}")
        return None


def display_visual_analysis(analysis):
    """視覚分析結果の表示"""
    if not analysis:
        return

    # 効果スコアの表示
    cols = st.columns(4)
    with cols[0]:
        st.metric("全体的な効果スコア", f"{analysis['effectiveness_score']}/100")
    with cols[1]:
        st.metric("レイアウトスコア", f"{analysis['element_scores']['layout']}/100")
    with cols[2]:
        st.metric("階層性スコア", f"{analysis['element_scores']['hierarchy']}/100")
    with cols[3]:
        st.metric("視認性スコア", f"{analysis['element_scores']['visibility']}/100")

    # 視線の流れ
    st.subheader("👀 視線の流れ")
    flow_cols = st.columns(3)
    with flow_cols[0]:
        st.info(f"**最初の注目点**\n{analysis['attention_flow']['first_view']}")
    with flow_cols[1]:
        st.info(f"**次の注目点**\n{analysis['attention_flow']['second_view']}")
    with flow_cols[2]:
        st.info(f"**最後の注目点**\n{analysis['attention_flow']['final_view']}")

    # 主要ポイントと改善提案
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 主要な注目ポイント")
        for point in analysis["key_points"]:
            st.success(point)

    with col2:
        st.subheader("💡 改善提案")
        for rec in analysis["recommendations"]:
            st.warning(rec)


def display_color_analysis(analysis):
    """色彩分析結果の表示"""
    if not analysis:
        return

    st.subheader("🎨 配色分析")
    st.info(f"配色タイプ: {analysis['color_scheme']['type']}")
    st.progress(analysis["color_scheme"]["effectiveness"] / 100)
    st.write(analysis["color_scheme"]["harmony_description"])

    # 主要な色の表示
    st.subheader("使用されている主な色")
    cols = st.columns(len(analysis["dominant_colors"]))
    for i, color in enumerate(analysis["dominant_colors"]):
        with cols[i]:
            st.markdown(
                f"""
                <div style="background-color: {color['color']}; padding: 20px; border-radius: 5px;">
                    <h4 style="color: black;">{color['color']}</h4>
                    <p style="color: black;">{color['percentage']}%</p>
                    <p style="color: black;">{color['psychological_effect']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ターゲット層への影響
    st.subheader("👥 ターゲット層への影響")
    impact = analysis["target_audience_impact"]
    cols = st.columns(3)
    with cols[0]:
        st.write("**年齢層への効果**")
        for effect in impact["age_groups"]:
            st.write(f"- {effect}")
    with cols[1]:
        st.write("**性別による訴求力**")
        for appeal in impact["gender_appeal"]:
            st.write(f"- {appeal}")
    with cols[2]:
        st.write("**文化的な影響**")
        for factor in impact["cultural_factors"]:
            st.write(f"- {factor}")


def display_marketing_analysis(analysis):
    """マーケティング分析結果の表示"""
    if not analysis:
        return

    # 4P分析の表示
    st.subheader("🎯 マーケティング4P分析")
    tabs = st.tabs(["Product", "Price", "Place", "Promotion"])

    for tab, (p_type, data) in zip(tabs, analysis["marketing_4p"].items()):
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**現状分析**\n{data['current_status']}")
                st.markdown(
                    f"**市場での位置づけ**\n{data.get('competitive_position', data.get('market_positioning', ''))}"
                )
            with col2:
                st.markdown("**改善提案**")
                for suggestion in data["suggestions"]:
                    st.info(suggestion)

    # 消費者行動分析
    st.subheader("👥 消費者行動分析")

    # スコアの表示
    journey_data = []
    for stage, data in analysis["consumer_journey"].items():
        journey_data.append({"stage": stage.capitalize(), "score": data["score"]})

    journey_df = pd.DataFrame(journey_data)
    fig = px.bar(
        journey_df, x="stage", y="score", title="消費者行動スコア", range_y=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)

    # 競合分析
    st.subheader("🏢 競合分析")
    comp_analysis = analysis["competitive_analysis"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**市場での位置づけ**\n{comp_analysis['market_position']}")
        st.markdown("**独自の強み (USP)**")
        for usp in comp_analysis["unique_selling_points"]:
            st.success(f"✨ {usp}")

        with col2:
            st.metric("競合との差別化レベル", f"{comp_analysis['threat_level']}/100")
            st.markdown("**市場機会**")
            for opp in comp_analysis["opportunities"]:
                st.info(f"🎯 {opp}")

        # アクショナブルな示唆
        st.subheader("💡 実践的な示唆")

        # 優先順位でソート
        sorted_insights = sorted(
            analysis["actionable_insights"], key=lambda x: x["priority"], reverse=True
        )

        for insight in sorted_insights:
            with st.expander(f"優先度 {insight['priority']}/100: {insight['insight']}"):
                st.write(f"**期待される効果:** {insight['expected_impact']}")

        # 次のステップ
        st.subheader("📈 推奨アクション")

        for i, step in enumerate(analysis["next_steps"], 1):
            with st.expander(f"ステップ {i}: {step['action']}"):
                st.write(f"**実施時期:** {step['timeline']}")
                st.write(f"**期待される結果:** {step['expected_outcome']}")


# 総合評価表示用の新しい関数
def display_overall_impression(analysis):
    """総合評価の表示（既存のコードから抽出）"""
    if not analysis:
        return

    # 全体スコア
    st.metric(label="総合評価スコア", value=f"{analysis['overall_score']}/100")

    # ターゲット分析
    st.subheader("👥 ターゲットオーディエンス")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**主要ターゲット**")
        for target in analysis["target_audience"]["primary"]:
            st.write(f"- {target}")

    with col2:
        st.write("**副次ターゲット**")
        for target in analysis["target_audience"]["secondary"]:
            st.write(f"- {target}")

    # 強みと弱み
    st.subheader("💪 強みと課題")
    col1, col2 = st.columns(2)
    with col1:
        st.success("### 強み")
        for strength in analysis["strengths"]:
            st.write(f"✅ {strength}")

    with col2:
        st.warning("### 改善点")
        for weakness in analysis["weaknesses"]:
            st.write(f"📝 {weakness}")

    # 将来性
    st.subheader("🚀 将来性")
    for potential in analysis["future_potential"]:
        st.info(potential)


# 画像表示用の新しい関数
def display_pdf_images(image_bytes):
    """PDFから抽出した画像を表示"""
    if image_bytes:
        st.divider()
        st.subheader("📸 分析対象の画像")

        # 画像の表示サイズ設定
        display_width = st.slider(
            "画像の表示サイズ", min_value=300, max_value=1200, value=800, step=100
        )

        for idx, img_bytes in enumerate(image_bytes):
            try:
                image = Image.open(io.BytesIO(img_bytes))
                # アスペクト比を維持しながらリサイズ
                aspect_ratio = image.size[1] / image.size[0]
                display_height = int(display_width * aspect_ratio)

                # 画像をリサイズして表示
                st.image(image, caption=f"画像 {idx + 1}", width=display_width)

                # 画像の情報を表示
                with st.expander(f"画像 {idx + 1} の詳細情報"):
                    st.write(f"元のサイズ: {image.size[0]} x {image.size[1]}")
                    st.write(f"フォーマット: {image.format}")
                    st.write(f"モード: {image.mode}")
            except Exception as e:
                st.error(f"画像 {idx + 1} の表示中にエラーが発生しました: {str(e)}")


# 画像表示関数を分析タイプごとに分ける
def display_visual_analysis_image(image_bytes):
    """視覚分析用の画像表示"""
    if image_bytes and len(image_bytes) > 0:
        st.divider()
        st.subheader("👁️ 視覚分析に使用した画像")

        cols = st.columns(len(image_bytes))
        for idx, img_bytes in enumerate(image_bytes):
            with cols[idx]:
                try:
                    image = Image.open(io.BytesIO(img_bytes))
                    st.image(image, caption="視覚要素の分析対象", use_column_width=True)

                    with st.expander("画像詳細"):
                        st.write(f"サイズ: {image.size[0]} x {image.size[1]}")
                except Exception as e:
                    st.error(f"画像の表示中にエラーが発生しました: {str(e)}")


def display_color_analysis_image(image_bytes):
    """色彩分析用の画像表示"""
    if image_bytes and len(image_bytes) > 0:
        st.divider()
        st.subheader("🎨 色彩分析に使用した画像")

        # 画像の表示サイズ設定
        display_width = st.slider(
            "画像の表示サイズ",
            min_value=300,
            max_value=1200,
            value=800,
            step=100,
            key="color_analysis_slider",
        )

        for idx, img_bytes in enumerate(image_bytes):
            try:
                image = Image.open(io.BytesIO(img_bytes))
                st.image(image, caption="色彩分析の対象", width=display_width)

                # 色彩情報の表示
                with st.expander("色彩情報"):
                    st.write(f"カラーモード: {image.mode}")
                    if image.mode == "RGB":
                        st.write("RGB画像として分析")
                    elif image.mode == "CMYK":
                        st.write("CMYK画像として分析")
            except Exception as e:
                st.error(f"画像の表示中にエラーが発生しました: {str(e)}")


def display_overall_analysis_image(image_bytes):
    """総合評価用の画像表示"""
    if image_bytes and len(image_bytes) > 0:
        st.divider()
        st.subheader("📊 総合評価で分析した画像")

        for idx, img_bytes in enumerate(image_bytes):
            try:
                image = Image.open(io.BytesIO(img_bytes))
                st.image(image, caption="総合評価の対象", use_column_width=True)
            except Exception as e:
                st.error(f"画像の表示中にエラーが発生しました: {str(e)}")


def display_marketing_analysis_image(image_bytes):
    """マーケティング分析用の画像表示"""
    if image_bytes and len(image_bytes) > 0:
        st.divider()
        st.subheader("📈 マーケティング分析対象")

        cols = st.columns(len(image_bytes))
        for idx, img_bytes in enumerate(image_bytes):
            with cols[idx]:
                try:
                    image = Image.open(io.BytesIO(img_bytes))
                    st.image(
                        image, caption="マーケティング分析の対象", use_column_width=True
                    )
                except Exception as e:
                    st.error(f"画像の表示中にエラーが発生しました: {str(e)}")


def main():
    st.title("🤖 AI広告分析ダッシュボード")

    # サイドバー
    with st.sidebar:
        st.header("📊 分析設定")
        st.write("広告分析のための各種設定を行えます。")

        # ターゲット市場の選択
        target_market = st.multiselect(
            "ターゲット市場",
            ["一般消費者", "ビジネス", "若年層", "シニア層", "ファミリー"],
            default=["一般消費者"],
        )

        # 業界選択
        industry = st.selectbox(
            "業界", ["小売", "サービス", "製造", "テクノロジー", "金融", "その他"]
        )

        st.divider()

        # 分析の実行ボタン
        analyze_button = st.button("分析を実行", type="primary")

        # ヘルプ情報
        with st.expander("ヘルプ"):
            st.markdown(
                """
            ### 使い方
            1. PDFファイルをアップロード
            2. 分析設定を調整
            3. 「分析を実行」をクリック
            4. 各タブで結果を確認
            
            ### 注意事項
            - PDFは20MB以下推奨
            - テキストが抽出可能なPDFのみ対応
            - 分析には1-2分程度かかります
            """
            )

    # メインコンテンツ
    uploaded_file = st.file_uploader(
        "分析したい広告のPDFをアップロード",
        type=["pdf"],
        help="広告や販促物のPDFファイルをアップロードしてください",
    )

    if uploaded_file and analyze_button:
        with st.spinner("🔄 PDFを分析中..."):
            text = extract_text_from_pdf(uploaded_file)
            image_bytes = extract_image_bytes(uploaded_file)

            if text:
                # プログレスバーの表示
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 視覚分析
                status_text.text("視覚要素を分析中...")
                visual_analysis = analyze_with_gemini(text, "visual_analysis")
                progress_bar.progress(25)

                # 色彩分析
                status_text.text("色彩を分析中...")
                color_analysis = analyze_with_gemini(text, "color_analysis")
                progress_bar.progress(50)

                # 全体印象分析
                status_text.text("全体的な印象を分析中...")
                overall_impression = analyze_with_gemini(text, "overall_impression")
                progress_bar.progress(75)

                # マーケティング分析
                status_text.text("マーケティング戦略を分析中...")
                marketing_analysis = analyze_marketing_strategy(text)
                progress_bar.progress(100)

                # プログレス表示のクリア
                status_text.empty()
                progress_bar.empty()

                tab1, tab2, tab3, tab4 = st.tabs(
                    [
                        "📊 総合評価",
                        "👁️ 視覚分析",
                        "🎨 色彩分析",
                        "📈 マーケティング分析",
                    ]
                )

                with tab1:
                    st.header("総合評価")
                    if overall_impression:
                        display_analysis_images(image_bytes)  # 共通の画像表示
                        display_overall_impression(overall_impression)

                with tab2:
                    st.header("視覚要素の分析")
                    display_analysis_images(image_bytes)  # 共通の画像表示
                    display_visual_analysis(visual_analysis)

                with tab3:
                    st.header("色彩分析")
                    display_analysis_images(image_bytes)  # 共通の画像表示
                    display_color_analysis(color_analysis)

                with tab4:
                    st.header("マーケティング分析")
                    display_analysis_images(image_bytes)  # 共通の画像表示
                    display_marketing_analysis(marketing_analysis)

                # 分析レポートのダウンロード
                st.divider()
                st.subheader("📑 分析レポートのダウンロード")

                # 分析結果をJSON形式で保存
                analysis_results = {
                    "visual_analysis": visual_analysis,
                    "color_analysis": color_analysis,
                    "overall_impression": overall_impression,
                    "marketing_analysis": marketing_analysis,
                }

                json_str = json.dumps(analysis_results, ensure_ascii=False, indent=2)
                st.download_button(
                    label="JSON形式でダウンロード",
                    data=json_str,
                    file_name="analysis_report.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
