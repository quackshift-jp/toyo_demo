import tempfile
import fitz
import os
from streamlit.runtime.uploaded_file_manager import UploadedFile


def extract_image_bytes(pdf_file: UploadedFile) -> list[bytes]:
    """PDFから画像をバイト列のリストとして抽出"""
    extract_bytes = []
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


# import streamlit as st
# from analysis import analyze_with_gemini
# import io
# from PIL import Image

# uploaded_file = st.file_uploader(
#     "分析したい広告のPDFをアップロード",
#     type=["pdf"],
#     help="広告や販促物のPDFファイルをアップロードしてください",
# )
# if uploaded_file:
#     with st.spinner("🔄 PDFを分析中..."):
#         image_bytes = extract_image_bytes(uploaded_file)
#         with Image.open(io.BytesIO(image_bytes[0])) as img:
#             print(analyze_with_gemini(img, "visual_analysis"))
