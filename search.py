import re
import pandas as pd
vocab = pd.read_excel(r"码表.xlsx")
import streamlit as st


def answer_format(source):
    # 定义包含目标文本的字符串
    text = str(source)

    # 内容项
    start_marker = "page_content="
    end_marker = ", metadata="

    content_pattern = re.compile(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker))
    content_matches = content_pattern.finditer(text)

    content_extracted_texts = []

    for match in content_matches:
        extracted_text = match.group(1)
        content_extracted_texts.append(extracted_text)

    # 来源公司项
    source_start_maker = ''', metadata={'source': 'D:\\PycharmFile\\test\\save_files\\'''
    source_end_maker = ".txt'})"

    source_pattern = re.compile(re.escape(source_start_maker) + r"(.*?)" + re.escape(source_end_maker))
    source_matches = source_pattern.finditer(text)

    source_extracted_texts = []

    for match in source_matches:
        extracted_text = match.group(1)
        extracted_text = extracted_text[1:]
        source_extracted_texts.append(extracted_text)

    # 打印

    for i in range(len(content_extracted_texts)):

            f"\033[1m来源{i + 1}\033[0m", "\n\n",
            content_extracted_texts[i][1:-1].replace("\\n", '\n'), "\n\n",
            f"\033[1mSource:{vocab.loc[vocab['缩写'] == source_extracted_texts[i], '公司名称'].values[0]}\033[0m","\n\n"