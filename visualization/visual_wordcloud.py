import os
import argparse
import json
import glob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba  # 用于中文分词
import pandas as pd

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置路径常量
ROOTPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATAPATH = os.path.join(ROOTPATH, 'data', 'CED_Dataset')
RUMORPATH = os.path.join(DATAPATH, 'rumor-repost')
NONRUMORPATH = os.path.join(DATAPATH, 'non-rumor-repost')
ORIGINPATH = os.path.join(DATAPATH, 'original-microblog')

def generate_wordcloud(text, title, is_rumor=True):
    """
    生成词云图

    Args:
        text (str): 要生成词云的文本
        title (str): 词云图的标题
        is_rumor (bool): 是否为谣言词云图，用于设置不同的颜色
    """
    # 设置词云参数
    wordcloud = WordCloud(
        font_path='simhei.ttf',  # 中文字体路径，需要确保系统中有该字体
        background_color='white',
        max_words=75,
        max_font_size=100,
        width=400,
        height=300,
        random_state=42,
        collocations=False,  # 不显示词语搭配，避免重复
    )

    # 生成词云
    wordcloud.generate(text)

    # 显示词云
    plt.figure(figsize=(6, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.show()

def collect_all_text(is_rumor=True):
    """
    收集所有谣言或非谣言的文本

    Args:
        is_rumor (bool): 是否收集谣言文本

    Returns:
        str: 所有文本的拼接
    """
    # 确定文件路径
    repost_path = RUMORPATH if is_rumor else NONRUMORPATH
    origin_path = ORIGINPATH

    all_text = []

    # 收集原始微博文本
    print(f"正在收集{'谣言' if is_rumor else '非谣言'}原始微博文本...")
    repost_files = glob.glob(os.path.join(repost_path, '*.json'))

    for repost_file in repost_files:
        # 获取对应的原始微博文件名
        filename = os.path.basename(repost_file)
        origin_file = os.path.join(origin_path, filename)

        if os.path.exists(origin_file):
            with open(origin_file, 'r', encoding='utf-8') as f:
                origin_data = json.load(f)
                if 'text' in origin_data:
                    all_text.append(origin_data['text'])

    # 收集转发微博文本
    print(f"正在收集{'谣言' if is_rumor else '非谣言'}转发微博文本...")

    for repost_file in repost_files:
        with open(repost_file, 'r', encoding='utf-8') as f:
            repost_data = json.load(f)
            df = pd.DataFrame(repost_data)

            # 提取所有非空文本
            texts = df['text'].dropna().tolist()
            all_text.extend(texts)

    # 拼接所有文本
    combined_text = ' '.join(str(text) for text in all_text if text)

    # 中文停用词列表（精简版）
    CHINESE_STOPWORDS = {
        '的', '了', '和', '是', '在', '我', '有', '不是', '人', '都', '一', 
        '你', '这', '就', '不', '也', '还', '啊', '吗', '好', '要', '说',
        '吧', '去', '才', '么', '他', '又', '会', '呢', '就是', '可以',
        '看', '能', '来', '过', '让', '再', '那', '被', '为', '哈哈',
        '给', '啦', '什么', '转发', '微博', '@', '//', 'http', 'https',
        '这个', '那个', '我们', '你们', '他们', '她们', '它的', '自己',
        '今天', '明天', '昨天', '现在', '时候', '时间', '事情', '东西'
    }

    # 使用jieba进行中文分词
    print("正在进行中文分词...")
    # words = jieba.lcut(combined_text)
    words = [word for word in jieba.lcut(combined_text) 
             if word.strip() and word not in CHINESE_STOPWORDS]
    

    # 调试：检查重复词语
    from collections import Counter
    word_counts = Counter(words)
    # 只显示出现次数大于1的词语
    duplicate_words = {word: count for word, count in word_counts.items() if count > 1}
    if duplicate_words:
        print(f"检测到重复词语 {len(duplicate_words)} 个，前10个高频重复词：")
        for word, count in sorted(duplicate_words.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  '{word}' 出现 {count} 次")

    segmented_text = ' '.join(words)

    print(f"共收集到{'谣言' if is_rumor else '非谣言'}文本 {len(all_text)} 条")
    print(f"分词后文本长度: {len(segmented_text)} 字符")
    print(f"分词后词语总数: {len(words)} 个")
    print(f"分词后不同词语数: {len(word_counts)} 个")

    return segmented_text

def visualize_wordclouds(rumor_only=False, nonrumor_only=False):
    """
    可视化词云图

    Args:
        rumor_only (bool): 是否只生成谣言词云图
        nonrumor_only (bool): 是否只生成非谣言词云图
    """
    if rumor_only or not nonrumor_only:
        # 生成谣言词云图
        rumor_text = collect_all_text(is_rumor=True)
        generate_wordcloud(rumor_text, '谣言文本词云图', is_rumor=True)

    if nonrumor_only or not rumor_only:
        # 生成非谣言词云图
        nonrumor_text = collect_all_text(is_rumor=False)
        generate_wordcloud(nonrumor_text, '非谣言文本词云图', is_rumor=False)

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='生成谣言和非谣言的词云图')

    parser.add_argument('--rumor', action='store_true', help='只生成谣言词云图')
    parser.add_argument('--nonrumor', action='store_true', help='只生成非谣言词云图')

    args = parser.parse_args()

    visualize_wordclouds(rumor_only=args.rumor, nonrumor_only=args.nonrumor)

if __name__ == '__main__':
    main()
