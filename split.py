import json


def find_coco_image_ids(annotation_file, keywords):
    """
    在 COCO 标注文件中检索包含所有指定关键词的图片 ID。
    """
    print(f"正在加载 {annotation_file}，请稍候...")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data['annotations']

    # 统一将关键词转换为小写，实现不区分大小写的匹配
    keywords = [kw.lower() for kw in keywords]

    matched_images = {}

    for ann in annotations:
        caption = ann['caption'].lower()
        # 检查是否所有关键词都在当前 caption 中
        if all(kw in caption for kw in keywords):
            img_id = ann['image_id']
            if img_id not in matched_images:
                matched_images[img_id] = []
            matched_images[img_id].append(ann['caption'])

    return matched_images


# --- 配置区域 ---
# 请将这里的路径替换为你本地的 COCO 真值标注文件路径
# 例如: 'captions_train2014.json' 或 'captions_val2014.json'
annotation_path = 'data/captions_val2014.json'

# 初始关键词：橙色工作服、火车
# 如果返回结果太多，可以尝试加入 'worker', 'black and white' 等词进一步缩小范围
search_keywords = ['orange', 'train','a','worker']

if __name__ == "__main__":
    try:
        results = find_coco_image_ids(annotation_path, search_keywords)

        print(f"\n检索完毕！共找到 {len(results)} 张可能匹配的图片：\n")
        print("=" * 50)

        for img_id, captions in results.items():
            # COCO 图片文件名通常格式为 COCO_train2014_000000XXXXXX.jpg
            # 这里将 ID 格式化为标准的 12 位数字

            print(f"图片 ID: {img_id}")
            print("包含关键词的图像描述:")
            for cap in captions:
                print(f"  - {cap}")
            print("-" * 50)

    except FileNotFoundError:
        print(f"错误: 未找到文件 '{annotation_path}'。请检查路径或文件名是否正确。")