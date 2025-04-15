import json
import pandas as pd
import os
import re
import random
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

"""
数据处理总体流程：
1. 创建目录结构
2. 加载API和Mashup数据
3. 从API数据中提取类别信息
4. 根据Mashup数据生成互补API对
5. 创建类别对关系并过滤低频类别对
6. 生成训练集、验证集和测试集
7. 为训练集生成负样本（用于triplet loss）
8. 创建API嵌入向量
9. 保存处理后的数据集
10. 打印统计信息

输入文件：
- data/raw/apiData.json: API数据，包含API名称、描述、类别等信息
- data/raw/mashupData.json: Mashup数据，包含Mashup名称和相关API信息

输出文件：
- data/processed/train_data.json: 训练集
- data/processed/val_data.json: 验证集
- data/processed/test_data.json: 测试集
- data/processed/enhanced_train_data.json: 增强训练数据（含负样本）
- data/processed/api_embeddings.json: API嵌入向量
- data/processed/category_pairs.json: 类别对关系
"""

# 创建目录结构
def create_directory_structure():
    """
    创建数据处理所需的目录结构
    
    输入：无
    输出：在当前工作目录下创建以下文件夹：
        - data/raw: 存放原始数据
        - data/processed: 存放处理后的数据
        - data/final: 存放最终数据
        - outputs: 存放输出结果
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/final",
        "outputs"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("目录结构创建完成")

# 加载API和Mashup数据
def load_data(api_path, mashup_path):
    """
    从指定路径加载API和Mashup数据
    
    输入：
        - api_path (str): API数据文件路径
        - mashup_path (str): Mashup数据文件路径
        
    输出：
        - apis (list): API数据列表，每个元素是一个dict，包含API的各种属性
        - mashups (list): Mashup数据列表，每个元素是一个dict，包含Mashup的各种属性
        
    数据格式示例：
    API数据：
    {
        "Name": "Scrapfly",
        "Description": "Scrapfly API is a simple but powerful Web scraping API...",
        "Primary Category": "Extraction",
        ...
    }
    
    Mashup数据：
    {
        "Name": "Website-Grader.com",
        "Related APIs": "Google PageSpeed Insights",
        ...
    }
    """
    # 加载API数据
    with open(api_path, 'r', encoding='utf-8') as f:
        apis = json.load(f)
    
    # 加载Mashup数据
    with open(mashup_path, 'r', encoding='utf-8') as f:
        mashups = json.load(f)
    
    print(f"加载了 {len(apis)} 个API和 {len(mashups)} 个Mashup")
    return apis, mashups

# 提取API的类别
def extract_categories(apis):
    """
    从API数据中提取所有不同的Primary Category
    
    输入：
        - apis (list): API数据列表
        
    输出：
        - categories (list): 所有不同的Primary Category列表
        
    处理步骤：
    1. 遍历所有API
    2. 提取每个API的Primary Category
    3. 使用set去重
    4. 转换为列表返回
    """
    primary_categories = set()
    
    # 遍历所有API，提取Primary Category
    for api in apis:
        if "Primary Category" in api:
            primary_categories.add(api["Primary Category"])
    
    categories = list(primary_categories)
    print(f"提取了 {len(categories)} 个不同的API类别")
    return categories

# 根据Mashup数据生成互补API对
def generate_complementary_pairs(apis, mashups):
    """
    根据Mashup数据生成互补API对
    
    输入：
        - apis (list): API数据列表
        - mashups (list): Mashup数据列表
        
    输出：
        - complementary_pairs (list): 互补API对列表，每个元素是一个元组(api1, api2)
        
    处理步骤：
    1. 创建API名称到API对象的映射
    2. 遍历所有Mashup
    3. 对每个Mashup，提取其Related APIs
    4. 检查每对API是否属于不同类别
    5. 如果是，则认为它们是互补的，添加到结果中
    6. 确保API对不重复（通过对API名称排序和使用集合）
    
    注意：
    - 只考虑不同类别的API对作为互补对
    - 对API名称进行排序，确保(api1, api2)和(api2, api1)被视为相同的对
    """
    # 创建API名称到API对象的映射，用于快速查找
    api_name_to_obj = {api["Name"]: api for api in apis}
    
    # 在开始处理之前，确保所有API都有类别信息
    api_name_to_obj = process_api_categories(api_name_to_obj)
    
    # 使用set来确保不重复
    complementary_pairs = set()
    
    # 遍历所有Mashup
    for mashup in mashups:
        if "Related APIs" in mashup and mashup["Related APIs"]:
            # 提取关联的API名称，使用正则表达式分割（处理逗号和空格）
            related_apis = re.split(r',\s*', mashup["Related APIs"])
            
            # 过滤掉不在我们API列表中的API
            valid_apis = [api for api in related_apis if api in api_name_to_obj]
            
            # 如果有多个有效的API，创建所有可能的对
            for i in range(len(valid_apis)):
                for j in range(i+1, len(valid_apis)):
                    api1 = valid_apis[i]
                    api2 = valid_apis[j]
                    
                    # 检查这两个API是否属于不同类别
                    if (api1 in api_name_to_obj and api2 in api_name_to_obj):
                        api1_obj = api_name_to_obj[api1]
                        api2_obj = api_name_to_obj[api2]
                        
                        # 获取类别
                        cat1 = api1_obj.get("Primary Category", "Unknown")
                        cat2 = api2_obj.get("Primary Category", "Unknown")
                        
                        # 如果属于不同类别，则认为是互补的
                        if cat1 != cat2:
                            # 对API名称进行排序，确保(api1, api2)和(api2, api1)被视为相同的对
                            if api1 < api2:
                                complementary_pairs.add((api1, api2))
                            else:
                                complementary_pairs.add((api2, api1))
    
    # 转换为列表返回
    result = list(complementary_pairs)
    print(f"生成了 {len(result)} 对互补API")
    return result

# 在处理API数据时添加类别检查和设置
def process_api_categories(api_name_to_obj):
    """
    处理API数据，确保所有API都有类别信息
    如果API没有类别，设置为'Unknown'
    """
    # 遍历所有API对象
    for api_name, api_obj in api_name_to_obj.items():
        # 检查是否存在类别信息，如果没有则设置为'Unknown'
        if 'Primary Category' not in api_obj or not api_obj['Primary Category']:
            api_obj['Primary Category'] = 'Unknown'
            print(f"Set category for API {api_name} to Unknown")
    return api_name_to_obj

# 创建类别对关系
def create_category_pairs(apis, complementary_pairs):
    """
    从互补API对中创建类别对关系
    
    输入：
        - apis (list): API数据列表
        - complementary_pairs (list): 互补API对列表
        
    输出：
        - filtered_category_pairs (dict): 过滤后的类别对关系，键为(category1, category2)，值为出现次数
        
    处理步骤：
    1. 创建API名称到API对象的映射
    2. 遍历所有互补API对
    3. 获取每对API的类别
    4. 统计每对类别的出现次数
    5. 过滤掉出现次数低于阈值的类别对
    
    注意：
    - 类别对也进行排序，确保(cat1, cat2)和(cat2, cat1)被视为相同的对
    - 使用阈值过滤低频类别对，减少噪声
    """
    # 创建API名称到API对象的映射
    api_name_to_obj = {api["Name"]: api for api in apis}
    
    # 在开始处理之前，确保所有API都有类别信息
    api_name_to_obj = process_api_categories(api_name_to_obj)
    
    # 创建类别对字典，使用defaultdict自动初始化计数为0
    category_pairs = defaultdict(int)
    
    # 遍历所有互补API对
    for api1, api2 in complementary_pairs:
        if api1 in api_name_to_obj and api2 in api_name_to_obj:
            # 获取类别（现在不需要再次调用process_api_categories）
            cat1 = api_name_to_obj[api1]['Primary Category']
            cat2 = api_name_to_obj[api2]['Primary Category']
            
            # 始终将字母序较小的类别放在前面
            if cat1 > cat2:
                cat1, cat2 = cat2, cat1
            
            category_pair = (cat1, cat2)
            category_pairs[category_pair] += 1
    
    # 过滤掉频率低的类别对
    threshold = 2  # 设置阈值，可以根据实际情况调整
    filtered_category_pairs = {pair: count for pair, count in category_pairs.items() if count >= threshold}
    
    print(f"创建了 {len(filtered_category_pairs)} 对互补类别关系（阈值={threshold}）")
    return filtered_category_pairs

# 生成训练集、验证集和测试集
def generate_datasets(apis, complementary_pairs, category_pairs):
    """
    生成训练集、验证集和测试集
    
    输入：
        - apis (list): API数据列表
        - complementary_pairs (list): 互补API对列表
        - category_pairs (dict): 类别对关系
        
    输出：
        - train_data (list): 训练集
        - val_data (list): 验证集
        - test_data (list): 测试集
        
    处理步骤：
    1. 创建API名称到API对象的映射
    2. 从互补API对中筛选符合类别对约束的数据
    3. 随机打乱数据集
    4. 划分为训练集、验证集和测试集
    5. 特殊处理：确保测试集中的种子API不在训练集中（模拟冷启动）
    
    数据格式：
    {
        "seed_api": "API1",
        "complementary_api": "API2",
        "seed_category": "Category1",
        "complementary_category": "Category2"
    }
    """
    # 创建API名称到API对象的映射
    api_name_to_obj = {api["Name"]: api for api in apis}
    
    # 整理数据为所需的格式
    dataset = []
    
    # 遍历所有互补API对
    for api1, api2 in complementary_pairs:
        if api1 in api_name_to_obj and api2 in api_name_to_obj:
            api1_obj = api_name_to_obj[api1]
            api2_obj = api_name_to_obj[api2]
            
            # 获取类别
            cat1 = api1_obj.get("Primary Category", "Unknown")
            cat2 = api2_obj.get("Primary Category", "Unknown")
            
            # 检查这个类别对是否在我们筛选的类别对中
            # 确保类别对的顺序一致
            check_cat1, check_cat2 = (cat1, cat2) if cat1 < cat2 else (cat2, cat1)
                
            if (check_cat1, check_cat2) in category_pairs:
                # 将API对添加到数据集
                dataset.append({
                    "seed_api": api1,
                    "complementary_api": api2,
                    "seed_category": cat1,
                    "complementary_category": cat2
                })
    
    # 随机打乱数据集
    random.shuffle(dataset)
    
    # 划分训练集、验证集和测试集（80%训练，10%验证，10%测试）
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # 特殊处理：确保测试集中的种子API不在训练集中（模拟冷启动）
    train_seed_apis = set(item["seed_api"] for item in train_data)
    
    # 找出在训练集中出现的测试集种子API
    overlapping_apis = [item for item in test_data if item["seed_api"] in train_seed_apis]
    
    # 从重复的API中随机选择一些放入验证集，其余保留在测试集
    if overlapping_apis:
        move_to_val, keep_in_test = train_test_split(overlapping_apis, test_size=0.5, random_state=42)
        
        # 从测试集中移除这些项
        test_data = [item for item in test_data if item not in move_to_val]
        
        # 将它们添加到验证集
        val_data.extend(move_to_val)
    
    print(f"划分数据集: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
    return train_data, val_data, test_data


def generate_negative_samples(apis, train_data, complementary_pairs):
    """
    为训练集生成负样本，确保任何曾经共同调用过的API对都不会作为负样本
    
    输入：
        - apis (list): API数据列表
        - train_data (list): 训练集
        - complementary_pairs (list): 互补API对列表
        
    输出：
        - enhanced_train_data (list): 增强训练数据，包含负样本
    """
    # 创建并处理API字典
    api_name_to_obj = {api["Name"]: api for api in apis}
    api_name_to_obj = process_api_categories(api_name_to_obj)
    
    # 创建已知的互补API对集合(无序对)
    complementary_api_pairs = set()
    for api1, api2 in complementary_pairs:
        # 确保对的顺序一致性
        if api1 < api2:
            complementary_api_pairs.add((api1, api2))
        else:
            complementary_api_pairs.add((api2, api1))
    
    # 增强训练数据集添加负样本
    enhanced_train_data = []
    
    # 遍历训练集中的每个样本
    for item in train_data:
        seed_api = item["seed_api"]
        complementary_api = item["complementary_api"]
        complementary_category = item["complementary_category"]
        
        # 为每个正例寻找一个负例
        # 选择同一类别的API作为潜在负样本
        potential_negatives = [
            name for name, api in api_name_to_obj.items()
            if api.get("Primary Category", "") == complementary_category 
            and name != complementary_api
        ]
        
        # 过滤掉那些与种子API曾经有互补关系的API
        valid_negatives = []
        for neg_api in potential_negatives:
            # 检查这个潜在负样本是否与种子API形成过互补对
            pair = (seed_api, neg_api) if seed_api < neg_api else (neg_api, seed_api)
            if pair not in complementary_api_pairs:
                valid_negatives.append(neg_api)
        
        # 如果找到了有效的负样本
        if valid_negatives:
            # 随机选择一个负样本
            negative_api = random.choice(valid_negatives)
            
            # 添加增强的训练样本
            enhanced_train_data.append({
                "seed_api": seed_api,
                "positive_api": complementary_api,
                "negative_api": negative_api,
                "seed_category": item["seed_category"],
                "target_category": complementary_category
            })
        else:
            # 如果找不到合适的负样本，可以跳过此样本或使用其他策略
            print(f"警告: 无法为种子API '{seed_api}' 和类别 '{complementary_category}' 找到合适的负样本")
    
    print(f"生成了 {len(enhanced_train_data)} 条带负样本的增强训练数据")
    return enhanced_train_data

# 创建API描述的嵌入向量（特征提取）
def create_api_embeddings(apis):
    """
    创建API描述的嵌入向量（特征提取）
    
    输入：
        - apis (list): API数据列表
        
    输出：
        - 保存API嵌入向量到文件
        
    处理步骤：
    1. 遍历所有API
    2. 提取相关特征（描述长度、类别等）
    3. 生成嵌入向量
    4. 保存到文件
    
    注意：
    这里使用简化方法，只基于描述长度和随机向量生成嵌入
    在实际应用中，应该使用更复杂的NLP模型（如BERT）生成更有意义的嵌入
    """
    # 创建API嵌入字典
    api_embeddings = {}
    
    # 遍历所有API
    for api in apis:
        # 使用描述长度和类别作为特征
        desc_length = len(api.get("Description", ""))
        primary_category = api.get("Primary Category", "Unknown")
        
        # 生成一个64维的随机向量，在实际实现中应该使用NLP模型
        embedding = np.random.rand(64)
        
        # 将部分特征映射到向量中
        embedding[0] = desc_length / 1000  # 归一化描述长度
        
        # 存储API嵌入
        api_embeddings[api["Name"]] = embedding.tolist()
    
    # 保存API嵌入
    with open("data/processed/api_embeddings.json", 'w', encoding='utf-8') as f:
        json.dump(api_embeddings, f)
    
    print(f"为 {len(api_embeddings)} 个API创建了嵌入向量")

# 保存处理后的数据集
def save_datasets(train_data, val_data, test_data, enhanced_train_data, category_pairs):
    """
    保存处理后的数据集
    
    输入：
        - train_data (list): 训练集
        - val_data (list): 验证集
        - test_data (list): 测试集
        - enhanced_train_data (list): 增强训练数据（含负样本）
        - category_pairs (dict): 类别对关系
        
    输出：
        - 保存数据集到文件
    """
    # 保存划分好的数据集
    with open("data/processed/train_data.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    
    with open("data/processed/val_data.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)
    
    with open("data/processed/test_data.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
    
    # 保存带负样本的增强训练数据
    with open("data/processed/enhanced_train_data.json", 'w', encoding='utf-8') as f:
        json.dump(enhanced_train_data, f, indent=4)
    
    # 保存类别对信息
    with open("data/processed/category_pairs.json", 'w', encoding='utf-8') as f:
        # 将defaultdict转换为普通dict，并将tuple键转换为字符串
        serializable_pairs = {f"{cat1}_{cat2}": count for (cat1, cat2), count in category_pairs.items()}
        json.dump(serializable_pairs, f, indent=4)
    
    print("数据集已保存到 data/processed/ 目录")

# 打印数据统计信息
def print_statistics(apis, complementary_pairs, category_pairs, train_data, val_data, test_data):
    """
    打印数据统计信息
    
    输入：
        - apis (list): API数据列表
        - complementary_pairs (list): 互补API对列表
        - category_pairs (dict): 类别对关系
        - train_data (list): 训练集
        - val_data (list): 验证集
        - test_data (list): 测试集
        
    输出：
        - 打印统计信息到控制台
    """
    print("\n=== 数据集统计信息 ===")
    print(f"API总数: {len(apis)}")
    print(f"互补API对总数: {len(complementary_pairs)}")
    print(f"类别对总数: {len(category_pairs)}")
    print(f"训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 统计类别分布
    categories = {}
    for api in apis:
        cat = api.get("Primary Category", "Unknown")
        if cat in categories:
            categories[cat] += 1
        else:
            categories[cat] = 1
    
    print("\n=== 类别分布 ===")
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_categories[:10]:  # 只显示前10个类别
        print(f"{cat}: {count} APIs")
    
    if len(sorted_categories) > 10:
        print(f"... 以及 {len(sorted_categories) - 10} 个其他类别")
    
    # 统计类别对分布
    print("\n=== 热门类别对 ===")
    sorted_pairs = sorted(category_pairs.items(), key=lambda x: x[1], reverse=True)
    for (cat1, cat2), count in sorted_pairs[:10]:  # 只显示前10个类别对
        print(f"{cat1} <-> {cat2}: {count} 对互补API")
    
    if len(sorted_pairs) > 10:
        print(f"... 以及 {len(sorted_pairs) - 10} 个其他类别对")

# 主函数
def main(api_path, mashup_path):
    """
    主函数，协调整个数据处理流程
    
    输入：
        - api_path (str): API数据文件路径
        - mashup_path (str): Mashup数据文件路径
        
    输出：
        - 处理后的数据集和统计信息
    """
    # 步骤1: 创建目录结构
    create_directory_structure()
    
    # 步骤2: 加载数据
    print("\n正在加载数据...")
    apis, mashups = load_data(api_path, mashup_path)
    
    # 步骤3: 提取类别
    print("\n正在提取API类别...")
    categories = extract_categories(apis)
    
    # 步骤4: 生成互补API对
    print("\n正在生成互补API对...")
    complementary_pairs = generate_complementary_pairs(apis, mashups)
    
    # 步骤5: 创建类别对关系
    print("\n正在创建类别对关系...")
    category_pairs = create_category_pairs(apis, complementary_pairs)
    
    # 步骤6: 生成数据集
    print("\n正在生成训练、验证和测试数据集...")
    train_data, val_data, test_data = generate_datasets(apis, complementary_pairs, category_pairs)
    
    # 步骤7: 生成带负样本的增强训练数据
    print("\n正在生成带负样本的增强训练数据...")
    enhanced_train_data = generate_negative_samples(apis, train_data, complementary_pairs)
    
    # 步骤8: 创建API嵌入
    print("\n正在创建API嵌入向量...")
    create_api_embeddings(apis)
    
    # 步骤9: 保存数据集
    print("\n正在保存处理后的数据集...")
    save_datasets(train_data, val_data, test_data, enhanced_train_data, category_pairs)
    
    # 步骤10: 打印统计信息
    print_statistics(apis, complementary_pairs, category_pairs, train_data, val_data, test_data)
    
    print("\n数据集处理完成!")

if __name__ == "__main__":
    # 设置文件路径
    api_path = "data/raw/apiData.json"
    mashup_path = "data/raw/mashupData.json"
    
    # 执行主函数
    main(api_path, mashup_path)