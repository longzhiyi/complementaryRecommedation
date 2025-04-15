import pandas
import sklearn
import numpy as np
import torch
import pytorch_lightning
import json
import os

print(f"Pandas version: {pandas.__version__}")
print(f"Sklearn version: {sklearn.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch Lightning version: {pytorch_lightning.__version__}")

def find_unknown_category_apis():
    """
    找出原始数据集中类别为Unknown的API
    """
    try:
        # 加载原始API数据
        with open('data/raw/apiData.json', 'r', encoding='utf-8') as f:
            apis = json.load(f)
        
        # 找出类别为Unknown的API
        unknown_category_apis = [
            api for api in apis 
            if api.get("Primary Category") == "Unknown"
        ]
        
        # 打印结果
        print(f"\n总共找到 {len(unknown_category_apis)} 个类别为Unknown的API\n")
        print("详细信息：")
        print("-" * 80)
        
        for i, api in enumerate(unknown_category_apis, 1):
            print(f"\n{i}. API详细信息：")
            for key, value in api.items():
                # 格式化输出，使长文本更易读
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}:\n  {value}")
                else:
                    print(f"{key}: {value}")
            print("-" * 80)

    except FileNotFoundError:
        print("找不到API数据文件，请确认文件路径是否正确")
    except json.JSONDecodeError:
        print("JSON文件格式错误，请检查数据文件")
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    find_unknown_category_apis()