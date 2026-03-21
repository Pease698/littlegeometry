import json
import re

def generate_point_mapping(unique_points):
    """
    根据去重后的点列表，生成映射字典
    """
    mapping = {}
    for i, point in enumerate(unique_points):
        quotient = i // 26
        remainder = i % 26
        base_char = chr(97 + remainder)
        
        if quotient == 0:
            mapping[point] = base_char
        else:
            mapping[point] = f"{base_char}{quotient}"
    return mapping

def process_geometry_jsonl(input_file, output_file):
    """
    处理整个 JSONL 文件，进行一致性的点映射
    """
    # 正则表达式匹配点名称
    point_pattern = re.compile(r'\b[a-z]\d*\b')
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            data = json.loads(line.strip())
            
            # 收集所有需要扫描的文本，确定点的出现顺序
            text_elements_for_scanning = []
            text_elements_for_scanning.extend(data.get('premises', []))
            if 'target' in data:
                text_elements_for_scanning.append(data['target'])
            text_elements_for_scanning.extend(data.get('auxiliary_points', []))
            
            # 提取并去重该题目的所有点
            unique_points = []
            for text in text_elements_for_scanning:
                points_in_text = point_pattern.findall(text)
                for pt in points_in_text:
                    if pt not in unique_points:
                        unique_points.append(pt)
                        
            # 生成该题目的映射字典
            mapping = generate_point_mapping(unique_points)
            
            # 定义替换回调函数
            def replace_func(match):
                return mapping[match.group(0)]
            
            # 对各个字段进行替换并更新 data 字典
            if 'premises' in data:
                data['premises'] = [point_pattern.sub(replace_func, p) for p in data['premises']]
                
            if 'target' in data:
                data['target'] = point_pattern.sub(replace_func, data['target'])
                
            if 'auxiliary_points' in data:
                data['auxiliary_points'] = [point_pattern.sub(replace_func, ap) for ap in data['auxiliary_points']]
                
            # 处理后的数据写回新文件
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    INPUT_FILE = "traindata/synthetic_data_v2.jsonl"
    OUTPUT_FILE = "traindata/synthetic_data_v3.jsonl"
    
    process_geometry_jsonl(INPUT_FILE, OUTPUT_FILE)
    print(f"数据转换完成！处理后的数据已保存至: {OUTPUT_FILE}")
