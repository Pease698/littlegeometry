import os
import sys
import random
import json
from absl import logging

# 动态配置系统路径，确保能无缝调用 reuse 里的模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'reuse'))
sys.path.append(os.path.join(PROJECT_ROOT, 'origin'))
sys.path.append(os.path.join(PROJECT_ROOT, 'meliad_lib', 'meliad'))

from reuse import problem
from reuse import graph
from reuse import ddar
from reuse import trace_back
from reuse import pretty


def proof_step_string(
      proof_step: problem.Dependency, refs: dict[tuple[str, ...], int], last_step: bool) -> str:
    """Translate proof to natural language.

    Args:
      proof_step: problem.Dependency with .name and .args
      refs: dict(hash: int) to keep track of derived predicates
      last_step: boolean to keep track whether this is the last step.

    Returns:
      a string of (pseudo) natural language of the proof step for human reader.
    """
    premises, [conclusion] = proof_step

    premises_nl = ' & '.join(
        [
            natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
            for p in premises
        ]
    )

    if not premises:
      premises_nl = 'similarly'

    refs[conclusion.hashed()] = len(refs)

    conclusion_nl = natural_language_statement(conclusion)
    if not last_step:
      conclusion_nl += ' [{:02}]'.format(refs[conclusion.hashed()])

    return f'{premises_nl} \u21d2 {conclusion_nl}'


def natural_language_statement(logical_statement: problem.Dependency) -> str:
    """Convert logical_statement to natural language.

    Args:
      logical_statement: problem.Dependency with .name and .args

    Returns:
      a string of (pseudo) natural language of the predicate for human reader.
    """
    # 判断是否有 .name 属性，没有则直接使用 str() 转化为字符串
    names = [a.name.upper() if hasattr(a, 'name') else str(a).upper() for a in logical_statement.args]
    names = [(n[0] + '_' + n[1:]) if len(n) > 1 else n for n in names]
    
    # 获取自然语言，如果无法匹配对应的自然语言，降级使用底层 DSL
    nl = pretty.pretty_nl(logical_statement.name, names)
    if not nl:
        nl = pretty.pretty([logical_statement.name] + names)
    return nl


def write_solution(g: graph.Graph, p: problem.Problem, out_file: str) -> None:
    """Output the solution to out_file.

    Args:
      g: graph.Graph object, containing the proof state.
      p: problem.Problem object, containing the theorem.
      out_file: file to write to, empty string to skip writing to file.
    """
    setup, aux, proof_steps, refs = ddar.get_proof_steps(
        g, p.goal, merge_trivials=False
    )

    solution = '\n=========================='
    solution += '\n * From theorem premises:\n'
    premises_nl = []
    for premises, [points] in setup:
      solution += ' '.join([p.name.upper() for p in points]) + ' '
      if not premises:
        continue
      premises_nl += [
          natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
          for p in premises
      ]
    solution += ': Points\n' + '\n'.join(premises_nl)

    solution += '\n\n * Target conclusion:\n' + natural_language_statement(p.goal)

    solution += '\n\n * Auxiliary Constructions:\n'
    aux_premises_nl = []
    for premises, [points] in aux:
      solution += ' '.join([p.name.upper() for p in points]) + ' '
      aux_premises_nl += [
          natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()])
          for p in premises
      ]
    if aux_premises_nl:
      solution += ': Points\n' + '\n'.join(aux_premises_nl)
    else:
      solution += '(None)'

    # some special case where the deduction rule has a well known name.
    r2name = {
        'r32': '(SSS)',
        'r33': '(SAS)',
        'r34': '(Similar Triangles)',
        'r35': '(Similar Triangles)',
        'r36': '(ASA)',
        'r37': '(ASA)',
        'r38': '(Similar Triangles)',
        'r39': '(Similar Triangles)',
        'r40': '(Congruent Triangles)',
        'a00': '(Distance chase)',
        'a01': '(Ratio chase)',
        'a02': '(Angle chase)',
    }

    solution += '\n\n * Proof steps:\n'
    for i, step in enumerate(proof_steps):
      _, [con] = step
      nl = proof_step_string(step, refs, last_step=i == len(proof_steps) - 1)
      rule_name = r2name.get(con.rule_name, '')
      nl = nl.replace('\u21d2', f'{rule_name}\u21d2 ')
      solution += '{:03}. '.format(i + 1) + nl + '\n'

    solution += '==========================\n'

    print(solution)
    if out_file:
        with open(out_file, 'w') as f:
            f.write(solution)
        print(f'Solution written to {out_file}.')


def generate_random_premises(num_extra_clauses=3):
    """
    随机前提生成器
    :param num_extra_clauses: 除了基础三角形外，想要额外添加的随机前提数量
    """
    points = ['a', 'b', 'c']
    premises_str = "a b c = triangle"
    
    # 准备英文字母表，用于给新产生的点命名
    available_names = [chr(i) for i in range(100, 123)] # d 到 z
    
    # 定义辅助作图动作池
    # 格式：(动作名称, 该动作需要的已知点数量)
    safe_actions = [
        ("midpoint", 2),   # 取两点连线的中点
        ("on_line", 2),    # 在两点连线上任取一点
        ("on_bline", 2),   # 在两点中垂线上任取一点
        ("foot", 3),       # 作一点到另外两点连线的垂足
        ("on_pline", 3),   # 过一点作另外两点连线的平行线，并在上面任取一点
        ("on_tline", 3),   # 过一点作另外两点连线的垂线，并在上面任取一点
    ]
    
    # 循环生成随机作图步骤
    for _ in range(num_extra_clauses):
        if not available_names:
            break # 字母用完则停止
            
        # 随机挑选一个作图动作
        action, required_points = random.choice(safe_actions)
        
        # 从已经存在的点中，随机抽取不重复的点作为动作的输入
        sampled_points = random.sample(points, required_points)
        
        # 取出一个新字母作为新点的名字
        new_p = available_names.pop(0)
        
        # 语法格式："新点 = 动作 新点 已知点1 已知点2 ..."
        if required_points == 2:
            clause = f"{new_p} = {action} {new_p} {sampled_points[0]} {sampled_points[1]}"
        elif required_points == 3:
            clause = f"{new_p} = {action} {new_p} {sampled_points[0]} {sampled_points[1]} {sampled_points[2]}"
        
        # 将新条件追加到总字符串中
        premises_str += f"; {clause}"
        
        # 将新点加入已有的“点池”
        points.append(new_p)
        
    return f"synthetic_random_graph\n{premises_str}"


def format_dep(dep, refs):
    """
    【新增】格式化工具函数：将 Dependency 对象转化为带引用编号的自然语言
    """
    # 解析参数名，处理可能仍是 Point 对象的参数
    args_str = [arg.name if hasattr(arg, 'name') else str(arg) for arg in dep.args]
    # 调用 pretty 转化为自然语言
    nl = pretty.pretty_nl(dep.name, args_str) or pretty.pretty([dep.name] + args_str)
    
    # 尝试从全局 refs 字典中获取该定理的编号
    h = dep.hashed()
    if h in refs:
        return f"{nl} [{refs[h]:02d}]"
    return nl


def generate_data(num_extra_clauses = 5, file_path = None):
    """
    生成合成几何数据
    :param num_extra_clauses: 额外添加的随机前提数量
    :param file_path: 可选参数，指定将生成的数据保存到哪个文件
    """
    output_flag = True
    if file_path is not None:
        output_flag = False

    if output_flag:
        print("=== 启动 AlphaGeometry 微型数据生成器 ===")

    # 加载几何定义和推理规则
    defs_path = os.path.join(PROJECT_ROOT, 'defs.txt')
    rules_path = os.path.join(PROJECT_ROOT, 'rules.txt')

    # 导入几何定义和推导规则
    definitions = problem.Definition.from_txt_file(defs_path, to_dict=True)
    theorems = problem.Theorem.from_txt_file(rules_path, to_dict=True)

    generation_attempts = 0 # 推导前提随机次数

    while True:
        generation_attempts += 1
        if output_flag:
            print(f"\n=== 第 {generation_attempts} 次随机前提生成尝试 ===")

        # 模拟“随机采样前提”，随机连续作图若干次
        simulated_random_premises = generate_random_premises(num_extra_clauses)
        if output_flag:
            print(f"[*] 初始随机字符串:\n{simulated_random_premises}\n")

        # 使用 Problem 类解析
        p = problem.Problem.from_txt(simulated_random_premises, translate=True)
    
        try:
            g, added_deps = graph.Graph.build_problem(p, definitions, verbose=False)
        except ValueError as e:
            if output_flag:
                print(f"[-] 生成了退化图形 ({e})，正在重新生成...")
            continue

        if output_flag:
            print(f"[*] 建图成功！当前图内已有 {len(g.all_nodes())} 个初始节点。")

        # 按照 ddar.solve 的真实签名传参
        # g: 状态图, theorems: 规则列表, controller: 问题对象 p
        # 因为生成数据时 p.goal 是 None，所以它会一直运行到 max_level 或饱和为止
        g, level_times, status, branches, all_added = ddar.solve(
            g, 
            theorems, 
            controller=p, 
            max_level=1000, 
            timeout=600
        )

        if output_flag:
            print(f"[*] 推导完成！图中当前共有 {len(g.all_nodes())} 个几何结论节点。")

        if not all_added:
            if output_flag:
                print("[-] 初始条件太简单")
            return

        target_dep = all_added[-1]
        
        # 提取原始逻辑链
        setup_raw, aux_raw, log_raw, setup_points = trace_back.get_logs(target_dep, g, merge_trivials=True)

        if file_path is not None and len(aux_raw) == 0:
            print("[-] 没有辅助点，重新生成")
            continue # 如果没有辅助构造点，则重新生成

        if file_path is not None:
            # 提取 DSL 格式的输入 (Input)
            # 使用 pretty.pretty 函数将 Dependency 对象转回类似 "T a b c d" 的底层符号串
            setup_dsl = [pretty.pretty([dep.name] + [a.name if hasattr(a, 'name') else str(a) for a in dep.args]) for dep in setup_raw]
            
            target_args = [a.name if hasattr(a, 'name') else str(a) for a in target_dep.args]
            target_dsl = pretty.pretty([target_dep.name] + target_args)

            # 提取 DSL 格式的标签 (Label - 辅助点)
            aux_dsl = [pretty.pretty([dep.name] + [a.name if hasattr(a, 'name') else str(a) for a in dep.args]) for dep in aux_raw]

            # 构建训练样本字典
            training_sample = {
                "problem_id": f"synthetic_{random.randint(10000, 99999)}",
                "premises": setup_dsl,        # 模型输入 1
                "target": target_dsl,         # 模型输入 2
                "auxiliary_points": aux_dsl,  # 模型预测标签
                "proof_length": len(log_raw)  # 元数据：可用于后续按难度过滤数据
            }

            #以追加模式 ('a') 写入 JSONL 文件
            try:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                # 写入成功后退出循环
                break 
            except Exception as e:
                print(f"写入文件失败: {e}")
                break
        else:
            if output_flag:
                print("\n========================================================")
                print("合成几何题生成完毕 (Synthetic Data Ready)")
                print("========================================================")

            target_args = [a.name if hasattr(a, 'name') else str(a) for a in target_dep.args]
            p.goal = problem.Construction(target_dep.name, target_args)

            write_solution(g, p, "")
            break


def main():
    file_path = os.path.join(PROJECT_ROOT, 'synthetic_data.jsonl')
    generate_data(20, None)


if __name__ == "__main__":
    main()

