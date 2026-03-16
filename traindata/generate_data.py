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


def generate_random_premises(num_extra_clauses = 3, dof_chooser = 0.7):
    """
    随机前提生成器
    :param num_extra_clauses: 除了基础三角形外，想要额外添加的随机前提数量
    :param dof_chooser: 选择 0dof 作图的概率，剩余概率为 1dof 作图
    """
    points = ['a', 'b', 'c']
    premises_str = "a b c = triangle"
    
    # 准备英文字母表，用于给新产生的点命名
    available_names = [chr(i) for i in range(100, 123)] # d 到 z
    
    # 直接唯一确定一个点
    actions_0dof = [
        ("midpoint", 2),       # 中点
        ("foot", 3),           # 垂足
        ("circumcenter", 3),   # 外心
        ("orthocenter", 3)     # 垂心
    ]

    # 两约束条件共同确定一个点
    actions_1dof = [
        ("on_line", 2),    # 在两点连线上
        ("on_bline", 2),   # 在两点中垂线上
        ("on_pline", 3),   # on_pline p a b c   ->  p 在过 a 平行 bc 的线上
        ("on_tline", 3),   # on_tline p a b c   ->  p 在过 a 垂直 bc 的线上
        ("on_circum", 3)   # on_circum p a b c  ->  p 在 abc 外接圆上
    ]
    
    # 循环生成随机作图步骤
    for _ in range(num_extra_clauses):
        if not available_names:
            break # 字母用完则停止
        
        # 取出一个新字母作为新点的名字
        new_p = available_names.pop(0)

        # 随机决定使用 0dof 作图还是 1dof 作图
        if random.random() < dof_chooser:
            action, req_pts = random.choice(actions_0dof)
            sampled = random.sample(points, req_pts)
            args_str = " ".join(sampled)
            clause = f"{new_p} = {action} {new_p} {args_str}"
            
        else:
            action1, req_pts1 = random.choice(actions_1dof)
            action2, req_pts2 = random.choice(actions_1dof)
            
            sampled1 = random.sample(points, req_pts1)
            sampled2 = random.sample(points, req_pts2)
            
            args_str1 = " ".join(sampled1)
            args_str2 = " ".join(sampled2)
            
            clause = f"{new_p} = {action1} {new_p} {args_str1}, {action2} {new_p} {args_str2}"
        
        # 将新条件追加到总字符串中
        premises_str += f"; {clause}"
        
        # 将新点加入已有的“点池”
        points.append(new_p)
        
    return f"synthetic_random_graph\n{premises_str}"


def generate_data(syn_data_index, num_extra_clauses = 5, dof_chooser = 0.7, num_targets = 3, file_path = None):
    """
    生成合成几何数据
    :param syn_data_index: 生成数据的索引
    :param num_extra_clauses: 额外添加的随机前提数量
    :param dof_chooser: 选择 0dof 作图的概率，剩余概率为 1dof 作图
    :param num_targets: 随机选择以进行回溯的目标结论数量
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
        simulated_random_premises = generate_random_premises(num_extra_clauses, dof_chooser)
        if output_flag:
            print(f"[*] 初始随机字符串:\n{simulated_random_premises}\n")

        # 使用 Problem 类解析
        try:
            p = problem.Problem.from_txt(simulated_random_premises, translate=True)
        except Exception as e:
            if output_flag: print(f"[-] 字符串解析失败 ({e})，正在重新生成...")
            continue
    
        try:
            g, added_deps = graph.Graph.build_problem(p, definitions, verbose=False)
        except Exception as e:
            if output_flag:
                print(f"[-] 生成了退化图形 ({e})，正在重新生成...")
            continue

        if output_flag:
            print(f"[*] 建图成功！当前图内已有 {len(g.all_nodes())} 个初始节点。")

        try:
            # 按照 ddar.solve 的真实签名传参
            # g: 状态图, theorems: 规则列表, controller: 问题对象 p
            # 因为生成数据时 p.goal 是 None，所以它会一直运行到 max_level 或饱和为止
            g, level_times, status, branches, all_added = ddar.solve(
                g, 
                theorems, 
                controller=p, 
                max_level=1000, 
                timeout=60
            )
        except Exception as e:
            if output_flag: print(f"[-] 符号演绎阶段异常 ({e})，已拦截...")
            continue

        if output_flag:
            print(f"[*] 推导完成！图中当前共有 {len(g.all_nodes())} 个几何结论节点。")

        if not all_added:
            if output_flag:
                print("[-] 初始条件太简单")
            return

        # target_dep = all_added[-1]
        actual_num_targets = min(num_targets, len(all_added))
        sampled_targets = random.sample(all_added, actual_num_targets)
        valid_samples_for_this_graph = 0
        if output_flag:
            print(f"[*] 图构建与推导成功！从 {len(all_added)} 个新结论中随机抽取了 {actual_num_targets} 个作为求证目标。")

        for i, target_dep in enumerate(sampled_targets):
            if output_flag:
                print(f"\n--- 开始处理目标 {i+1}/{actual_num_targets} ---")

            try:
                # 借助 g.cache，即使处理多个目标，回溯速度也会越来越快
                setup_raw, aux_raw, log_raw, setup_points = trace_back.get_logs(target_dep, g, merge_trivials=True)
            except Exception as e:
                if output_flag: print(f"  [-] 回溯树构建异常 ({e})，跳过该结论...")
                continue

            if file_path is not None and len(aux_raw) == 0:
                if output_flag: print("  [-] 该结论无需辅助点即可证明，跳过...")
                continue 

            if file_path is not None:
                setup_dsl = [pretty.pretty([dep.name] + [a.name if hasattr(a, 'name') else str(a) for a in dep.args]) for dep in setup_raw]
                target_args = [a.name if hasattr(a, 'name') else str(a) for a in target_dep.args]
                target_dsl = pretty.pretty([target_dep.name] + target_args)
                aux_dsl = [pretty.pretty([dep.name] + [a.name if hasattr(a, 'name') else str(a) for a in dep.args]) for dep in aux_raw]

                training_sample = {
                    "problem_id": f"synthetic_{syn_data_index:06d}", # 加上后缀区分同源图
                    "premises": setup_dsl,        
                    "target": target_dsl,         
                    "auxiliary_points": aux_dsl,  
                    "proof_length": len(log_raw)  
                }
                syn_data_index += 1

                try:
                    with open(file_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                    valid_samples_for_this_graph += 1
                except Exception as e:
                    print(f"写入文件失败: {e}")
            else:
                if output_flag:
                    print("\n========================================================")
                    print("合成几何题生成完毕 (Synthetic Data Ready)")
                    print("========================================================")

                # 终端模式展示
                target_args = [a.name if hasattr(a, 'name') else str(a) for a in target_dep.args]
                p.goal = problem.Construction(target_dep.name, target_args)

                # 调用复刻的输出函数
                write_solution(g, p, "")
                graph.nm.draw(
                    g.type2nodes[graph.Point],
                    g.type2nodes[graph.Line],
                    g.type2nodes[graph.Circle],
                    g.type2nodes[graph.Segment])
                valid_samples_for_this_graph += 1

        if valid_samples_for_this_graph > 0:
            if output_flag:
                print(f"\n========================================================")
                print(f"本轮测试结束，共成功展示了 {valid_samples_for_this_graph} 个包含证明链的结论。")
                print(f"========================================================")
            break
    return syn_data_index


def main():
    syn_data_index = 0
    num_extra_clauses = 15
    dof_chooser = 0.5
    num_targets = 30
    file_path = os.path.join(PROJECT_ROOT, 'synthetic_data.jsonl')
    sum_syn_data = 1

    while syn_data_index < sum_syn_data:
        syn_data_index = generate_data(
            syn_data_index,
            num_extra_clauses,
            dof_chooser,
            num_targets,
            None
        )
    
    print(f"数据生成完成，当前索引已达{syn_data_index}")

if __name__ == "__main__":
    main()

