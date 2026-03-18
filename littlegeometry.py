import os
import sys
import traceback
import torch
from absl import app
from absl import flags
from absl import logging

import ddar
import graph as gh
import pretty as pt
import problem as pr

# 导入模型和分析器 
from transformers import GPT2LMHeadModel
from newmodel.tokenizer import GeometryTokenizer 

_PROBLEMS_FILE = flags.DEFINE_string('problems_file', 'imo_ag_30.txt', '题目文件')
_PROBLEM_NAME = flags.DEFINE_string('problem_name', 'imo_2000_p1', '要测试的题目名')
_DEFS_FILE = flags.DEFINE_string('defs_file', 'defs.txt', '定义文件')
_RULES_FILE = flags.DEFINE_string('rules_file', 'rules.txt', '规则文件')
_MODEL_DIR = flags.DEFINE_string('model_dir', './mini_ag_weights', '我们自己训练的模型路径')
_OUT_FILE = flags.DEFINE_string('out_file', 'solution_output.txt', '输出结果文件')
_BEAM_SIZE = flags.DEFINE_integer('beam_size', 2, '波束搜索的大小')
_SEARCH_DEPTH = flags.DEFINE_integer('search_depth', 2, '最大探索深度')

DEFINITIONS = None
RULES = None

def load_mini_ag_model(model_dir):
    """加载我们自己的 PyTorch 模型和分词器"""
    logging.info("正在加载自定义的 Mini-AlphaGeometry 模型...")
    tokenizer = GeometryTokenizer()
    tokenizer.build_vocab_from_jsonl("traindata/synthetic_data_v2.jsonl") 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer, device

def generate_candidates_hf(model, tokenizer, device, pstring, beam_size):
    """
    用 Hugging Face 模型替代 DeepMind 的原生模型生成辅助点，
    并返回 (生成文本, 对数概率得分) 的列表。
    """
    # 解析 DeepMind 的问题字符串
    if ' ? ' in pstring:
        setup, goal = pstring.split(' ? ')
    else:
        setup, goal = pstring, ""
        
    # 转换问题格式
    prompt = f"<bos> {setup.strip()} <sep> {goal.strip()} <sep>"
    
    # 转化为 Input IDs
    input_ids = [tokenizer.vocab.get(t, 0) for t in prompt.split()]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # 调用 HF 的波束搜索
    with torch.no_grad():
        stop_token_id = tokenizer.vocab.get(";", tokenizer.vocab["<eos>"])
        outputs = model.generate(
            input_tensor,
            max_new_tokens = 20,
            num_beams = beam_size,
            num_return_sequences = beam_size,
            pad_token_id = tokenizer.vocab["<pad>"],
            eos_token_id = stop_token_id,
            output_scores = True,
            return_dict_in_generate = True,
            early_stopping = True
        )
    
    # 解码并提取得分
    candidates = []
    # sequences_scores 包含了每个波束的得分
    for seq, score in zip(outputs.sequences, outputs.sequences_scores):
        generated_ids = seq[len(input_ids):].tolist()
        
        # 截断 <eos> 及其之后的内容
        if tokenizer.vocab["<eos>"] in generated_ids:
            eos_idx = generated_ids.index(tokenizer.vocab["<eos>"])
            generated_ids = generated_ids[:eos_idx]
            
        pred_str = tokenizer.decode(generated_ids)
        candidates.append((pred_str, score.item()))
        
    return candidates

# ==============================================================

def natural_language_statement(logical_statement: pr.Dependency) -> str:
    names = [a.name.upper() for a in logical_statement.args]
    names = [(n[0] + '_' + n[1:]) if len(n) > 1 else n for n in names]
    return pt.pretty_nl(logical_statement.name, names)

def proof_step_string(proof_step: pr.Dependency, refs: dict, last_step: bool) -> str:
    premises, [conclusion] = proof_step
    premises_nl = ' & '.join([natural_language_statement(p) + ' [{:02}]'.format(refs[p.hashed()]) for p in premises])
    if not premises: premises_nl = 'similarly'
    refs[conclusion.hashed()] = len(refs)
    conclusion_nl = natural_language_statement(conclusion)
    if not last_step: conclusion_nl += ' [{:02}]'.format(refs[conclusion.hashed()])
    return f'{premises_nl} \u21d2 {conclusion_nl}'

def write_solution(g: gh.Graph, p: pr.Problem, out_file: str) -> None:
    setup, aux, proof_steps, refs = ddar.get_proof_steps(g, p.goal, merge_trivials=False)
    solution = '\n==========================\n * Proof steps:\n'
    for i, step in enumerate(proof_steps):
        nl = proof_step_string(step, refs, last_step=i == len(proof_steps) - 1)
        solution += '{:03}. '.format(i + 1) + nl + '\n'
    solution += '==========================\n'
    logging.info(solution)
    if out_file:
        with open(out_file, 'w') as f: f.write(solution)

def run_ddar(g: gh.Graph, p: pr.Problem, out_file: str) -> bool:
    ddar.solve(g, RULES, p, max_level=1000)
    goal_args = g.names2nodes(p.goal.args)
    if not g.check(p.goal.name, goal_args): return False
    write_solution(g, p, out_file)
    return True

def translate_constrained_to_constructive(
    point: str, name: str, args: list[str]
) -> tuple[str, list[str]]:
  """Translate a predicate from constraint-based to construction-based.

  Args:
    point: str: name of the new point
    name: str: name of the predicate, e.g., perp, para, etc.
    args: list[str]: list of predicate args.

  Returns:
    (name, args): translated to constructive predicate.
  """
  if name in ['T', 'perp']:
    a, b, c, d = args
    if point in [c, d]:
      a, b, c, d = c, d, a, b
    if point == b:
      a, b = b, a
    if point == d:
      c, d = d, c
    if a == c and a == point:
      return 'on_dia', [a, b, d]
    return 'on_tline', [a, b, c, d]

  elif name in ['P', 'para']:
    a, b, c, d = args
    if point in [c, d]:
      a, b, c, d = c, d, a, b
    if point == b:
      a, b = b, a
    return 'on_pline', [a, b, c, d]

  elif name in ['D', 'cong']:
    a, b, c, d = args
    if point in [c, d]:
      a, b, c, d = c, d, a, b
    if point == b:
      a, b = b, a
    if point == d:
      c, d = d, c
    if a == c and a == point:
      return 'on_bline', [a, b, d]
    if b in [c, d]:
      if b == d:
        c, d = d, c  # pylint: disable=unused-variable
      return 'on_circle', [a, b, d]
    return 'eqdistance', [a, b, c, d]

  elif name in ['C', 'coll']:
    a, b, c = args
    if point == b:
      a, b = b, a
    if point == c:
      a, b, c = c, a, b
    return 'on_line', [a, b, c]

  elif name in ['^', 'eqangle']:
    a, b, c, d, e, f = args

    if point in [d, e, f]:
      a, b, c, d, e, f = d, e, f, a, b, c

    x, b, y, c, d = b, c, e, d, f
    if point == b:
      a, b, c, d = b, a, d, c

    if point == d and x == y:  # x p x b = x c x p
      return 'angle_bisector', [point, b, x, c]

    if point == x:
      return 'eqangle3', [x, a, b, y, c, d]

    return 'on_aline', [a, x, b, c, y, d]

  elif name in ['cyclic', 'O']:
    a, b, c = [x for x in args if x != point]
    return 'on_circum', [point, a, b, c]

  return name, args

def check_valid_args(name: str, args: list[str]) -> bool:
  """Check whether a predicate is grammarically correct.

  Args:
    name: str: name of the predicate
    args: list[str]: args of the predicate

  Returns:
    bool: whether the predicate arg count is valid.
  """
  if name == 'perp':
    if len(args) != 4:
      return False
    a, b, c, d = args
    if len({a, b}) < 2:
      return False
    if len({c, d}) < 2:
      return False
  elif name == 'para':
    if len(args) != 4:
      return False
    a, b, c, d = args
    if len({a, b, c, d}) < 4:
      return False
  elif name == 'cong':
    if len(args) != 4:
      return False
    a, b, c, d = args
    if len({a, b}) < 2:
      return False
    if len({c, d}) < 2:
      return False
  elif name == 'coll':
    if len(args) != 3:
      return False
    a, b, c = args
    if len({a, b, c}) < 3:
      return False
  elif name == 'cyclic':
    if len(args) != 4:
      return False
    a, b, c, d = args
    if len({a, b, c, d}) < 4:
      return False
  elif name == 'eqangle':
    if len(args) != 8:
      return False
    a, b, c, d, e, f, g, h = args
    if len({a, b, c, d}) < 3:
      return False
    if len({e, f, g, h}) < 3:
      return False
  return True

def try_translate_constrained_to_construct(string: str, g: gh.Graph) -> str:
  """Whether a string of aux construction can be constructed.

  Args:
    string: str: the string describing aux construction.
    g: gh.Graph: the current proof state.

  Returns:
    str: whether this construction is valid. If not, starts with "ERROR:".
  """
  string = string.strip()  # 去除可能存在的首尾空格
  if not string:           # 如果字符串是空的，直接当做错误生成处理
      return 'ERROR: empty prediction from LM'

  if string[-1] != ';':
    return 'ERROR: must end with ;'

  head, prem_str = string.split(' : ')
  point = head.strip()

  if len(point) != 1 or point == ' ':
    return f'ERROR: invalid point name {point}'

  existing_points = [p.name for p in g.all_points()]
  if point in existing_points:
    return f'ERROR: point {point} already exists.'

  prem_toks = prem_str.split()[:-1]  # remove the EOS ' ;'
  prems = [[]]

  for i, tok in enumerate(prem_toks):
    if tok.isdigit():
      if i < len(prem_toks) - 1:
        prems.append([])
    else:
      prems[-1].append(tok)

  if len(prems) > 2:
    return 'ERROR: there cannot be more than two predicates.'

  clause_txt = point + ' = '
  constructions = []

  for prem in prems:
    name, *args = prem

    if point not in args:
      return f'ERROR: {point} not found in predicate args.'

    if not check_valid_args(pt.map_symbol(name), args):
      return 'ERROR: Invalid predicate ' + name + ' ' + ' '.join(args)

    for a in args:
      if a != point and a not in existing_points:
        return f'ERROR: point {a} does not exist.'

    try:
      name, args = translate_constrained_to_constructive(point, name, args)
    except:  # pylint: disable=bare-except
      return 'ERROR: Invalid predicate ' + name + ' ' + ' '.join(args)

    if name == 'on_aline':
      if args.count(point) > 1:
        return f'ERROR: on_aline involves twice {point}'

    constructions += [name + ' ' + ' '.join(args)]

  clause_txt += ', '.join(constructions)
  clause = pr.Clause.from_txt(clause_txt)

  try:
    g.copy().add_clause(clause, 0, DEFINITIONS)
  except:  # pylint: disable=bare-except
    return 'ERROR: ' + traceback.format_exc()

  return clause_txt

def insert_aux_to_premise(pstring: str, auxstring: str) -> str:
    setup, goal = pstring.split(' ? ')
    return setup + '; ' + auxstring.replace(' ;', '') + ' ? ' + goal

class BeamQueue:
    def __init__(self, max_size=512): self.queue = []; self.max_size = max_size
    def add(self, node, val):
        if len(self.queue) < self.max_size: self.queue.append((val, node)); return
        min_idx, (min_val, _) = min(enumerate(self.queue), key=lambda x: x[1])
        if val > min_val: self.queue[min_idx] = (val, node)
    def __iter__(self): return iter(self.queue)
    def __len__(self): return len(self.queue)

def format_lm_prediction(raw_str: str) -> str:
    """
    将模型输出的脏字符串清洗为 DeepMind 严格要求的标准格式
    例如: "j : C c j g ; <pad>" -> "j : C c j g ;"
    例如: "m : C m b g" -> "m : C m b g ;"
    """
    # 1. 移除模型可能吐出的各种特殊 Token
    clean_str = raw_str.replace("<pad>", "").replace("<eos>", "").replace("<bos>", "").strip()
    
    # 2. 如果字符串已经自带分号，先把它剥离掉，防止出现多个分号或没有空格的情况
    if clean_str.endswith(";"):
        clean_str = clean_str[:-1].strip()
        
    # 3. 只要剥离后还有内容，我们就给它穿上标准的“制服”（加个空格和分号）
    if clean_str:
        clean_str = clean_str + " ;"
        
    return clean_str

# ==============================================================================
# 修改后的主循环: run_alphageometry
# ==============================================================================

def run_alphageometry(model, tokenizer, device, p: pr.Problem, search_depth: int, beam_size: int, out_file: str) -> bool:
    g, _ = gh.Graph.build_problem(p, DEFINITIONS)

    logging.info('尝试符号引擎求解')
    if run_ddar(g, p, out_file):
        logging.info('求解完成')
        return True

    beam_queue = BeamQueue(max_size=beam_size)
    # 起始节点: (图状态, pstring)
    beam_queue.add(node=(g, p.txt()), val=0.0)

    for depth in range(search_depth):
        logging.info(f'\n--- 开始探索深度: {depth} ---')
        new_queue = BeamQueue(max_size=beam_size)

        for prev_score, (g, pstring) in beam_queue:
            
            # 使用模型生成辅助点
            candidates = generate_candidates_hf(model, tokenizer, device, pstring, beam_size)
            
            # 遍历模型给出的几个建议
            for lm_out, score in candidates:
                cleaned_out = format_lm_prediction(lm_out)

                logging.info(f'模型建议 (得分 {score:.2f}): "{lm_out}"')
                
                translation = try_translate_constrained_to_construct(cleaned_out, g)
                if translation.startswith('ERROR:'):
                  logging.info(f"被引擎拒绝: {translation}")
                  continue
                
                # 将辅助点加入到题目已知条件中
                candidate_pstring = insert_aux_to_premise(pstring, translation)
                logging.info(f'符号引擎开始验证包含辅助点的新状态')
                
                p_new = pr.Problem.from_txt(candidate_pstring)
                g_new, _ = gh.Graph.build_problem(p_new, DEFINITIONS)
                
                # 再次调用符号引擎
                if run_ddar(g_new, p_new, out_file):
                    logging.info('解题成功')
                    return True

                new_queue.add(node=(g_new, candidate_pstring), val=prev_score + score)

        beam_queue = new_queue

    logging.info('达到了最大探索深度，未能找出证明')
    return False

def main(_):
    global DEFINITIONS, RULES
    DEFINITIONS = pr.Definition.from_txt_file(_DEFS_FILE.value, to_dict=True)
    RULES = pr.Theorem.from_txt_file(_RULES_FILE.value, to_dict=True)

    # 加载题目
    problems = pr.Problem.from_txt_file(_PROBLEMS_FILE.value, to_dict=True, translate=True)
    if _PROBLEM_NAME.value not in problems:
        raise ValueError(f'在 {_PROBLEMS_FILE.value} 中找不到题目 {_PROBLEM_NAME.value}')
    
    this_problem = problems[_PROBLEM_NAME.value]

    # 加载我们的微型模型
    model, tokenizer, device = load_mini_ag_model(_MODEL_DIR.value)
    
    # 运行大闭环
    run_alphageometry(
        model, tokenizer, device,
        this_problem,
        _SEARCH_DEPTH.value,
        _BEAM_SIZE.value,
        _OUT_FILE.value
    )

if __name__ == '__main__':
    app.run(main)

