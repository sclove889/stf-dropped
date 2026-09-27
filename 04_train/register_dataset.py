# -*- coding: utf-8 -*-
"""把自定义数据集注册进 LLaMA-Factory 的 data/dataset_info.json。

在 AutoDL 上跑：
    python register_dataset.py huashu_smoke /root/autodl-tmp/work/data/smoke.jsonl
    python register_dataset.py huashu       /root/autodl-tmp/work/data/huashu.jsonl

为什么要脚本而不是手改 JSON：
  dataset_info.json 有 100 多个条目，手改容易漏逗号 / 打错引号。
  改坏了 LLaMA-Factory 报的是 JSONDecodeError，和你的数据毫无关系，会查错方向。
"""
import json
import os
import shutil
import sys

LF = os.environ.get('LF_HOME', '/root/autodl-tmp/LLaMA-Factory')


def main():
    if len(sys.argv) != 3:
        sys.exit(f'用法：python {os.path.basename(__file__)} <数据集名> <jsonl 绝对路径>')
    name, path = sys.argv[1], os.path.abspath(sys.argv[2])

    if not os.path.exists(path):
        sys.exit(f'❌ 找不到数据文件 {path}')

    info = os.path.join(LF, 'data', 'dataset_info.json')
    if not os.path.exists(info):
        sys.exit(f'❌ 找不到 {info}\n   LLaMA-Factory 装在别处？设一下 LF_HOME 环境变量')

    # 先校验数据本身，别把坏数据注册进去
    with open(path, encoding='utf-8') as f:
        rows = []
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f'❌ 第 {i} 行不是合法 JSON：{e}')
            for k in ('instruction', 'output'):
                if not r.get(k):
                    sys.exit(f'❌ 第 {i} 行缺 "{k}" 或为空')
            rows.append(r)
    if not rows:
        sys.exit('❌ 文件是空的')

    d = json.load(open(info, encoding='utf-8'))
    existed = name in d
    d[name] = {'file_name': path, 'formatting': 'alpaca',
               'columns': {'prompt': 'instruction', 'query': 'input', 'response': 'output'}}

    if not os.path.exists(info + '.bak'):
        shutil.copy(info, info + '.bak')       # 只备份一次，保住出厂版本
    with open(info, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

    lens = sorted(len(r['instruction']) + len(r.get('input', '')) + len(r['output']) for r in rows)
    rec = max(2048, (int(lens[-1] / 1.4) * 13 // 10 // 256 + 1) * 256)

    print(f'✅ {"更新" if existed else "注册"}数据集「{name}」→ {path}')
    print(f'   条数        {len(rows)}')
    print(f'   字符数      最短 {lens[0]}  中位 {lens[len(lens)//2]}  最长 {lens[-1]}')
    print(f'   建议 cutoff_len ≥ {rec}   ← 填进 yaml，宁大勿小')
    print(f'\n   yaml 里写：dataset: {name}')
    print(f'   原始 dataset_info.json 已备份到 {info}.bak')


if __name__ == '__main__':
    main()
