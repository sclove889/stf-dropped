# -*- coding: utf-8 -*-
"""看清 CSDS 的真实结构，并自动猜出 7 个字段名。

用法：  python 00_spec/inspect_csds.py
"""
import json
import os

PATH = '01_raw/csds/test.json'

if not os.path.exists(PATH):
    print(f'❌ 找不到 {PATH}')
    print('   请先完成步骤 2：把 CSDS 的 train.json / val.json / test.json')
    print('   下载到 01_raw/csds/ 目录下')
    raise SystemExit(1)

data = json.load(open(PATH, encoding='utf-8'))

# 顶层可能是 list，也可能是 dict
if isinstance(data, dict):
    print(f'⚠️  顶层是 dict，key 有: {list(data.keys())[:10]}')
    for k, v in data.items():
        if isinstance(v, list) and v:
            print(f'   → 取 data["{k}"] 作为对话列表\n')
            data = v
            break

sample = data[0]
print('=' * 72)
print(f'  对话总数: {len(data)}')
print('=' * 72)


# ────────────────────────── 画结构树 ──────────────────────────
def show(obj, prefix='', depth=0, max_depth=3):
    if depth > max_depth:
        print(prefix + '…')
        return
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            print(f'{prefix}{k}   ← 对象')
            show(v, prefix + '    ', depth + 1)
        elif isinstance(v, list):
            print(f'{prefix}{k}   ← 列表，共 {len(v)} 项')
            if v:
                if isinstance(v[0], dict):
                    print(f'{prefix}    └─ 每一项是对象，内容：')
                    show(v[0], prefix + '       ', depth + 1)
                else:
                    s = repr(v[:5])
                    print(f'{prefix}    └─ 前几项 = {s[:70]}')
        else:
            s = repr(v)
            s = s[:66] + '…' if len(s) > 68 else s
            print(f'{prefix}{k}   = {s}')


print('\n【第一条对话的结构】\n')
show(sample)


# ────────────────────────── 自动猜字段名 ──────────────────────────
HINTS = {
    'K_DIALOG':   (['dialog', 'utteranc', 'session', 'convers', 'turns', 'content'],
                   '对话内容列表'),
    'K_QA':       (['qa', 'pair'], 'QA 对列表'),
    'K_INTENT':   (['intent', 'topic', 'categ', 'label', 'class'], '分类标签'),
    'K_PAIN':     (['quesumm', 'quesum', 'question', 'usersumm'], '用户问题摘要'),
    'K_SOLUTION': (['anssummlong', 'anssumm', 'answer', 'agentsumm'], '客服回答摘要'),
    'K_PAIN_IDS': (['quesummuttid', 'quesummid', 'queid'], '问题证据下标'),
    'K_SOL_IDS':  (['anssummlonguttid', 'anssummuttid', 'ansid'], '回答证据下标'),
    'K_UTT_TEXT': (['utteranc', 'text', 'sentence', 'content'], '单句文字'),
    'K_UTT_ROLE': (['speaker', 'role', 'who', 'agent'], '说话人'),
}

found = {}      # key名 -> (所在层, 样例值)
LIST_ONLY = {'K_DIALOG', 'K_QA'}   # 这两个必须是列表，挡掉 DialogueID 这种误匹配
is_list = {}                       # key名 -> 值是不是列表


def collect(obj, where='顶层'):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in found:
                s = repr(v)
                found[k] = (where, s[:50] + '…' if len(s) > 52 else s)
                is_list[k] = isinstance(v, list)
            if isinstance(v, dict):
                collect(v, f'{where} > {k}')
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                collect(v[0], f'{where} > {k}[]')


collect(sample)

print('\n' + '=' * 72)
print('  自动猜测结果（★ = 猜的，必须对照上面的结构树核对）')
print('=' * 72 + '\n')

guessed = {}
for var, (pats, desc) in HINTS.items():
    hit = None
    for p in pats:
        for k in found:
            if var in LIST_ONLY and not is_list.get(k):
                continue
            if p in k.lower().replace('_', ''):
                hit = k
                break
        if hit:
            break
    guessed[var] = hit
    if hit:
        where, val = found[hit]
        print(f"★ {var:<12} = '{hit}'")
        print(f"     {desc}，位于 {where}")
        print(f"     样例值 → {val}")
    else:
        print(f"？ {var:<12} = 没猜出来  （{desc}）—— 请自己从结构树里找")
    print()

print('=' * 72)
print('  下面这段就是要填进 06_data/convert.py 顶部的内容')
print('=' * 72 + '\n')

lines = []
for var, hit in guessed.items():
    lines.append(f"{var:<12} = '{hit}'" if hit else f"{var:<12} = '???'   # ← 自己填")
block = '\n'.join(lines)
print(block)

os.makedirs('00_spec', exist_ok=True)
with open('00_spec/字段名.txt', 'w', encoding='utf-8') as f:
    f.write('# 由 inspect_csds.py 自动猜出，需肉眼核对后使用\n')
    f.write('# 核对方法：对照结构树，确认每个名字确实装着注释里说的东西\n\n')
    f.write(block + '\n')

print('\n→ 已写入 00_spec/字段名.txt')
print('\n⚠️  下一步是核对，重点看这三个的「样例值」——')
print('   K_INTENT   应该像「配送周期」这种分类名，不是一整句话')
print('   K_PAIN     应该像「用户询问…」这种一句话摘要')
print('   K_SOLUTION 应该像「客服让用户…」这种一句话摘要')
print('   对不上就手动改 00_spec/字段名.txt，改完再做步骤 3.5。')

with open('01_raw/sample_pretty.json', 'w', encoding='utf-8') as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)
print('\n→ 完整的第一条已存到 01_raw/sample_pretty.json（结构树看不清时打开它）')
