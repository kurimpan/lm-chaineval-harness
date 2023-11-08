# evaluators.pyからEvaluatorクラスをインポートします。
from memo2_codeeval import load_evaluator
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# テスト用のデータセットを作成します。
# この例では、2つのエラーコードと修正コードのペアを含んでいます。
test_data = [
    {
        'result': "assert add(2,3)==5",
        'model_output': "def add(a, b): return a+b"
    },
    {
        'result': "assert multiply(2,3)==6",
        'model_output': "def multiply(a, b): return a*b"
    }
]

# keymapを作成します。この例では、'result'と'model_output'キーをそのまま使用します。
keymap = {'result': 'result', 'model_output': 'model_output'}

# code_evalメトリックを使ってEvaluatorをロードします。
evaluator = load_evaluator('code_eval')

# 空のレコードを作成します。これにスコアが格納されます。
record = {}

# calculate関数を使用してスコアを計算します。
score_record = evaluator.calculate(test_data, keymap, record)

# 結果を表示します。期待される出力は、pass@1のスコアが1つ目のテストケースで1.0、
# 2つ目で0.0となり、平均は0.5となることです。
print("Calculated Score Record:", score_record)
