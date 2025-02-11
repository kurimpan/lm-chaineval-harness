# evaluators.py
from evaluate import load

class Evaluator:
    def __init__(self, metric):
        self.metric = metric

    def calculate(self, data, keymap, record):
        # ここでメトリックに応じて適切なデータ構造を準備する
        # 例えば、code_evalはテストケースと予測をリストで受け取る
        # BLEUやaccuracyは参照と予測を両方ともリストで受け取るが、
        # 形式は異なる可能性がある

        # 以下はcode_eval用のデータ準備のサンプルです
        if self.metric.name == "code_eval":
            test_cases = [d[keymap.get('result')] for d in data]
            candidates = [[d['model_output']] for d in data]
            # Compute pass@k for each candidate
            score = self.metric.compute(references=test_cases, predictions=candidates, k=[1])['pass@1']
            #pass_at_k, results = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
            #print(pass_at_k)
            #{'pass@1': 1.0}
            # For this example, we are considering pass@1

        elif self.metric.name == "accuracy":
            # 正解データと予測データのリストを準備
            references = [d[keymap.get('result')] for d in data]
            predictions = [d['model_output'] for d in data]
            # accuracy スコアを計算
            score = self.metric.compute(predictions=predictions, references=references)['accuracy']

        elif self.metric.name == "f1":
            # F1スコアの計算に必要な正解ラベルと予測ラベルのリストを準備
            references = [d[keymap.get('result')] for d in data]
            predictions = [d['model_output'] for d in data]
            # F1スコアを計算
            score = self.metric.compute(predictions=predictions, references=references)["f1"]

        else:
            pass

        #record['score'] = score
        return score

def load_evaluator(metric_name):
    # 指定されたメトリック名に基づいてメトリックをロードする
    metric = load(metric_name)
    return Evaluator(metric)



