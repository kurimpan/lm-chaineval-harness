from evaluate import load

class Evaluator:
    def __init__(self, metric):
        self.metric = metric

    def calculate(self, data, keymap, record):
        # ここでメトリックに応じて適切なデータ構造を準備する
        # ...

        # F1スコアの計算のための条件分岐を追加
        if self.metric.name == "f1":
            # F1スコアの計算に必要な正解ラベルと予測ラベルのリストを準備
            references = [d[keymap.get('result')] for d in data]
            predictions = [d['model_output'] for d in data]
            # F1スコアを計算
            score = self.metric.compute(predictions=predictions, references=references)["f1"]
            # `score` には通常、precision, recall, f1 のキーが含まれている
            #f1_score = score['f1']
            #score = f1_score

        else:
            pass

        # スコアをレコードに格納
        #record['score'] = score  # コメントアウトされている場合は不要かもしれない
        return score

def load_evaluator(metric_name):
    # 指定されたメトリック名に基づいてメトリックをロードする
    metric = load(metric_name)
    return Evaluator(metric)

# 使用例:
# evaluator = load_evaluator("F1-score")
# f1_score = evaluator.calculate(data, keymap, record)
