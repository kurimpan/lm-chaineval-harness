from evaluate import load

class Evaluator:
    def __init__(self, metric):
        self.metric = metric

    def calculate(self, data, keymap, record):
        if self.metric.name == "code_eval":
            # テストケースと予測値のリストを準備
            test_cases = [d[keymap.get('result')] for d in data]
            candidates = [[d['model_output']] for d in data]
            # 各候補に対してpass@kを計算し、平均を取得
            scores = []
            for reference, candidate in zip(test_cases, candidates):
                # ここでは各ペアに対して独立してpass@kを計算
                pass_at_k = self.metric.compute(references=[reference], predictions=[candidate], k=[1])
                # pass@kのスコアをリストに追加
                scores.append(pass_at_k['pass@1'])
            # 平均値を計算
            average_score = sum(scores) / len(scores)
            score = average_score
        elif self.metric.name == "accuracy":
            # accuracy用の処理...
            pass
        else:
            # 他のメトリクス用の処理...
            pass

        # スコアを記録
        record['score'] = score
        return record

def load_evaluator(metric_name):
    # 指定されたメトリック名に基づいてメトリックをロードする
    metric = load(metric_name)
    return Evaluator(metric)
