a = "tp: 185, fp: 182, fn: 10"
b = "tp: 63, fp: 68, fn: 3"


def get_val(line):
    line = line.split(",")
    tp = int(line[0].split(":")[-1].strip())
    fp = int(line[1].split(":")[-1].strip())
    fn = int(line[2].split(":")[-1].strip())
    return tp, fp, fn


tp_a, fp_a, fn_a = get_val(a)
tp_b, fp_b, fn_b = get_val(b)
tp = tp_a + tp_b
fp = fp_a + fp_b
fn = fn_a + fn_b
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f"{tp} & {fp} & {fn} & {f1 * 100:.2f}")
