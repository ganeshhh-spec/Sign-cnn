import json

with open('results/metrics/four_way_evaluation.json') as f:
    d = json.load(f)

print('=== 1. HELD-OUT WEBCAM VALIDATION (23 IMAGES) ===')
orig_wc = d['evaluation_webcam_val']['original']
cand_wc = d['evaluation_webcam_val']['candidate']
print(f'Original  - Accuracy: {orig_wc["accuracy"]*100:.1f}%, Macro F1: {orig_wc["macro_f1"]:.4f}')
print(f'Candidate - Accuracy: {cand_wc["accuracy"]*100:.1f}%, Macro F1: {cand_wc["macro_f1"]:.4f}')

print('\nPer-class comparison on held-out webcam:')
for c in sorted(orig_wc['per_class'].keys()):
    o_acc = orig_wc['per_class'][c]['accuracy']
    c_acc = cand_wc['per_class'][c]['accuracy']
    tot = orig_wc['per_class'][c]['total_samples']
    diff = c_acc - o_acc
    flag = " [IMPROVED]" if diff > 0 else (" [REGRESSED]" if diff < 0 else " [SAME]")
    print(f'  {c:3s} ({tot} sample): Original={o_acc*100:5.1f}% | Candidate={c_acc*100:5.1f}%{flag}')

print('\nConfusion Matrix (Webcam Val):')
print('Labels:', orig_wc['confusion_matrix']['labels'])
print('Original CM:', orig_wc['confusion_matrix']['matrix'])
print('Candidate CM:', cand_wc['confusion_matrix']['matrix'])

print('\n=== 2. KAGGLE TEST SPLIT (13,050 IMAGES) ===')
orig_kg = d['evaluation_kaggle_test']['original']
cand_kg = d['evaluation_kaggle_test']['candidate']
print(f'Original  - Accuracy: {orig_kg["accuracy"]*100:.2f}%, Macro F1: {orig_kg["macro_f1"]:.4f}')
print(f'Candidate - Accuracy: {cand_kg["accuracy"]*100:.2f}%, Macro F1: {cand_kg["macro_f1"]:.4f}')

print('\n=== 3. PHYSICAL WEBCAM RECORDING ATTEMPTS (21 ATTEMPTS) ===')
phys = d['evaluation_physical_attempts']
print(f'Total attempts evaluated: {phys["total_attempts"]}')
print(f'Original Accuracy:  {phys["original_accuracy"]*100:.1f}%')
print(f'Candidate Accuracy: {phys["candidate_accuracy"]*100:.1f}%')
print('\nDetails per attempt:')
for att in phys['attempts']:
    o_mark = "OK" if att["original_correct"] else "FAIL"
    c_mark = "OK" if att["candidate_correct"] else "FAIL"
    print(f'  {att["file"]}: Intended={att["intended"]} ({att["hand"]:5s}) | Orig={att["original_pred"]} ({att["original_conf"]:.2f}) [{o_mark}] | Cand={att["candidate_pred"]} ({att["candidate_conf"]:.2f}) [{c_mark}]')
