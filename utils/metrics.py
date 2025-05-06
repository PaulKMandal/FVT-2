import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def compute_metrics(model, loader, task, metrics_list, annotation_json=None):
    """
    Evaluate the model on the given loader.
    For classification, compute accuracy, precision, recall.
    For detection, compute COCO mAP and IoU.
    """
    model.eval()
    device = next(model.parameters()).device

    # Unwrap PEFT model if needed
    call_model = model.base_model if isinstance(model, PeftModel) else model

    if task == 'classification':
        preds, labels = [], []
        for batch in loader:
            # Only forward pixel_values and labels
            inputs = {'pixel_values': batch['pixel_values'].to(device)}
            if 'labels' in batch:
                gold = batch['labels'].to(device)
            else:
                gold = batch['label'].to(device)

            with torch.no_grad():
                outputs = call_model(**inputs)
                # Some models return just logits
                logits = getattr(outputs, 'logits', outputs)
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().numpy())
            labels.extend(gold.cpu().numpy())

        results = {}
        if 'accuracy' in metrics_list:
            results['accuracy'] = accuracy_score(labels, preds)
        if 'precision' in metrics_list:
            results['precision'] = precision_score(labels, preds, average='macro', zero_division=0)
        if 'recall' in metrics_list:
            results['recall'] = recall_score(labels, preds, average='macro', zero_division=0)
        return results

    # Detection: needs COCO annotations JSON
    coco_gt = COCO(annotation_json)
    pred_list = []
    for batch in loader:
        imgs = batch['pixel_values'].to(device)
        with torch.no_grad():
            outputs = call_model(images=imgs)
        # Convert model outputs to COCO format
        pred_list.extend(convert_to_coco(outputs))

    coco_dt = coco_gt.loadRes(pred_list)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        'mAP': coco_eval.stats[0],
        'IoU': coco_eval.stats[1]
    }


def convert_to_coco(outputs):
    """Convert model detection outputs to COCO JSON format list."""
    # TODO: implement conversion based on model output structure
    return []
