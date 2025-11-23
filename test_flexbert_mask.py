import sys
import os
import torch
from transformers import AutoTokenizer

current_dir = os.getcwd()
if "ModernBERT" not in current_dir:
    sys.path.append(os.path.join(current_dir, "ModernBERT"))
else:
    sys.path.append(current_dir)

try:
    from src.flex_bert import create_flex_bert_mlm
except ImportError:
    sys.path.append("/teamspace/studios/this_studio/ModernBERT")
    from src.flex_bert import create_flex_bert_mlm

checkpoint_dir = "/teamspace/studios/this_studio/ModernBERT/downloaded_model/checkpoint-27501"
checkpoint_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
tokenizer_path = "/teamspace/studios/this_studio/ModernBERT/tokenizer/HUIT-BERT"

print(f"Loading tokenizer from {tokenizer_path}...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

print(f"Loading model from {checkpoint_dir}...")

model = create_flex_bert_mlm(
    pretrained_model_name=checkpoint_dir,
    pretrained_checkpoint=checkpoint_file,
    tokenizer_name=tokenizer_path,
)
print("Model loaded successfully.")

sentences = [
    "Hà Nội là thủ đô của Việt Nam.",
    "Tôi thích ăn phở bò vào buổi sáng.",
    "Hôm nay trời rất đẹp và trong xanh.",
    "Con mèo đang ngủ ngon lành trên ghế sofa.",
    "Chúng ta cần chung tay bảo vệ môi trường.",
    "Học lập trình Python rất thú vị và hữu ích.",
    "Bác Hồ là vị lãnh tụ kính yêu của dân tộc Việt Nam.",
    "Sông Hồng chảy qua trung tâm thành phố Hà Nội.",
    "Cây tre là biểu tượng kiên cường của làng quê Việt Nam.",
    "Mùa thu Hà Nội mang vẻ đẹp lãng mạn và cổ kính."
]

flex_bert_model = model.model
flex_bert_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
flex_bert_model.to(device)

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Model vocab size: {flex_bert_model.config.vocab_size}")

print(f"\nTesting on {len(sentences)} sentences...\n")

correct_count = 0
total_count = 0

for i, sent in enumerate(sentences):
    inputs = tokenizer(sent, return_tensors="pt")
    token_ids = inputs["input_ids"][0].tolist()
    
    seq_len = len(token_ids)
    if seq_len <= 3:
        continue 

    mask_idx = seq_len // 2
    
    special_ids = tokenizer.all_special_ids
    while token_ids[mask_idx] in special_ids and mask_idx > 0:
        mask_idx -= 1
        
    original_token_id = token_ids[mask_idx]
    original_word = tokenizer.decode([original_token_id])
    
    token_ids[mask_idx] = tokenizer.mask_token_id
    masked_sent_str = tokenizer.decode(token_ids)
    
    inputs["input_ids"] = torch.tensor([token_ids]).to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = flex_bert_model(**inputs)
        
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs[0]
            
    print(f"Logits shape: {logits.shape}")
    
    mask_logits = logits[mask_idx]
    top_k = 5
    top_probs, top_indices = torch.topk(torch.softmax(mask_logits, dim=-1), top_k)
    
    predicted_token_id = mask_logits.argmax(axis=-1).item()
    predicted_word = tokenizer.decode([predicted_token_id])
    
    is_correct = (predicted_token_id == original_token_id)
    if is_correct:
        correct_count += 1
    total_count += 1
    
    print(f"Sentence {i+1}:")
    print(f"  Original:  {sent}")
    print(f"  Masked:    {masked_sent_str}")
    print(f"  Target:    '{original_word}' (id: {original_token_id})")
    print(f"  Predicted: '{predicted_word}' (id: {predicted_token_id})")
    print(f"  Probability: {top_probs[0].item():.4f}")
    print("  Top 5 predictions:")
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx.item()])
        print(f"    {token}: {prob.item():.4f} (id: {idx.item()})")
    print(f"  Result:    {'CORRECT' if is_correct else 'INCORRECT'}")
    print("-" * 50)

accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
print(f"\nTotal Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
