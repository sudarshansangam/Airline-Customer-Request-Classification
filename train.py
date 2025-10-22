from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd

# 1. Load or prepare your dataset
# Example dataframe
df = pd.read_csv("airline_support_data.csv")

# Encode the label as integers
df['label_id'] = df['label'].astype('category').cat.codes
label2id = {label: idx for idx, label in enumerate(df['label'].astype('category').cat.categories)}
id2label = {idx: label for label, idx in label2id.items()}

# 2. Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# 3. Tokenization
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize_batch(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=64)
dataset = dataset.map(tokenize_batch, batched=True)
dataset = dataset.remove_columns(['label'])
dataset = dataset.rename_column("label_id", "label")


# 4. Prepare model
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 5. Split dataset
split = dataset.train_test_split(test_size=0.1)
train_dataset = split['train']
eval_dataset = split['test']

# 6. Training parameters
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    logging_steps=10,
    save_strategy='epoch',
    learning_rate=5e-5
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 8. Train
trainer.train()

# 9. Save your model
trainer.save_model("./airline_classifier_model")
tokenizer.save_pretrained("./airline_classifier_model")
