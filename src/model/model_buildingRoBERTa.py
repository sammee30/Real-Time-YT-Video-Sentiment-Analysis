import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class TextDataset(Dataset):
    """Custom Dataset for text data."""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_model(model, dataloader, optimizer, device):
    """Train the RoBERTa model."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    logger.debug(f"Training loss: {avg_loss}")


def evaluate_model(model, dataloader, device):
    """Evaluate the model and print a classification report."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, digits=4)
    logger.info(f"Evaluation Report:\n{report}")
    print(report)


def main():
    try:
        # Load dataset
        root_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(root_dir, 'data/interim/train_processed.csv')
        df = pd.read_csv(dataset_path)

        # Extract texts and labels
        texts = df['clean_comment'].tolist()
        labels = df['category'].tolist()  # Ensure `category` contains numeric labels

        # Parameters
        max_len = 128
        batch_size = 16
        learning_rate = 2e-5
        num_epochs = 3

        # Initialize tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(set(labels)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Create dataset and dataloader
        dataset = TextDataset(texts, labels, tokenizer, max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            logger.debug(f"Epoch {epoch + 1}/{num_epochs}")
            train_model(model, dataloader, optimizer, device)

        # Save model
        model_save_path = os.path.join(root_dir, 'roberta_model.pt')
        torch.save(model.state_dict(), model_save_path)
        logger.debug(f"Model saved at {model_save_path}")

        # Evaluate the model
        evaluate_model(model, dataloader, device)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == '__main__':
    main()
