from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import bitsandbytes as bnb

model1b_name = "meta-llama/Llama-3.2-1B-Instruct"
model3b_name = "meta-llama/Llama-3.2-3B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True                  # Enable 8-bit quantization    
)

training_args = TrainingArguments(
    output_dir="llama_finetune_mind2web",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if on GPU
    save_strategy="epoch",  # Automatically save checkpoints after each epoch
    report_to="none",
    push_to_hub=True
)

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the padding token as the EOS token if padding token is missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA adapters
    lora_config = LoraConfig(
        r=16,  # Low-rank dimension
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Target attention modules for adaptation
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    return model, tokenizer

def preprocess_data(examples):
    inputs = ["Task: " + task for task in examples["confirmed_task"]]
    targets = ["\n".join(actions) for actions in examples["action_reprs"]]

    # Tokenize inputs and targets with padding
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=64)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=64).input_ids
    model_inputs["labels"] = labels
    return model_inputs

def load_data():
    dataset_train = load_dataset("osunlp/Mind2Web")
    dataset_test_domain = load_dataset('json', data_files='./data/test_domain/*.json')
    dataset_test_task = load_dataset('json', data_files='./data/test_task/*.json')
    dataset_test_website = load_dataset('json', data_files='./data/test_website/*.json')
    return dataset_train, dataset_test_domain, dataset_test_task, dataset_test_website

def tokenize_data(dataset_train, dataset_test_domain, dataset_test_task, dataset_test_website):
    tokenized_train = dataset_train.map(preprocess_data, batched=True, batch_size=8)
    tokenized_test_domain = dataset_test_domain.map(preprocess_data, batched=True, batch_size=8)
    tokenized_test_task = dataset_test_task.map(preprocess_data, batched=True, batch_size=8)
    tokenized_test_website = dataset_test_website.map(preprocess_data, batched=True, batch_size=8)
    return tokenized_train, tokenized_test_domain, tokenized_test_task, tokenized_test_website

def fine_tune(model, training_args, train, eval, tokenizer):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train['train'],
        eval_dataset=eval['train'],
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.push_to_hub(commit_message="llama-3.2-1b-it-finetuned_mind2web")

def select_model():
    while True:
        try:
            user_input = int(input("Please choose the model\n1. 'meta-llama/Llama-3.2-1B-Instruct'\n2. 'meta-llama/Llama-3.2-3B-Instruct'\n"))
            if user_input == 1:
                return model1b_name
            elif user_input == 2:
                return model3b_name
            else:
                print("Invalid choice. Please choose again.")
        except ValueError:
            print("Please enter a valid integer (1 or 2).")

if __name__ == "__main__":
    model_name = select_model()
    print(f"model: {model_name}, Let's start fine-tuning!")
    model, tokenizer = load_model_and_tokenizer(model_name)
    dataset_train, dataset_test_domain, dataset_test_task, dataset_test_website = load_data()
    print("load_data DONE")
    tokenized_train, tokenized_test_domain, tokenized_test_task, tokenized_test_website = tokenize_data(dataset_train, dataset_test_domain, dataset_test_task, dataset_test_website)
    print("tokenization DONE")
    print(model)
    fine_tune(model, training_args, tokenized_train, tokenized_test_domain, tokenizer)
    print("SUCCESS!")