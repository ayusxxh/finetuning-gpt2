# finetuning-gpt2
A conversational model fine tuned over GPT-2 in order to generate Salesman Responses to customer concerns.

For this assignment, I generated a small synthetic dataset comprising realistic dialogue scenarios to reflect critical thinking tasks, particularly focusing on decision-making in a sales context. Using AI tools like Claude and ChatGPT, I created 35 sample conversations that capture typical customer concerns and the corresponding responses from sales representatives. Each conversation was designed to mirror real-world interactions, emphasizing the sales reps' ability to address objections, offer solutions, and engage effectively with customers. The pre-trained GPT-2 model from Hugging Face’s transformers library was selected for this task due to its robust language generation capabilities. The fine-tuning process aimed to enhance the model’s performance in generating relevant and accurate sales responses, ultimately improving its practical application in sales environments.

Fine-Tuning GPT-2 for Sales Representative Response Generation

Setup and Dependencies:
The project uses the Transformers library from Hugging Face, which provides pre-trained models and tools for fine-tuning.
Key imports include GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, and TrainingArguments.

Data Preparation:
Data is sourced from the JSON file ('data.json') containing customer queries and sales representative responses, generate using Clude and ChatGPT 3.5 turbo.

Model and Tokenizer Initialization:
The pre-trained GPT-2 model ("gpt2") is loaded using GPT2LMHeadModel.from_pretrained().
The corresponding tokenizer is initialized with GPT2Tokenizer.from_pretrained().

Dataset Creation:
The prepared data is written to a text file ('prepared_data.txt').
A TextDataset is created using this file, the tokenizer, and a block_size of 128 tokens.
This dataset structure is optimized for language model training.

Data Collator:
DataCollatorForLanguageModeling is used to create batches of data efficiently.
It handles padding and creates attention masks automatically.

Training Configuration:
TrainingArguments are set with the following key parameters:

output_dir: "./results"
overwrite_output_dir: True
num_train_epochs: 60
per_device_train_batch_size: 4
save_steps: 10,000
save_total_limit: 2

Trainer Initialization:
A Trainer is created with the model, training arguments, data collator, and training dataset.

Training Process:
The training is initiated with trainer.train().
This process fine-tunes the GPT-2 model on the sales conversation dataset.

Model Saving:
After training, the fine-tuned model and tokenizer are saved to "./fine_tuned_gpt2".
This allows for easy loading and use of the model in future sessions.

Response Generation Function:

generate_responses(): This function takes a list of prompts, tokenizes them, and generates responses using the specified model. It controls the generation process to avoid repetition and adjusts the temperature for diverse outputs.
Test Prompts:

A list of test prompts addresses common customer concerns about product customization, reliability, reporting capabilities, mobile features, and third-party integrations.
Generating and Comparing Responses:

The function generate_responses() is used to generate responses to the test prompts from both the base and fine-tuned models.

The fine-tuned model's response to the training concern prompt (from earlier) shows it handles customer objections more effectively by addressing specific concerns, asking relevant questions, and highlighting product benefits.

Calculating Perplexity for Both Models:

The function calculate_perplexity() is used to compute the perplexity for both the base GPT-2 model and the fine-tuned GPT-2 model using the test prompts.
Output:

The perplexity values for both models are printed, showing that the fine-tuned model has a lower perplexity, indicating it better predicts the given prompts compared to the base model.
