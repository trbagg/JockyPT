# JockyPT

A python script designed to finetune using peft lora, and inference the trained llm as a Discord bot.

## Installation
1. Clone the repository
    ```bash
    git clone https://github.com/trbagg/JockyPT.git
    ```
2. Navigate to the directory
    ```bash
    cd ./jockypt
    ```
3. Install python requirements
    ```bash
    pip install requirements.txt
    ```
4. Install CUDA compatible for your drivers.

5. Install torch compiled for your CUDA version according to [PyTorch](https://pytorch.org/get-started/locally/)

## Usage
### Finetuning
1. Compile the dataset messages into a json called scraped-dataset:

2. Run the dataset json formatter.
    ```bash
    python ./content/formatted.py
    ```

3. Create/verify a `.env` file exists and contains valid key-value pairs for: TENOR_API_KEY. 

4. Run the dataset automated converter. 
    ```bash
    python ./content/automated.py
    ```
    This will output you're training json called "final_output.json"
    
    
5. Run the finetuning script. Note: Default hyperparameters may need to be adjusted according to the dataset.
    ```bash
    python ./content/finetunejockypt.py
    ```
    A prompt to input a wandb api key will show up if wandb does not have existing credentials.
    This will output several files and folders, as well as a wandb printout link for reading training logs.
    The checkpoints will be available in the `./jockypt-pt/` folder, containing the safetensor and various training files alongside.

### Inferencing

1. Create/verify a `.env` file exists and contains valid key-value pairs for: DISCORD_TOKEN, DISCORD_GUILD, and TENOR_API_KEY. 
    
2. Run the finetuning script. Note: The default context of 8192 may be too high for some consumer GPUs.
    ```bash
    python ./content/jocky_bot.py
    ```

3. Various commands are available.  
    `!save` can be used to save the existing conversation context to the `./conversations/` folder, as a json file with the current timestamp.  
    `!dump` to purge the existing context history.  
    `!ignore` to skip over the current message when inferencing.  
    `!shutdown`' `to gracefully shutdown via messaging.  