# Skinwalker
Tool that steals a Discord account and uses a large language model to pretend to be them.

## Usage

```
usage: skinwalker.py [-h] -k KEY [-u USERNAME] -p PROMPT [-c]

Skinwalker: Tool that steals a Discord account and uses a large language model to pretend to be them.

options:
  -h, --help            show this help message and exit
  -k KEY, --key KEY     filename containing OpenAI API key
  -u USERNAME, --username USERNAME
                        filename containing usernames of accounts the LLM will talk to (will attempt to chat to everyone if no value is supplied)
  -p PROMPT, --prompt PROMPT
                        filename containing the system prompt for the LLM
  -c, --captcha         solves Discord captcha using a multi-modal model
```

## Note About Legacy Items
Every file in the legacy folder comes from my attempts to train and use the LLama-3.1-8B model.
None of these files are used in the final iteration of the project.
They soley exist as a proof of work.
Some of these files could not be uploaded to the repository due to privacy reasons.
The model and the weights also could not be uploaded due to the large size of the files.
