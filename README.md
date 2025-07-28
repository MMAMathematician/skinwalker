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
