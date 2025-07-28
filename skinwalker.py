import argparse
from start_site import run_app

def main():
    parser = argparse.ArgumentParser(
        description='Skinwalker: Tool that steals a Discord account and uses a large language model to pretend to be them.'
    )
    
    parser.add_argument(
        '-k', '--key',
        help='filename containing OpenAI API key',
        required=True
    )
    
    parser.add_argument(
        '-u', '--username',
        help='filename containing usernames of accounts the LLM will talk to (will attempt to chat to everyone if no value is supplied)',
        default=None
    )
    
    parser.add_argument(
        '-p', '--prompt',
        help='filename containing the system prompt for the LLM',
        required=True
    )
    
    parser.add_argument(
        '-c', '--captcha',
        help='solves Discord captcha using a multi-modal model',
        action='store_true'
    )
    
    args = parser.parse_args()
    
    gpt_token = open(args.key).read()
    
    if args.username:
        usernames = open(args.username).read().splitlines()
    else:
        usernames = []
        
    system_prompt = open(args.prompt, encoding='utf-8').read()
    automate_captcha = args.captcha
    
    run_app(gpt_token, system_prompt, usernames, automate_captcha)
    
if __name__ == '__main__':
    main()