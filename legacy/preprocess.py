# Most of the code in this script was borrowed from https://github.com/LatentMindAI/perzonalized-ai-chatbot

import argparse
import json
import re
import os

def txt_to_json(txt_file_path, json_file_path, whatsapp_name):
    # Initialize the conversations list
    conversations = []
    # Regular expression pattern to match the chat lines
    pattern = r"\[(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}:\d{2})\s+[AP]M\] ([^:]+): (.+)"

    with open(txt_file_path, 'r', encoding='utf-8') as file:
        # Skip the first line (WhatsApp export header)
        file.readline()

        for line in file:
            match = re.match(pattern, line)
            if not match:
                continue

            sender = match.group(3).strip()
            content = match.group(4).strip()

            # Skip missed call notifications
            if "Missed video call" in content or "Missed voice call" in content:
                continue

            # Determine role
            from_field = "gpt" if whatsapp_name and whatsapp_name in sender else "human"

            # Group consecutive messages by the same sender
            if conversations and conversations[-1]["from"] == from_field:
                # Append to the existing entry
                conversations[-1]["value"] += "\n" + content
            else:
                # Start a new entry
                conversations.append({
                    "from": from_field,
                    "value": content
                })

    # Skip if too short to be useful
    if len(conversations) < 3:
        return

    # Split into chunks of up to 40 messages
    if len(conversations) > 40:
        for i in range(0, len(conversations), 40):
            partial = conversations[i:i+40]
            
            # Skip conversations that are too short
            if(len(conversations) < 3):
                return
            
            json.dump({"conversations": partial},
                      open(f"{json_file_path}_{i}.json", 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=4)


def process_folder(data_folder, output_folder, whatsapp_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(data_folder, filename)
            base = os.path.splitext(filename)[0]
            out_path = os.path.join(output_folder, base)
            txt_to_json(txt_path, out_path, whatsapp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert WhatsApp chat .txt files to JSON format."
    )
    parser.add_argument(
        "whatsapp_name", type=str,
        help="WhatsApp name to identify messages from that sender"
    )
    args = parser.parse_args()

    data_folder = "data/raw_data"
    output_folder = "data/preprocessed"
    process_folder(data_folder, output_folder, args.whatsapp_name)
