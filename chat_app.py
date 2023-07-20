import os
from typing import Optional

import fire

from llama import Llama

QUIT_MESSAGES = ["GOODBYE", "FAREWELL", "BYE", "SEE YA", "SEE YOU"]

DEFAULT_SYSTEM = """\
You are an assistant. Always answer as helpfully as possible.

If a question does not make any sense, or is not factually coherent, explain \
why instead of answering something not correct. If you don't know the answer \
to a question, please don't share false information."""


def main(
    ckpt_dir: str = "llama-2-7b-chat/",
    tokenizer_path: str = "tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    system_text: str = DEFAULT_SYSTEM,
):
    major_separator = '=' * os.get_terminal_size().columns
    minor_separator = '-' * os.get_terminal_size().columns

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    print(
        "\nWelcome to the chat app. To enable multiline mode for a message, "
        "type a single '*' and press enter. To send a message while in "
        "multiline mode, send a single '^' and press enter."
    )

    if system_text == DEFAULT_SYSTEM:
        system_text = input(
            "If desired, enter a system message to use for this chat > "
        )
        if system_text.strip() == "":
            system_text = DEFAULT_SYSTEM

    current_chat = [
        {
            "role": "system",
            "content": system_text,
        }
    ]

    print(major_separator)

    user_message = ""
    while user_message.upper() not in QUIT_MESSAGES:
        multiline = False
        user_message = ""

        print("You: ", end="", flush=True)
        while user_message == "" or multiline:
            message_line = input().strip()
            if message_line == "^":
                break
            if message_line == "*":
                multiline = True
                continue
            user_message += message_line + '\n'
        user_message = user_message.strip()
        current_chat.append({"role": "user", "content": user_message})

        print(f"{minor_separator}\nChatBot: ...", end="", flush=True)
        result = generator.chat_completion(
            [current_chat],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        generated_content = result["generation"]["content"]
        print(f"\rChatBot: {generated_content}\n{minor_separator}")
        current_chat.append(
            {"role": "assistant", "content": generated_content}
        )


if __name__ == "__main__":
    fire.Fire(main)
