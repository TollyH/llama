from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str = "llama-2-7b/",
    tokenizer_path: str = "tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: Optional[int] = None,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompt = input(">>> ").strip()

    result = generator.text_completion(
        [prompt],  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )[0]
    print(result["generation"])


if __name__ == "__main__":
    fire.Fire(main)
