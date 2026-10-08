import vital
from pathlib import Path
import torch
import torch.nn as nn

from parts import tokenizer
from parts.training_data import encode_prompt

from parts.training_data import encode_prompt

TOKENISER_DIR = Path(__file__).resolve().parent / "checkpoints" / "latest"/ "tokenizer.json"

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters = torch.load(
        Path(__file__).resolve().parent / "checkpoints" / "latest" / "model.pt",
        map_location=device, weights_only=True,
    )
    if not parameters:
        raise ValueError("模型参数为空，请先完成训练并保存有效权重。")

    model = vital.Vital(parameters["vocab_chart.weight"].shape[0]).to(device)
    model.load_state_dict(parameters)
    tokeniser = vital.load_bpe(TOKENISER_DIR)
    model.eval()

    while True:
        prompt = input("user: ")
        if prompt == "exit":
            break

        tokenised = encode_prompt(tokeniser, prompt)
        tensor = torch.tensor([tokenised], dtype=torch.long, device = device)
        responses : list[int] = []
        with torch.no_grad():
            while True:
                next_id = model.forward(tensor).argmax(dim = -1).item()
                if next_id == tokeniser.eos_token_id:
                    output = tokeniser.decode(responses)
                    print(f'Vital:{output}')
                    break
                responses.append(next_id)
                tokenised.append(next_id)
                tensor = torch.tensor([tokenised], dtype=torch.long, device = device)

if __name__ == "__main__":
    main()
