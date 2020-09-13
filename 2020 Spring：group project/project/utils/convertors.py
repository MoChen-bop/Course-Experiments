import torch


def captions_to_tensor(captions, device):

    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} '
    text_t = torch.zeros(len(captions), len(alphabet), 201, device=device)
    for i, caption in enumerate(captions):
        for j, tok in enumerate(caption.lower()):
            if j > 201:
            	break
            text_t[i, alphabet.index(tok), j] = 1
    return text_t


if __name__ == '__main__':
    test_text = ["This is test text.", "This is another test text."]
    text_tensor = captions_to_tensor(test_text, 'cpu')
    print(test_text)
    print(text_tensor.shape)