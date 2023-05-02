import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from button_detector import ButtonDetector
from data_treiner import train_button_detector
from utils import ButtonDataset, capture_game_window, detect_buttons


class ConvertToRGB:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


data_transforms = {
    'treino': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        ConvertToRGB(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validacao': transforms.Compose([
        transforms.Resize((128, 128)),
        ConvertToRGB(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Substitua os caminhos abaixo pelos caminhos das suas pastas de treinamento e validação
train_dir = 'data_dir/treino'
val_dir = 'data_dir/validacao'

# Carregue os conjuntos de dados de treinamento e validação
train_data = ButtonDataset(train_dir, transform=data_transforms['treino'])
val_data = ButtonDataset(val_dir, transform=data_transforms['validacao'])

# Crie DataLoaders para os conjuntos de dados de treinamento e validação
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)

# Carregue o modelo treinado
button_detector = ButtonDetector()

train_button_detector(button_detector, train_loader, val_loader)

button_detector.load_state_dict(torch.load('button_detector.pth'))
button_detector.eval()

# Substitua 'nome_janela_jogo' pelo nome da janela do seu jogo
game_window = 'Yu-Gi-Oh! DUEL LINKS'

# Capture a tela do jogo
game_screen = capture_game_window(game_window)

# Converta a imagem capturada para PIL.Image
game_screen_pil = Image.fromarray(game_screen)

# Verifique se a imagem contém um botão
if detect_buttons(game_screen_pil, button_detector):
    print("Botão detectado")
else:
    print("Botão não detectado")

# Adicione o código para clicar nos botões detectados aqui
