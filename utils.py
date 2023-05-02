import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
import torch
import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def capture_game_window(window_name):
    try:
        # Encontre a janela do jogo pelo título da janela
        game_window = gw.getWindowsWithTitle(window_name)[0]

        # Obtenha as coordenadas da janela do jogo
        x, y, width, height = game_window.left, game_window.top, game_window.width, game_window.height

        # Capture a tela da janela do jogo
        screenshot = pyautogui.screenshot(region=(x, y, width, height))

        # Converta a imagem capturada para uma matriz NumPy no formato BGR
        screenshot_np = np.array(screenshot)
        game_screen = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

        return game_screen

    except IndexError:
        print(f'Janela com o título "{window_name}" não encontrada.')
        return None

def detect_buttons(image, model, threshold=0.5):
    # Defina a transformação necessária para a imagem
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Aplique a transformação na imagem
    image_tensor = data_transforms(image).unsqueeze(0)

    # Certifique-se de que o modelo está em modo de avaliação
    model.eval()

    # Realize a inferência e obtenha a probabilidade
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()

    # Verifique se a probabilidade é maior ou igual ao limite
    if prob >= threshold:
        return True  # botão detectado
    else:
        return False  # botão não detectado

#def click_button(window, button_box):

class ButtonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if "button" in os.path.basename(image_path):
            label = 1
        else:
            label = 0

        if self.transform:
            image = self.transform(image)

        return image, label
