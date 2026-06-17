import tkinter as tk
from PIL import Image
from redboxflip.gui.editor import ShotEditor
from redboxflip.models import ShotResult, Face, Settings

root = tk.Tk(); root.withdraw()
img = Image.new("RGB", (600, 400), "white")
shot = ShotResult(input_path="demo.jpg", face=Face.FRONT, title="Demo")
ed = ShotEditor(root, img, shot, Settings())
print("RESULT:", ed.get())
