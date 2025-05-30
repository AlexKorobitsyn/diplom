from gui.main_window import ModelApp
import tkinter as tk

def run_app():
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()