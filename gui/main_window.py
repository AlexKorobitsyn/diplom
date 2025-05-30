# gui/main_window.py
import tkinter as tk
from tkinter import ttk
from models.solow import SolowModel
from models.RCK import RCKModel
from models.ak import AKModel


class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Модели экономического роста")

        self.model_var = tk.StringVar(value="Solow")

        self.models = {
            'Solow': SolowModel(),
            'RCK': RCKModel(),
            'AK': AKModel()
        }

        self.params = {
            'alpha': tk.DoubleVar(value=0.5),
            'delta': tk.DoubleVar(value=0.05),
            'n': tk.DoubleVar(value=0.01),
            'g': tk.DoubleVar(value=0.02),
            'rho': tk.DoubleVar(value=0.02),
            'theta': tk.DoubleVar(value=1.5),
            's': tk.DoubleVar(value=0.3),
            'k0_1': tk.DoubleVar(value=1.2),
            'k0_2': tk.DoubleVar(value=7.5),
            'k0_3': tk.DoubleVar(value=31.0),
            'k0_4': tk.DoubleVar(value=42.0)
        }

        self.create_widgets()
        self.model_var.trace_add('write', self.update_ui)
        self.update_ui()

    def create_widgets(self):
        ttk.Label(self.root, text="Модель:").grid(row=0, column=0, sticky='w')
        model_menu = ttk.Combobox(self.root, textvariable=self.model_var,
                                values=['Solow', 'RCK', 'AK'], state='readonly')
        model_menu.grid(row=0, column=1, sticky='ew')
        self.param_frame = ttk.Frame(self.root)
        self.param_frame.grid(row=1, column=0, columnspan=2, sticky='nsew')
        ttk.Label(self.root, text="Начальные условия капитала:").grid(row=2, column=0, sticky='w', pady=(10, 0))
        self.k0_frame = ttk.Frame(self.root)
        self.k0_frame.grid(row=3, column=0, columnspan=2, sticky='nsew')

        for i in range(4):
            ttk.Label(self.k0_frame, text=f"k0_{i + 1}:").grid(row=0, column=i * 2, padx=(5, 0))
            ttk.Entry(self.k0_frame, textvariable=self.params[f'k0_{i + 1}'], width=8).grid(row=0, column=i * 2 + 1)
        ttk.Button(self.root, text="Построить график", command=self.run_model).grid(row=4, column=0, columnspan=2,
                                                                                  pady=10)

    def update_ui(self, *args):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        model = self.model_var.get()
        row = 0
        ttk.Label(self.param_frame, text="α (Доля капитала):").grid(row=row, column=0, sticky='e')
        ttk.Entry(self.param_frame, textvariable=self.params['alpha'], width=8).grid(row=row, column=1)
        row += 1

        ttk.Label(self.param_frame, text="δ (Амортизация):").grid(row=row, column=0, sticky='e')
        ttk.Entry(self.param_frame, textvariable=self.params['delta'], width=8).grid(row=row, column=1)
        row += 1

        ttk.Label(self.param_frame, text="n (Рост населения):").grid(row=row, column=0, sticky='e')
        ttk.Entry(self.param_frame, textvariable=self.params['n'], width=8).grid(row=row, column=1)
        row += 1

        ttk.Label(self.param_frame, text="g (Тех. прогресс):").grid(row=row, column=0, sticky='e')
        ttk.Entry(self.param_frame, textvariable=self.params['g'], width=8).grid(row=row, column=1)
        row += 1

        if model == 'Solow':
            ttk.Label(self.param_frame, text="s (Норма сбережений):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['s'], width=8).grid(row=row, column=1)
        elif model == 'RCK':
            ttk.Label(self.param_frame, text="ρ (Временные предпочтения):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['rho'], width=8).grid(row=row, column=1)
            row += 1

            ttk.Label(self.param_frame, text="θ (Эластичность замещения):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['theta'], width=8).grid(row=row, column=1)
        elif model == 'AK':
            ttk.Label(self.param_frame, text="s (Норма сбережений):").grid(row=row, column=0, sticky='e')
            ttk.Entry(self.param_frame, textvariable=self.params['s'], width=8).grid(row=row, column=1)

    def run_model(self):

        model_name = self.model_var.get()
        model = self.models[model_name]
        if model_name == 'Solow':


            params = {k: v.get() for k, v in self.params.items()}
            k0s = [params[f'k0_{i + 1}'] for i in range(4)]

            # Собираем все траектории
            t_values = []
            k_values = []
            k_star = model.steady_state(**params)

            for k0 in k0s:
                t, k = model.simulate(k0, **params)
                t_values.append(t)
                k_values.append(k)

            # Рисуем все на одном графике
            model.plot_results(t_values, k_values, k_star, model_name)

            model.phase_diagram(k0_list=k0s)
        else:
            params = {k: v.get() for k, v in self.params.items()}
            k0s = [params[f'k0_{i + 1}'] for i in range(4)]
            model.simulate(k0s[0], **params)
          #  model.make_3d_graph()
            model.phase_diagram()